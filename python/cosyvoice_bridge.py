"""
Qwen3-TTS Bridge — Python side for the embedded C++ TTS engine.

The C++ code still imports this module as ``cosyvoice_bridge`` to keep the
existing ABI stable. Internally it uses MLX-Audio with Qwen3-TTS on Apple
Silicon and returns int16 mono PCM chunks for AudioQueue playback.
"""

import logging
import os
import sys
import traceback
import warnings

import numpy as np

logging.getLogger("matplotlib").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .*")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except Exception:
    pass

_model = None
_initialized = False
_model_id = ""
_voice = "Vivian"
_language = "Chinese"
_speed = 1.0
_streaming_interval = 0.16
_source_sample_rate = 24000
_output_sample_rate = 24000
_max_audio_sec = 8.0

_speaker_map = {
    "中文女": "Vivian",
    "中文男": "Eric",
    "女声": "Vivian",
    "男声": "Eric",
    "default": "Vivian",
    "english_male": "Ryan",
    "english_female": "Serena",
}


def init(model_dir: str) -> str:
    """
    Load Qwen3-TTS once and keep it resident in memory.

    ``model_dir`` can be a Hugging Face model id or a local MLX model path.
    Environment overrides:
      QWEN_TTS_MODEL, QWEN_TTS_VOICE, QWEN_TTS_LANGUAGE,
      QWEN_TTS_STREAMING_INTERVAL, QWEN_TTS_SPEED, TTS_SAMPLE_RATE.
    """
    global _model, _initialized, _model_id, _voice, _language, _speed
    global _streaming_interval, _source_sample_rate, _output_sample_rate, _max_audio_sec

    if _initialized:
        return "ok"

    _model_id = (
        os.environ.get("QWEN_TTS_MODEL")
        or os.environ.get("TTS_MODEL_DIR")
        or model_dir
        or "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    )
    # ─── 发音人固定为女声 Vivian ───
    # 忽略 QWEN_TTS_VOICE / TTS_VOICE 环境变量，确保输出语音始终为同一女声。
    _voice = "Vivian"
    _language = os.environ.get("QWEN_TTS_LANGUAGE", "Chinese")
    _speed = float(os.environ.get("QWEN_TTS_SPEED", os.environ.get("TTS_SPEED", "1.0")))
    _streaming_interval = float(os.environ.get("QWEN_TTS_STREAMING_INTERVAL", "0.16"))
    _source_sample_rate = int(os.environ.get("QWEN_TTS_SOURCE_SAMPLE_RATE", "24000"))
    _output_sample_rate = int(os.environ.get("TTS_SAMPLE_RATE", str(_source_sample_rate)))
    _max_audio_sec = float(os.environ.get("QWEN_TTS_MAX_AUDIO_SEC", "8.0"))

    try:
        from mlx_audio.tts.utils import load_model
    except Exception as exc:
        raise RuntimeError(
            "Failed to import mlx_audio. Install it in the embedded Python env, "
            "for example: pip install mlx-audio"
        ) from exc

    print(f"[qwen_tts_bridge] Loading model: {_model_id}")
    print(
        "[qwen_tts_bridge] voice=%s language=%s speed=%.2f stream_interval=%.2fs sample_rate=%d max_audio=%.1fs"
        % (_voice, _language, _speed, _streaming_interval, _output_sample_rate, _max_audio_sec)
    )

    _model = load_model(_model_id)
    _initialized = True

    if os.environ.get("QWEN_TTS_WARMUP", "1") != "0":
        warmup_text = os.environ.get("QWEN_TTS_WARMUP_TEXT", "你好。")
        try:
            for _ in _generate(warmup_text, _voice, stream=True):
                break
            print("[qwen_tts_bridge] Warmup complete")
        except Exception as exc:
            print(f"[qwen_tts_bridge] Warmup skipped: {exc}")

    print("[qwen_tts_bridge] Model loaded successfully")
    return "ok"


def synthesize(text: str, spk_id: str = "中文女", mode: str = "sft") -> np.ndarray:
    """Synthesize full text and return int16 mono PCM."""
    del mode
    if not _ensure_ready():
        return np.array([], dtype=np.int16)

    try:
        chunks = []
        max_samples = int(_max_audio_sec * _output_sample_rate)
        total_samples = 0
        for result in _generate(text, _resolve_voice(spk_id), stream=False):
            pcm = _result_to_pcm(result)
            if pcm.size > 0:
                if total_samples + pcm.size > max_samples:
                    pcm = pcm[: max(0, max_samples - total_samples)]
                chunks.append(pcm)
                total_samples += pcm.size
                if total_samples >= max_samples:
                    print(f"[qwen_tts_bridge] Reached max audio length {_max_audio_sec:.1f}s")
                    break
        if not chunks:
            return np.array([], dtype=np.int16)
        return np.concatenate(chunks).astype(np.int16, copy=False)
    except Exception as exc:
        print(f"[qwen_tts_bridge] ERROR during synthesis: {exc}")
        traceback.print_exc()
        return np.array([], dtype=np.int16)


def synthesize_stream(text: str, spk_id: str = "中文女", mode: str = "sft"):
    """Stream synthesis and yield int16 mono PCM chunks as soon as they arrive."""
    del mode
    if not _ensure_ready():
        return

    try:
        voice = _resolve_voice(spk_id)
        max_samples = int(_max_audio_sec * _output_sample_rate)
        total_samples = 0
        for result in _generate(text, voice, stream=True):
            pcm = _result_to_pcm(result)
            if pcm.size > 0:
                if total_samples + pcm.size > max_samples:
                    pcm = pcm[: max(0, max_samples - total_samples)]
                yield pcm
                total_samples += pcm.size
                if total_samples >= max_samples:
                    print(f"[qwen_tts_bridge] Reached max audio length {_max_audio_sec:.1f}s")
                    break
    except Exception as exc:
        print(f"[qwen_tts_bridge] ERROR during stream synthesis: {exc}")
        traceback.print_exc()
        return


def _ensure_ready() -> bool:
    if _model is None:
        print("[qwen_tts_bridge] ERROR: Model not initialized! Call init() first.")
        return False
    return True


def _resolve_voice(spk_id: str) -> str:
    # 固定女声：忽略所有传入的 spk_id 和环境变量，始终返回 preset 的女声。
    # ASR/LLM 侧的任何发音人标签都无法切换 TTS 音色。
    return "Vivian"


def _generate(text: str, voice: str, stream: bool):
    """Call MLX-Audio while tolerating small API differences between releases."""
    base_kwargs = {
        "voice": voice,
        "stream": stream,
    }
    if _speed > 0:
        base_kwargs["speed"] = _speed
    if stream:
        base_kwargs["streaming_interval"] = _streaming_interval

    variants = []
    for kwargs in (base_kwargs, {k: v for k, v in base_kwargs.items() if k != "speed"}):
        if _language:
            variants.append({**kwargs, "language": _language})
            variants.append({**kwargs, "lang_code": _language})
        variants.append(kwargs)

    last_error = None
    for kwargs in variants:
        try:
            return _model.generate(text, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    raise last_error


def _result_to_pcm(result) -> np.ndarray:
    audio = getattr(result, "audio", result)
    source_sr = int(
        getattr(result, "sample_rate", 0)
        or getattr(result, "sampling_rate", 0)
        or _source_sample_rate
    )

    raw = np.asarray(audio).squeeze()
    if raw.size == 0:
        return np.array([], dtype=np.int16)

    if np.issubdtype(raw.dtype, np.integer):
        wav = raw.astype(np.float32) / 32768.0
    else:
        wav = raw.astype(np.float32)

    if source_sr != _output_sample_rate:
        wav = _resample(wav, source_sr, _output_sample_rate)

    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype(np.int16)


def _resample(wav: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr <= 0 or target_sr <= 0 or source_sr == target_sr:
        return wav

    try:
        from scipy.signal import resample_poly
        from math import gcd

        factor = gcd(source_sr, target_sr)
        return resample_poly(wav, target_sr // factor, source_sr // factor).astype(np.float32)
    except Exception:
        duration = wav.shape[0] / float(source_sr)
        target_len = max(1, int(round(duration * target_sr)))
        old_x = np.linspace(0.0, 1.0, wav.shape[0], endpoint=False)
        new_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
        return np.interp(new_x, old_x, wav).astype(np.float32)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    init(model)
    text = sys.argv[2] if len(sys.argv) > 2 else "你好，我是本地语音助手。"
    total = 0
    for idx, chunk in enumerate(synthesize_stream(text, "中文女"), 1):
        total += len(chunk)
        print(f"chunk #{idx}: {len(chunk)} samples")
    print(f"total: {total} samples")

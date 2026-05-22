"""
Qwen3-TTS Bridge — Python side for the embedded C++ TTS engine.

The C++ code still imports this module as ``cosyvoice_bridge`` to keep the
existing ABI stable. Internally it uses MLX-Audio with Qwen3-TTS on Apple
Silicon and returns int16 mono PCM chunks for AudioQueue playback.
"""

import logging
import os
import subprocess
import sys
import tempfile
import traceback
import warnings
import wave
from pathlib import Path

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
_backend = "macos"
_macos_voice = "Tingting"
_macos_rate = 250
_language = "Chinese"
_speed = 1.0
_streaming_interval = 0.16
_temperature = 0.9
_top_k = 20
_top_p = 1.0
_repetition_penalty = 1.05
_seed = 42
_max_tokens = 128
_normalize_peak = 0.85
_use_ref_audio = False
_ref_audio = ""
_ref_text = "你好，我是你的语音助手。今天天气不错，很高兴为你服务。"
_source_sample_rate = 24000
_output_sample_rate = 24000
_max_audio_sec = 8.0

# 当前 Qwen3-TTS Base 模型没有内置 speaker 表，传 voice 名称可能不会严格生效。
# 但 ref_audio 在 embedded Python 路径里会显著拖慢甚至卡住，所以默认先保证稳定出声。
_fixed_voice = "Vivian"


def init(model_dir: str) -> str:
    """
    Initialize the local TTS backend.

    Default backend is macOS Speech Synthesis for a stable, fixed speaker and
    fast short replies. Set TTS_BACKEND=qwen to use Qwen3-TTS.

    ``model_dir`` can be a Hugging Face model id or a local MLX model path.
    Environment overrides:
      TTS_BACKEND, MACOS_TTS_VOICE, MACOS_TTS_RATE,
      QWEN_TTS_MODEL, QWEN_TTS_LANGUAGE, QWEN_TTS_STREAMING_INTERVAL,
      QWEN_TTS_SPEED, QWEN_TTS_TEMPERATURE, QWEN_TTS_TOP_K,
      QWEN_TTS_TOP_P, QWEN_TTS_REPETITION_PENALTY, QWEN_TTS_SEED,
      QWEN_TTS_USE_REF_AUDIO, QWEN_TTS_REF_AUDIO, QWEN_TTS_REF_TEXT,
      TTS_SAMPLE_RATE.
    """
    global _model, _initialized, _model_id, _backend, _macos_voice, _macos_rate, _language, _speed
    global _streaming_interval, _temperature, _top_k, _top_p, _repetition_penalty, _seed, _max_tokens
    global _normalize_peak, _use_ref_audio, _ref_audio, _ref_text
    global _source_sample_rate, _output_sample_rate, _max_audio_sec

    if _initialized:
        return "ok"

    _backend = os.environ.get("TTS_BACKEND", "macos").strip().lower()
    if _backend in ("apple", "system", "say"):
        _backend = "macos"
    if _backend not in ("macos", "qwen"):
        print(f"[tts_bridge] WARNING: unsupported TTS_BACKEND={_backend}, falling back to macos")
        _backend = "macos"

    _macos_voice = os.environ.get("MACOS_TTS_VOICE", os.environ.get("TTS_VOICE", "Tingting"))
    _macos_rate = int(float(os.environ.get("MACOS_TTS_RATE", os.environ.get("TTS_RATE", "250"))))
    _model_id = (
        os.environ.get("QWEN_TTS_MODEL")
        or os.environ.get("TTS_MODEL_DIR")
        or model_dir
        or "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    )
    _language = os.environ.get("QWEN_TTS_LANGUAGE", "Chinese")
    _speed = float(os.environ.get("QWEN_TTS_SPEED", os.environ.get("TTS_SPEED", "1.0")))
    _streaming_interval = float(os.environ.get("QWEN_TTS_STREAMING_INTERVAL", "0.16"))
    _temperature = float(os.environ.get("QWEN_TTS_TEMPERATURE", "0.9"))
    _top_k = int(os.environ.get("QWEN_TTS_TOP_K", "20"))
    _top_p = float(os.environ.get("QWEN_TTS_TOP_P", "1.0"))
    _repetition_penalty = float(os.environ.get("QWEN_TTS_REPETITION_PENALTY", "1.05"))
    _seed = int(os.environ.get("QWEN_TTS_SEED", "42"))
    _source_sample_rate = int(os.environ.get("QWEN_TTS_SOURCE_SAMPLE_RATE", "24000"))
    _output_sample_rate = int(os.environ.get("TTS_SAMPLE_RATE", str(_source_sample_rate)))
    _max_audio_sec = float(os.environ.get("QWEN_TTS_MAX_AUDIO_SEC", "8.0"))
    default_max_tokens = max(32, int(_max_audio_sec * 16) + 16)
    _max_tokens = int(os.environ.get("QWEN_TTS_MAX_TOKENS", str(default_max_tokens)))
    _normalize_peak = float(os.environ.get("QWEN_TTS_NORMALIZE_PEAK", "0.85"))
    _use_ref_audio = os.environ.get("QWEN_TTS_USE_REF_AUDIO", "0") == "1"
    default_ref_audio = Path(__file__).resolve().parent / "assets" / "qwen_tts_female_ref.wav"
    _ref_audio = os.environ.get("QWEN_TTS_REF_AUDIO", str(default_ref_audio))
    _ref_text = os.environ.get("QWEN_TTS_REF_TEXT", _ref_text)
    if _use_ref_audio and (not _ref_audio or not os.path.exists(_ref_audio)):
        print(f"[qwen_tts_bridge] WARNING: reference audio not found: {_ref_audio}")
        _ref_audio = ""

    if _backend == "macos":
        print(
            "[tts_bridge] backend=macos voice=%s rate=%d sample_rate=%d stream_interval=%.2fs"
            % (_macos_voice, _macos_rate, _output_sample_rate, _streaming_interval)
        )
        _initialized = True
        return "ok"

    try:
        from mlx_audio.tts.utils import load_model
    except Exception as exc:
        raise RuntimeError(
            "Failed to import mlx_audio. Install it in the embedded Python env, "
            "for example: pip install mlx-audio"
        ) from exc

    print(f"[qwen_tts_bridge] Loading model: {_model_id}")
    print(
        "[qwen_tts_bridge] voice=%s voice_ref=%s language=%s speed=%.2f temp=%.2f top_k=%d top_p=%.2f seed=%d max_tokens=%d normalize_peak=%.2f stream_interval=%.2fs sample_rate=%d max_audio=%.1fs"
        % (
            _fixed_voice,
            _ref_audio if _use_ref_audio and _ref_audio else "disabled",
            _language,
            _speed,
            _temperature,
            _top_k,
            _top_p,
            _seed,
            _max_tokens,
            _normalize_peak,
            _streaming_interval,
            _output_sample_rate,
            _max_audio_sec,
        )
    )

    _model = load_model(_model_id)
    _initialized = True

    if os.environ.get("QWEN_TTS_WARMUP", "1") != "0":
        warmup_text = os.environ.get("QWEN_TTS_WARMUP_TEXT", "你好。")
        try:
            for _ in _generate(warmup_text, _fixed_voice, stream=True):
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

    if _backend == "macos":
        return _macos_synthesize_pcm(text)

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

    if _backend == "macos":
        pcm = _macos_synthesize_pcm(text)
        chunk_samples = max(1, int(_streaming_interval * _output_sample_rate))
        for start in range(0, pcm.size, chunk_samples):
            yield pcm[start:start + chunk_samples]
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
    if _backend == "macos":
        return _initialized
    if _model is None:
        print("[qwen_tts_bridge] ERROR: Model not initialized! Call init() first.")
        return False
    return True


def _macos_synthesize_pcm(text: str) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        return np.array([], dtype=np.int16)

    aiff_path = ""
    wav_path = ""
    try:
        fd, aiff_path = tempfile.mkstemp(prefix="verbot_tts_", suffix=".aiff")
        os.close(fd)
        fd, wav_path = tempfile.mkstemp(prefix="verbot_tts_", suffix=".wav")
        os.close(fd)

        subprocess.run(
            ["say", "-v", _macos_voice, "-r", str(_macos_rate), "-o", aiff_path, text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{_output_sample_rate}", aiff_path, wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
        )

        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if width != 2:
            print(f"[tts_bridge] Unsupported macOS TTS sample width: {width}")
            return np.array([], dtype=np.int16)

        pcm = np.frombuffer(frames, dtype="<i2").astype(np.int16, copy=True)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return pcm
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        print(f"[tts_bridge] macOS TTS failed: {stderr.strip()}")
        return np.array([], dtype=np.int16)
    except Exception as exc:
        print(f"[tts_bridge] macOS TTS error: {exc}")
        traceback.print_exc()
        return np.array([], dtype=np.int16)
    finally:
        for path in (aiff_path, wav_path):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


def _resolve_voice(spk_id: str) -> str:
    """Return the fixed voice; ASR/LLM speaker labels must not switch timbre."""
    del spk_id
    return _fixed_voice


def _generate(text: str, voice: str, stream: bool):
    """Call MLX-Audio while tolerating small API differences between releases."""
    base_kwargs = {
        "voice": voice,
        "stream": stream,
        "temperature": _temperature,
        "top_k": _top_k,
        "top_p": _top_p,
        "repetition_penalty": _repetition_penalty,
        "max_tokens": _max_tokens,
    }
    if _speed > 0:
        base_kwargs["speed"] = _speed
    if stream:
        base_kwargs["streaming_interval"] = _streaming_interval
    if _use_ref_audio and _ref_audio and _ref_text:
        base_kwargs["ref_audio"] = _ref_audio
        base_kwargs["ref_text"] = _ref_text

    variants = []
    for kwargs in (base_kwargs, {k: v for k, v in base_kwargs.items() if k != "speed"}):
        if _language:
            variants.append({**kwargs, "lang_code": _language})
            variants.append({**kwargs, "language": _language})
        variants.append(kwargs)

    last_error = None
    for kwargs in variants:
        try:
            _reset_random_seed()
            return _model.generate(text, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    raise last_error


def _reset_random_seed() -> None:
    try:
        import random

        random.seed(_seed)
        np.random.seed(_seed)
    except Exception:
        pass

    # Do not call mx.random.seed() here. In embedded C++ worker threads MLX may
    # not have a thread-local GPU stream yet, and touching mx.random can leave
    # the later mx.eval() without a current stream. We reduce variation with a
    # fixed voice and bounded sampling parameters instead.

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
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if _normalize_peak > 0.0 and 0.001 < peak < _normalize_peak:
        wav = np.clip(wav * min(_normalize_peak / peak, 8.0), -1.0, 1.0)
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

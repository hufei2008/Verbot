"""
CosyVoice TTS Bridge — Python 侧脚本，供 C++ 嵌入式 Python 解释器调用。

使用方式（C++ 中通过 PyImport_Exec 执行）：
  >>> import cosyvoice_bridge
  >>> cosyvoice_bridge.init("/path/to/model_dir")
  >>> pcm = cosyvoice_bridge.synthesize("你好世界", "中文女")
  >>> pcm  # numpy.ndarray, int16, 22050Hz

设计原则：
- 进程启动时调用 init() 加载模型，常驻内存
- synthesize() 返回 numpy int16 数组，C++ 侧通过 PyArray_DATA 直接读取
- 线程安全：通过 GIL 保护
"""

import os
import sys
import numpy as np
import logging

# 抑制非关键日志
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# 全局模型实例
_cosyvoice = None
_initialized = False
_tts_speed = 1.25
_stream_min_hop = 18
_stream_overlap = 8


def init(model_dir: str) -> str:
    """
    加载 CosyVoice 模型，只调用一次。
    
    Args:
        model_dir: 模型路径（本地路径或 ModelScope repo id）
    
    Returns:
        成功返回 "ok"，失败抛出异常
    """
    global _cosyvoice, _initialized, _tts_speed, _stream_min_hop, _stream_overlap
    
    if _initialized:
        return "ok"
    
    # 添加 CosyVoice 源码路径
    _add_cosyvoice_paths()
    _tts_speed = float(os.environ.get("COSYVOICE_TTS_SPEED", "1.25"))
    _stream_min_hop = int(os.environ.get("COSYVOICE_STREAM_MIN_HOP", "18"))
    _stream_overlap = int(os.environ.get("COSYVOICE_STREAM_OVERLAP", "8"))
    
    from cosyvoice.cli.cosyvoice import AutoModel
    
    print(f"[cosyvoice_bridge] Loading model from: {model_dir}")
    _cosyvoice = AutoModel(model_dir=model_dir)
    _configure_streaming_latency()
    _initialized = True
    
    # 检测可用的说话人列表
    import torch as _torch
    _spk2info_path = os.path.join(model_dir, 'spk2info.pt')
    global _available_spks
    _available_spks = ['中文女', '中文男', 'default']
    if os.path.exists(_spk2info_path):
        _spk_data = _torch.load(_spk2info_path, map_location='cpu')
        if isinstance(_spk_data, dict) and list(_spk_data.keys()) == ['default']:
            _available_spks = ['default']
        else:
            _available_spks = list(_spk_data.keys())
    print(f"[cosyvoice_bridge] Model loaded successfully! Available speakers: {_available_spks}, "
          f"speed={_tts_speed}, stream_min_hop={_stream_min_hop}, stream_overlap={_stream_overlap}")
    return "ok"


def synthesize(text: str, spk_id: str = "中文女", mode: str = "sft") -> np.ndarray:
    """
    将文本合成为语音。
    
    Args:
        text:    要合成的文本
        spk_id:  说话人 ID（如 "中文女", "中文男"）
        mode:    模式: "sft" | "zero_shot" | "instruct"
    
    Returns:
        numpy.ndarray, dtype=int16, shape=(samples,), 22050Hz
        如果失败返回空数组
    """
    global _cosyvoice
    
    if _cosyvoice is None:
        print("[cosyvoice_bridge] ERROR: Model not initialized! Call init() first.")
        return np.array([], dtype=np.int16)
    
    # CosyVoice3 要求文本包含 <|endofprompt|>
    text = _maybe_add_endofprompt(text)
    
    # 如果请求的 spk_id 不存在，尝试 fallback
    effective_spk_id = spk_id
    if spk_id not in _available_spks and 'default' in _available_spks:
        print(f"[cosyvoice_bridge] Speaker '{spk_id}' not available, falling back to 'default'")
        effective_spk_id = 'default'
    
    try:
        if mode == "sft":
            model_output = _cosyvoice.inference_sft(text, effective_spk_id, speed=_tts_speed)
        elif mode == "zero_shot":
            model_output = _cosyvoice.inference_zero_shot(text, effective_spk_id)
        elif mode == "instruct":
            model_output = _cosyvoice.inference_instruct(text, effective_spk_id, text)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 收集所有音频帧
        all_chunks = []
        for i in model_output:
            speech = i['tts_speech'].numpy()
            all_chunks.append(speech)
        
        if not all_chunks:
            return np.array([], dtype=np.int16)
        
        # 拼接并转换为 int16
        # CosyVoice3 输出 shape: (1, audio_len)
        # CosyVoice 2 输出 shape: (audio_len,)
        speech_full = np.concatenate(all_chunks, axis=-1).squeeze()
        pcm_int16 = (speech_full * (2 ** 15)).astype(np.int16)
        
        return pcm_int16
    
    except Exception as e:
        print(f"[cosyvoice_bridge] ERROR during synthesis: {e}")
        import traceback
        traceback.print_exc()
        return np.array([], dtype=np.int16)


def synthesize_stream(text: str, spk_id: str = "中文女", mode: str = "sft"):
    """
    流式合成：每生成一段音频就 yield 一个 int16 numpy.ndarray。
    """
    global _cosyvoice

    if _cosyvoice is None:
        print("[cosyvoice_bridge] ERROR: Model not initialized! Call init() first.")
        return

    text = _maybe_add_endofprompt(text)

    effective_spk_id = spk_id
    if spk_id not in _available_spks and 'default' in _available_spks:
        print(f"[cosyvoice_bridge] Speaker '{spk_id}' not available, falling back to 'default'")
        effective_spk_id = 'default'

    try:
        if mode == "sft":
            model_output = _cosyvoice.inference_sft(
                text, effective_spk_id, stream=True, speed=_tts_speed
            )
        elif mode == "zero_shot":
            model_output = _cosyvoice.inference_zero_shot(
                text, effective_spk_id, stream=True, speed=_tts_speed
            )
        elif mode == "instruct":
            model_output = _cosyvoice.inference_instruct(
                text, effective_spk_id, text, stream=True, speed=_tts_speed
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for i in model_output:
            speech = i['tts_speech'].numpy().squeeze()
            pcm_int16 = (speech * (2 ** 15)).astype(np.int16)
            if pcm_int16.size > 0:
                yield pcm_int16

    except Exception as e:
        print(f"[cosyvoice_bridge] ERROR during stream synthesis: {e}")
        import traceback
        traceback.print_exc()
        return


def _maybe_add_endofprompt(text: str) -> str:
    """CosyVoice3 要求输入文本中包含 <|endofprompt|> 标记。"""
    if _cosyvoice is not None:
        class_name = _cosyvoice.__class__.__name__
        if 'CosyVoice3' in class_name:
            marker = '<|endofprompt|>'
            if marker not in text:
                return marker + ' ' + text
    return text


def _configure_streaming_latency():
    """Tune CosyVoice stream chunk size at runtime to reduce first-audio latency."""
    model = getattr(_cosyvoice, "model", None)
    if model is None:
        return

    if hasattr(model, "token_min_hop_len"):
        old_min = model.token_min_hop_len
        old_max = getattr(model, "token_max_hop_len", old_min)
        old_overlap = getattr(model, "token_overlap_len", _stream_overlap)

        model.token_min_hop_len = max(8, _stream_min_hop)
        model.token_max_hop_len = max(model.token_min_hop_len, min(old_max, model.token_min_hop_len * 3))
        model.token_overlap_len = max(4, _stream_overlap)

        if hasattr(model, "flow") and getattr(model.flow, "input_frame_rate", 0):
            model.mel_overlap_len = max(
                1,
                int(model.token_overlap_len / model.flow.input_frame_rate * 22050 / 256)
            )
            model.mel_window = np.hamming(2 * model.mel_overlap_len)

        if hasattr(model, "stream_scale_factor"):
            model.stream_scale_factor = 1

        print("[cosyvoice_bridge] Stream latency tuned: "
              f"token_min_hop_len {old_min}->{model.token_min_hop_len}, "
              f"token_max_hop_len {old_max}->{model.token_max_hop_len}, "
              f"token_overlap_len {old_overlap}->{model.token_overlap_len}")
        return

    if hasattr(model, "token_hop_len"):
        old_hop = model.token_hop_len
        old_max = getattr(model, "token_max_hop_len", old_hop)
        model.token_hop_len = max(8, _stream_min_hop)
        model.token_max_hop_len = max(model.token_hop_len, min(old_max, model.token_hop_len * 3))
        if hasattr(model, "stream_scale_factor"):
            old_scale = model.stream_scale_factor
            model.stream_scale_factor = 1
        else:
            old_scale = "n/a"
        print("[cosyvoice_bridge] Stream latency tuned: "
              f"token_hop_len {old_hop}->{model.token_hop_len}, "
              f"token_max_hop_len {old_max}->{model.token_max_hop_len}, "
              f"stream_scale_factor {old_scale}->{getattr(model, 'stream_scale_factor', 'n/a')}")


def _add_cosyvoice_paths():
    """添加 CosyVoice 源码到 sys.path。"""
    # 尝试多个可能的路径
    candidates = [
        # 相对于当前脚本的位置
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '../../Documents/Codex/2026-05-20/gemma4/CosyVoice'),
        # 绝对路径
        '/Users/boby/Documents/Codex/2026-05-20/gemma4/CosyVoice',
    ]
    
    for path in candidates:
        resolved = os.path.abspath(path)
        if os.path.isdir(resolved):
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            # 添加 third_party/Matcha-TTS
            ttp = os.path.join(resolved, 'third_party', 'Matcha-TTS')
            if os.path.isdir(ttp) and ttp not in sys.path:
                sys.path.insert(0, ttp)
            return
    
    raise RuntimeError(
        f"Cannot find CosyVoice source directory. Tried: {candidates}"
    )


# ============================================================
# 测试（python -m cosyvoice_bridge）
# ============================================================
if __name__ == '__main__':
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    
    print(f"Initializing with model_dir={model_dir}...")
    init(model_dir)
    
    test_text = "你好，今天天气真不错。"
    print(f"Synthesizing: '{test_text}'")
    pcm = synthesize(test_text, "中文女")
    print(f"Got PCM: {len(pcm)} samples, dtype={pcm.dtype}")
    
    # 保存到文件验证
    output_path = "/tmp/test_bridge.pcm"
    pcm.tofile(output_path)
    print(f"Saved to {output_path}")

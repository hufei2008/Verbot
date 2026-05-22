#!/usr/bin/env python3
"""
TTS 发音人映射测试脚本

两层验证：
  1. 单元测试：验证 _resolve_voice() 映射逻辑（无需加载模型）
  2. 集成测试：实际用不同 spk_id 合成音频，保存为 WAV 文件（需加载模型）

用法：
  # 只跑单元测试（快，无需模型）
  python python/test_speaker.py --unit

  # 跑完整测试（需要加载模型，需等待）
  python python/test_speaker.py --full

  # 查看发音人映射表
  python python/test_speaker.py --list
"""

import argparse
import os
import sys
import struct
import wave

# 确保能找到 cosyvoice_bridge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))


def test_unit():
    """
    单元测试：直接测试 _resolve_voice() 映射逻辑。
    不需要加载模型，纯 Python 逻辑，秒出结果。
    """
    from cosyvoice_bridge import _resolve_voice, _speaker_map

    # ─── 测试用例 ───
    cases = [
        # (输入, 期望输出, 描述)
        ("中文女",   "Vivian",  "中文女 → Vivian"),
        ("中文男",   "Eric",    "中文男 → Eric"),
        ("女声",     "Vivian",  "女声 → Vivian"),
        ("男声",     "Eric",    "男声 → Eric"),
        ("Vivian",   "Vivian",  "Vivian → Vivian (identity)"),
        ("Eric",     "Eric",    "Eric → Eric (identity)"),
        ("vivian",   "Vivian",  "vivian → Vivian (大小写不敏感)"),
        ("eric",     "Eric",    "eric → Eric (大小写不敏感)"),
        ("default",  "Vivian",  "default → Vivian"),
        ("Ryan",     "Ryan",    "Ryan → Ryan (英文男声)"),
        ("Serena",   "Serena",  "Serena → Serena (英文女声)"),
        ("english_male",  "Ryan",   "english_male → Ryan"),
        ("english_female", "Serena", "english_female → Serena"),
        ("女",       "Vivian",  "女 → Vivian (找不到回退)"),
        ("xxx",      "Vivian",  "xxx → Vivian (找不到回退)"),
        ("",         "Vivian",  "'' → Vivian (空字符串回退)"),
    ]

    passed = 0
    failed = 0
    print("=" * 60)
    print("  发音人映射单元测试")
    print("=" * 60)
    print(f"  映射表共 {len(_speaker_map)} 项:")
    for k, v in _speaker_map.items():
        print(f"    {k!r:>20} → {v}")
    print()

    for spk_id, expected, desc in cases:
        actual = _resolve_voice(spk_id)
        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"  [{status:4s}] {desc:45s} → {actual!r:>8}  (期望 {expected!r})")

    print()
    print(f"  结果: {passed}/{passed+failed} 通过", end="")
    if failed > 0:
        print(f", {failed} 失败 ❌")
    else:
        print(" ✅")

    return failed == 0


def _pcm_to_wav(pcm: bytes, sample_rate: int, filepath: str):
    """将 int16 PCM 数据写入 WAV 文件。"""
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def test_full():
    """
    完整集成测试：加载模型并用不同发音人合成同一句话。
    将结果保存为 WAV 文件，方便用户对比试听。
    """
    model_dir = os.environ.get(
        "QWEN_TTS_MODEL",
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    )
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    from cosyvoice_bridge import init, synthesize, _speaker_map

    print("=" * 60)
    print("  发音人切换集成测试")
    print("=" * 60)
    print(f"  模型: {model_dir}")
    print(f"  输出: {output_dir}/")
    print()

    # ── Step 1: 初始化模型 ──
    print("  [*] 加载模型中（首次需要下载，请耐心等待）...")
    init(model_dir)
    print("  [✓] 模型加载完成")
    print()

    # ── Step 2: 用多个发音人合成同一句话 ──
    test_text = "你好，我是语音助手。我可以用不同的声音说话。"

    # 要测试的 spk_id 列表
    test_spk_ids = [
        "中文女",   # → Vivian
        "中文男",   # → Eric
        "Vivian",   # → Vivian (直通)
        "Eric",     # → Eric (直通)
    ]

    generated_files = []

    for spk_id in test_spk_ids:
        # 告诉用户实际会用什么 voice
        from cosyvoice_bridge import _resolve_voice
        actual_voice = _resolve_voice(spk_id)

        print(f"  [*] Synthesize with spk_id={spk_id!r} → voice={actual_voice!r}")

        pcm_array = synthesize(test_text, spk_id, mode="sft")

        if pcm_array is None or pcm_array.size == 0:
            print(f"  [✗] 合成失败！")
            continue

        # 保存为 WAV
        sample_rate = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
        filename = f"tts_speaker_test_{spk_id}_{actual_voice}.wav"
        filepath = os.path.join(output_dir, filename)
        _pcm_to_wav(pcm_array.tobytes(), sample_rate, filepath)

        duration = pcm_array.size / sample_rate
        print(f"  [✓] {pcm_array.size:>6} samples, {duration:.2f}s → {filepath}")
        generated_files.append(filepath)

    print()
    print(f"  共生成 {len(generated_files)} 个音频文件:")
    for f in generated_files:
        print(f"    {f}")
    print()
    print("  在不同发音人之间切换后，生成的音频音色应有明显区别。")
    print("  请使用播放器打开各 WAV 文件对比试听。")

    return len(generated_files) > 0


def list_speakers():
    """列出当前支持的发音人映射关系。"""
    from cosyvoice_bridge import _speaker_map

    print("=" * 60)
    print("  TTS 发音人映射表")
    print("=" * 60)
    print(f"  共 {len(_speaker_map)} 项映射:\n")

    # 按值分组显示
    by_voice = {}
    for key, val in _speaker_map.items():
        by_voice.setdefault(val, []).append(key)

    for voice, aliases in sorted(by_voice.items()):
        print(f"  MLX voice: {voice}")
        print(f"    别名: {', '.join(aliases)}")
        print()

    print("  调用 synthesize(text, spk_id) 时:")
    print("    - spk_id 会先经过 _speaker_map 查找")
    print("    - 找不到匹配时回退到 'Vivian'")
    print("    - 要切换发音人，修改 python/cosyvoice_bridge.py 中的 _speaker_map")
    print()


def main():
    parser = argparse.ArgumentParser(description="TTS Speaker Test")
    parser.add_argument("--unit", action="store_true", help="仅跑单元测试")
    parser.add_argument("--full", action="store_true", help="跑完整集成测试")
    parser.add_argument("--list", action="store_true", help="列出发音人映射表")
    args = parser.parse_args()

    # 默认：全部执行
    do_unit = args.unit or (not args.full and not args.list)
    do_full = args.full or (not args.unit and not args.list)
    do_list = args.list or (not args.unit and not args.full)

    ok = True

    if do_list:
        list_speakers()
        print()

    if do_unit:
        unit_ok = test_unit()
        ok = ok and unit_ok
        print()

    if do_full:
        full_ok = test_full()
        ok = ok and full_ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
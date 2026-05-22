#!/usr/bin/env python3
"""
TTS 稳定性测试：
用同一 voice 合成同一句话 3 次，检查输出是否一致。

如果 3 次输出差异很大，说明 Qwen3-TTS 模型本身存在
随机性导致的音色不稳定，不是你代码的问题。
"""

import argparse
import os
import sys
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))


def pcm_to_wav(pcm: np.ndarray, sample_rate: int, filepath: str):
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype(np.int16).tobytes())


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个信号的归一化互相关系数 (NCC)，1.0 = 完全一致。"""
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    aa = a - a.mean()
    bb = b - b.mean()
    denom = np.sqrt(np.dot(aa, aa) * np.dot(bb, bb))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def main():
    parser = argparse.ArgumentParser(description="TTS Stability Test")
    parser.add_argument("--model", default=None,
                        help="Model path/ID (default: env QWEN_TTS_MODEL)")
    parser.add_argument("--text", default="你好，我是你的语音助手。请问有什么可以帮你的？",
                        help="Test text")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of runs (default: 3)")
    parser.add_argument("--voice", default="Vivian",
                        help="Voice to test (default: Vivian)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    from cosyvoice_bridge import init, synthesize

    model_dir = args.model or os.environ.get(
        "QWEN_TTS_MODEL",
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    )
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "output"
    )
    os.makedirs(output_dir, exist_ok=True)

    sample_rate = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
    voice = args.voice
    test_text = args.text
    n_runs = max(2, min(args.n, 10))

    print("=" * 60)
    print("  TTS 稳定性测试")
    print("=" * 60)
    print(f"  Voice:      {voice}")
    print(f"  Text:       \"{test_text}\"")
    print(f"  Runs:       {n_runs}")
    print(f"  Model:      {model_dir}")
    print(f"  Output:     {output_dir}/")
    print()

    # 加载模型
    print("  [*] 加载模型...")
    init(model_dir)
    print("  [✓] 模型加载完成")
    print()

    pcm_list = []
    files = []

    for i in range(n_runs):
        print(f"  [*] Run #{i+1}/{n_runs} ...")
        pcm = synthesize(test_text, voice, mode="sft")
        if pcm is None or pcm.size == 0:
            print(f"  [✗] Run #{i+1} 合成失败")
            continue
        pcm_list.append(pcm)

        filename = f"tts_stability_test_{voice}_run{i+1}.wav"
        filepath = os.path.join(output_dir, filename)
        pcm_to_wav(pcm, sample_rate, filepath)
        files.append(filepath)

        duration = pcm.size / sample_rate
        rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        print(f"  [✓] {pcm.size:>6} samples, {duration:.2f}s, RMS={rms:.1f} → {filepath}")

    valid = len(pcm_list)
    if valid < 2:
        print("\n  [FAIL] 有效样本不足，无法比较")
        sys.exit(1)

    # 量化比较
    print()
    print("  ─── 互相关系数对比 ───")
    print()

    results = []
    for i in range(valid):
        for j in range(i + 1, valid):
            ncc = normalized_cross_correlation(pcm_list[i], pcm_list[j])
            results.append((i, j, ncc))
            label = "✅ 稳定" if ncc > 0.85 else ("⚠️ 轻微波动" if ncc > 0.7 else "❌ 不稳定")
            print(f"  Run #{i+1} vs Run #{j+1}:  NCC = {ncc:.4f}  {label}")

    print()
    avg_ncc = np.mean([r[2] for r in results]) if results else 0.0
    print(f"  平均 NCC: {avg_ncc:.4f}")
    print()

    if avg_ncc > 0.85:
        print("  ✅ 结论：模型输出稳定（NCC > 0.85）")
        print("     同一 voice 合成的音频基本一致，听感差异来自其他因素。")
    elif avg_ncc > 0.70:
        print("  ⚠️ 结论：模型存在轻微波动（0.70 < NCC ≤ 0.85）")
        print("     同一 voice 合成的音频有少量差异，可能导致轻度音色变化。")
        print("     这是 Qwen3-TTS 模型自身的随机性，属于正常现象。")
    else:
        print("  ❌ 结论：模型高度不稳定（NCC ≤ 0.70）")
        print("     同一 voice 每次合成的音频差异很大。")
        print("     这可能是模型采样 seed 随机导致的音色漂移。")
        print()
        print("  建议排查方向：")
        print("    - 检查 mlx-audio 版本，确认是否支持固定 seed")
        print("    - 检查 _generate() 是否每次调用重置了模型状态")
        print("    - 可尝试在 _generate 前设置 np.random.seed()")

    print()
    print(f"  生成文件:")
    for f in files:
        print(f"    {f}")


if __name__ == "__main__":
    main()
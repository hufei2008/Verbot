#!/usr/bin/env python3
"""
批量测试：对 test_audio 目录下的 WAV 文件运行 Whisper 识别，计算准确率
"""
import os
import subprocess
import sys
import re
import json
from datetime import datetime

def compute_cer(hypothesis, reference):
    """计算字错率 CER (Character Error Rate)"""
    # 移除空格和标点简化比较
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    # 编辑距离（Levenshtein）
    n, m = len(ref_chars), len(hyp_chars)
    if n == 0 and m == 0:
        return 0.0, 0.0, 0.0
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i-1] == hyp_chars[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost   # substitution
            )
    
    edit_dist = dp[n][m]
    cer = edit_dist / n if n > 0 else 1.0
    # 精确匹配
    exact_match = (ref_chars == hyp_chars)
    return edit_dist, cer, exact_match

def main():
    # 配置 - 转为绝对路径
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # study2 root
    test_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "build", "test_audio")
    binary_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base_dir, "build", "test_whisper")
    model_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(base_dir, "build", "models", "ggml-medium.bin")

    # 验证文件存在
    for name, p in [("Test audio dir", test_dir), ("Binary", binary_path), ("Model", model_path)]:
        if not os.path.exists(p):
            print(f"ERROR: {name} not found: {p}")
            return

    # 读取 ground truth
    gt_path = os.path.join(test_dir, "ground_truth.txt")
    ground_truth = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                case_id, text = line.split("|", 1)
                ground_truth[case_id] = text

    # 收集所有 WAV 文件
    wav_files = sorted([
        f for f in os.listdir(test_dir) if f.endswith(".wav")
    ])
    
    if not wav_files:
        print("No WAV files found!")
        return

    print(f"Found {len(wav_files)} test files in '{test_dir}'")
    print(f"Model: {model_path}")
    print(f"Binary: {binary_path}")
    print("=" * 80)

    # 结果统计
    results = []
    exact_matches = 0
    total_chars_ref = 0
    total_chars_hyp = 0
    total_edit_dist = 0
    total_cer = 0.0
    
    categories = {
        "numbers": {"total": 0, "correct": 0, "cer": 0.0},
        "greet": {"total": 0, "correct": 0, "cer": 0.0},
        "cmd": {"total": 0, "correct": 0, "cer": 0.0},
        "daily": {"total": 0, "correct": 0, "cer": 0.0},
        "name": {"total": 0, "correct": 0, "cer": 0.0},
        "tech": {"total": 0, "correct": 0, "cer": 0.0},
        "long": {"total": 0, "correct": 0, "cer": 0.0},
        "homo": {"total": 0, "correct": 0, "cer": 0.0},
        "bless": {"total": 0, "correct": 0, "cer": 0.0},
        "tongue": {"total": 0, "correct": 0, "cer": 0.0},
    }

    for wav_name in wav_files:
        case_id = wav_name.replace(".wav", "")
        if case_id not in ground_truth:
            print(f"  SKIP: {wav_name} (no ground truth)")
            continue

        wav_path = os.path.join(test_dir, wav_name)
        reference = ground_truth[case_id]

        # 运行 test_whisper
        cmd = [binary_path, model_path, wav_path]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=os.path.dirname(binary_path) if os.path.dirname(binary_path) else None
            )
            output = proc.stdout + proc.stderr
            
            # 识别结果在 "RESULT " 或 "[timestamp --> timestamp]" 之后
            # 提取识别的文本
            hypothesis = ""
            for line in output.split("\n"):
                # 查找时间戳行: [00:00:00 --> 00:00:01] 文本
                match = re.search(r'\[\d{2}:\d{2}:\d{2}.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}.\d{3}\]\s*(.*)', line)
                if match:
                    hypothesis += match.group(1)
                # 也检查是否有 "RESULT" 行
                if "RESULT" in line:
                    # test_whisper 输出格式中有 "=== Transcription Result"
                    pass
            
            # 如果 hypothesis 为空，尝试提取 failed 信息
            if not hypothesis:
                # check for "RESULT" pattern in our custom format (asr_demo)
                for line in output.split("\n"):
                    if "📝 RESULT:" in line:
                        # 提取 RESULT:之后的内容
                        idx = line.find("📝 RESULT:")
                        if idx >= 0:
                            hypothesis = line[idx + len("📝 RESULT:"):].strip()
            
            if not hypothesis:
                # 从所有 segment 行提取
                lines = output.split("\n")
                for i, line in enumerate(lines):
                    if "segment" in line.lower() and "]" in line:
                        # 可能是 [[timestamp --> timestamp]] 格式
                        m = re.search(r'\]\s*[-–—]\s*(.*)', line)
                        if m:
                            hypothesis += m.group(1)
            
            if not hypothesis:
                if "failed" in output.lower():
                    hypothesis = "[FAILED]"
                else:
                    # 尝试从 stdout 抓取任何中文文本
                    cn_chars = re.findall(r'[\u4e00-\u9fff]+', output)
                    if cn_chars:
                        hypothesis = "".join(cn_chars)
                    else:
                        # 截取输出中最后几行
                        last_lines = output.strip().split("\n")[-3:]
                        hypothesis = f"[NO_MATCH] {last_lines}" if last_lines else "[NO_MATCH]"

        except Exception as e:
            hypothesis = f"[ERROR: {e}]"

        # 计算 CER
        edit_dist, cer, exact = compute_cer(hypothesis, reference)
        
        exact_matches += 1 if exact else 0
        total_edit_dist += edit_dist
        total_chars_ref += len(reference)
        total_chars_hyp += len(hypothesis)
        total_cer += cer

        # 分类统计
        cat = case_id.split("_")[0]
        if cat in categories:
            categories[cat]["total"] += 1
            if exact:
                categories[cat]["correct"] += 1
            categories[cat]["cer"] += cer

        results.append({
            "case_id": case_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "exact": exact,
            "cer": cer,
            "edit_dist": edit_dist
        })

        # 打印单条结果
        status = "✅" if exact else "❌"
        print(f"{status} [{case_id}] ref='{reference}'")
        print(f"   hyp='{hypothesis}' CER={cer:.1%} dist={edit_dist}")

    # ===== 汇总统计 =====
    n = len(results)
    if n == 0:
        print("\nNo results!")
        return

    word_acc = exact_matches / n * 100
    avg_cer = total_cer / n * 100

    print("\n" + "=" * 80)
    print(f"{'=' * 35} 最终结果 {'=' * 35}")
    print("=" * 80)
    print(f"总测试数:     {n}")
    print(f"完全匹配:     {exact_matches}/{n} ({word_acc:.1f}%)")
    print(f"平均 CER:     {avg_cer:.2f}%")
    print(f"总编辑距离:   {total_edit_dist}")
    print(f"参考字数:     {total_chars_ref}")
    print(f"识别字数:     {total_chars_hyp}")
    print()

    # 分类统计
    print("-" * 50)
    print("分类统计:")
    print("-" * 50)
    cat_names = {
        "numbers": "数字",
        "greet": "问候",
        "cmd": "命令",
        "daily": "日常",
        "name": "人名",
        "tech": "科技",
        "long": "长句",
        "homo": "同音词",
        "bless": "吉祥话",
        "tongue": "绕口令",
    }
    for cat_key, cat_name in cat_names.items():
        c = categories[cat_key]
        if c["total"] > 0:
            acc = c["correct"] / c["total"] * 100
            avg_cer_cat = c["cer"] / c["total"] * 100
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"  {cat_name:6s}: {bar} {acc:5.1f}% ({c['correct']}/{c['total']}) CER={avg_cer_cat:.2f}%")

    print()

    # 错误详细分析
    print("-" * 50)
    print("错误详情 (CER > 0):")
    print("-" * 50)
    errors = [r for r in results if r["cer"] > 0]
    errors.sort(key=lambda r: -r["cer"])
    for r in errors[:20]:  # 只显示 TOP 20 错误
        print(f"  ❌ [{r['case_id']}]")
        print(f"     ref: {r['reference']}")
        print(f"     hyp: {r['hypothesis']}")
        print(f"     CER: {r['cer']:.1%}  edit_dist={r['edit_dist']}")
        print()

    # 保存 JSON 报告
    report_path = os.path.join(test_dir, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report = {
        "total": n,
        "exact_matches": exact_matches,
        "word_accuracy_pct": round(word_acc, 2),
        "avg_cer_pct": round(avg_cer, 2),
        "total_edit_distance": total_edit_dist,
        "total_ref_chars": total_chars_ref,
        "total_hyp_chars": total_chars_hyp,
        "categories": {cat_names.get(k, k): {"correct": v["correct"], "total": v["total"], "cer_pct": round(v["cer"]/v["total"]*100, 2) if v["total"] > 0 else 0} for k, v in categories.items()},
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"详细报告已保存: {report_path}")

if __name__ == "__main__":
    main()
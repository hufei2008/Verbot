#include "tts_engine.h"
#include "audio_player.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <unistd.h>

// ============================================================
// test_tts — TTS 语音合成独立测试程序
//
// 测试流程：
//   Step 0: AudioPlayer 独立播放测试（440Hz 正弦波）
//   Step 1: 初始化 TtsEngine（加载嵌入 Python Qwen3-TTS 模型）
//   Step 2: 同步合成 + 播放 3 段测试文本
//   Step 3: 流式合成测试（分段回调）
//
// 环境变量配置：
//   QWEN_TTS_MODEL       - TTS 模型路径或 HuggingFace ID
//   QWEN_TTS_PYTHON_HOME - conda Python 环境路径
//   QWEN_TTS_BRIDGE_DIR  - cosyvoice_bridge.py 所在目录
//   TTS_SAMPLE_RATE      - 采样率（Hz），默认 24000
// ============================================================

int main(int argc, char** argv) {
    fprintf(stdout, "========================================\n");
    fprintf(stdout, "  TTS Standalone Test\n");
    fprintf(stdout, "========================================\n");

    // 从环境变量读取配置；未设置时使用默认 Qwen3-TTS 0.6B MLX 模型
    const char* model_dir      = std::getenv("QWEN_TTS_MODEL");
    const char* python_home    = std::getenv("QWEN_TTS_PYTHON_HOME");
    const char* bridge_dir     = std::getenv("QWEN_TTS_BRIDGE_DIR");

    if (!model_dir) {
        model_dir = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16";
    }
    if (!python_home) {
        python_home = "/opt/homebrew/Caskroom/miniforge/base/envs/cosyvoice";
    }
    if (!bridge_dir) {
        bridge_dir = "/Users/boby/work/study2/python";
    }

    fprintf(stdout, "[TEST] model_dir:       %s\n", model_dir);
    fprintf(stdout, "[TEST] python_home:     %s\n", python_home);
    fprintf(stdout, "[TEST] bridge_dir:      %s\n", bridge_dir);
    // 从环境变量读取采样率，默认 24000Hz
    int sample_rate = 24000;
    if (const char* env_sample_rate = std::getenv("TTS_SAMPLE_RATE")) {
        int configured_sample_rate = std::atoi(env_sample_rate);
        if (configured_sample_rate > 0) {
            sample_rate = configured_sample_rate;
        }
    }

    // ── Step 0: AudioPlayer 独立播放测试 ──
    // 播放一个 440Hz 纯音（A4 音符），验证音频输出链路正常
    fprintf(stdout, "\n--- Step 0: AudioPlayer playback test ---\n");
    {
        AudioPlayer player;
        if (!player.init(sample_rate)) {
            fprintf(stderr, "[FAIL] AudioPlayer init failed!\n");
            return 1;
        }
        fprintf(stdout, "[PASS] AudioPlayer initialized\n");

        // 生成 440Hz 正弦波（A4 音符），持续 2 秒
        const float duration_sec = 2.0f;
        const float freq_hz = 440.0f;  // A4 音符频率
        const int n_samples = static_cast<int>(sample_rate * duration_sec);
        std::vector<int16_t> tone(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            float t = (float)i / sample_rate;
            float sample = sinf(2.0f * 3.14159f * freq_hz * t);
            // 放大到接近 int16 最大值（32767），确保可听见
            tone[i] = static_cast<int16_t>(sample * 30000.0f);
        }

        fprintf(stdout, "[TEST] Playing 440Hz sine tone for %.1f seconds...\n", duration_sec);
        player.play(tone);
        player.wait_for_finish();
        fprintf(stdout, "[PASS] Sine tone playback finished\n");
    }

    // ── Step 1: 初始化 TTS 引擎 ──
    // 加载嵌入式 Python 解释器 + Qwen3-TTS 模型
    fprintf(stdout, "\n--- Step 1: TtsEngine init ---\n");
    TtsEngine tts;
    if (!tts.init(model_dir, python_home, bridge_dir)) {
        fprintf(stderr, "[FAIL] TtsEngine init failed!\n");
        return 1;
    }
    fprintf(stdout, "[PASS] TtsEngine initialized\n");

    // ── Step 2: 同步合成 + 播放测试 ──
    // 对 3 段测试文本依次进行 TTS 合成，并播放结果
    fprintf(stdout, "\n--- Step 2: TTS synthesis test ---\n");
    const char* test_texts[] = {
        "你好，我是你的语音助手。",
        "今天天气真好。",
        "测试一下语音合成效果。",
    };
    const int num_tests = sizeof(test_texts) / sizeof(test_texts[0]);

    for (int i = 0; i < num_tests; ++i) {
        std::string text = test_texts[i];
        std::vector<int16_t> pcm;  // 存储合成后的 PCM 音频

        fprintf(stdout, "[TEST %d] Synthesizing: \"%s\"\n", i+1, text.c_str());

        // 同步合成：阻塞直到返回完整音频
        bool ok = tts.synthesize_sync(text, pcm, "中文女");
        if (!ok) {
            fprintf(stderr, "[FAIL] Synthesis failed for: %s\n", text.c_str());
            continue;
        }

        if (pcm.empty()) {
            fprintf(stderr, "[FAIL] Synthesis returned empty PCM!\n");
            continue;
        }

        // 打印 PCM 数据的统计信息（验证音频质量）
        int16_t min_val = 32767, max_val = -32768;
        double sum = 0;
        int nonzero = 0;
        for (size_t s = 0; s < pcm.size(); ++s) {
            if (pcm[s] < min_val) min_val = pcm[s];
            if (pcm[s] > max_val) max_val = pcm[s];
            sum += std::abs(pcm[s]);
            if (pcm[s] != 0) nonzero++;
        }
        double avg_abs = sum / pcm.size();  // 平均绝对值幅度

        float duration_ms = (float)pcm.size() * 1000.0f / (float)sample_rate;
        fprintf(stdout, "[PASS] Synthesized %zu samples (%.0f ms, %dHz)\n",
                pcm.size(), duration_ms, sample_rate);
        fprintf(stdout, "       PCM stats: min=%d max=%d avg_abs=%.1f nonzero=%d/%zu (%.1f%%)\n",
                min_val, max_val, avg_abs, nonzero, pcm.size(),
                (float)nonzero * 100.0f / pcm.size());

        // ── Step 3: 播放合成的语音 ──
        fprintf(stdout, "\n--- Step 3: AudioPlayer playback of TTS result ---\n");

        // 每次测试重新创建 AudioPlayer 实例（避免前一个播放器的 stop/dispose 状态干扰）
        AudioPlayer player;
        if (!player.init(sample_rate)) {
            fprintf(stderr, "[FAIL] AudioPlayer init failed!\n");
            continue;
        }

        fprintf(stdout, "[TEST] Playing synthesized speech...\n");
        player.play(pcm);
        player.wait_for_finish();
        fprintf(stdout, "[PASS] Playback finished\n\n");
    }

    // ── Step 4: 流式合成测试 ──
    // 验证 streaming API：通过回调逐 chunk 接收 PCM 数据
    fprintf(stdout, "\n--- Step 4: Streaming TTS repeat test ---\n");
    const char* stream_texts[] = {
        "好的，我已经准备好了。",
        "第二次流式合成测试。"
    };
    for (int i = 0; i < 2; ++i) {
        size_t total_samples = 0;
        int chunks = 0;
        // 流式合成：每生成一段 PCM 数据就触发回调
        bool ok = tts.synthesize_stream(stream_texts[i], "中文女",
            [&](const std::vector<int16_t>& pcm) {
                total_samples += pcm.size();
                chunks++;
                return true;  // 返回 true 继续接收后续 chunk
            });
        if (!ok || total_samples == 0) {
            fprintf(stderr, "[FAIL] Stream synthesis failed for: %s\n", stream_texts[i]);
            return 1;
        }
        fprintf(stdout, "[PASS] Stream %d: %d chunks, %zu samples\n",
                i + 1, chunks, total_samples);
    }

    fprintf(stdout, "========================================\n");
    fprintf(stdout, "  TTS Test Complete!\n");
    fprintf(stdout, "========================================\n");

    // 使用 _exit() 直接退出进程，跳过 atexit 处理程序和 C++ 析构函数
    // 这是嵌入式 Python 的安全退出方式（避免 Py_Finalize 崩溃）
    fflush(nullptr);
    _exit(0);
}

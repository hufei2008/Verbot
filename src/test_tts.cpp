#include "tts_engine.h"
#include "audio_player.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

// ============================================================
// TTS 独立测试
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
    int sample_rate = 24000;
    if (const char* env_sample_rate = std::getenv("TTS_SAMPLE_RATE")) {
        int configured_sample_rate = std::atoi(env_sample_rate);
        if (configured_sample_rate > 0) {
            sample_rate = configured_sample_rate;
        }
    }

    // ── Step 1: 测试 AudioPlayer 独立播放（播放一个纯音） ──
    fprintf(stdout, "\n--- Step 0: AudioPlayer playback test ---\n");
    {
        AudioPlayer player;
        if (!player.init(sample_rate)) {
            fprintf(stderr, "[FAIL] AudioPlayer init failed!\n");
            return 1;
        }
        fprintf(stdout, "[PASS] AudioPlayer initialized\n");

        // 生成一个 440Hz 正弦波，持续 2 秒，音量较大
        const float duration_sec = 2.0f;
        const float freq_hz = 440.0f;  // A4 note
        const int n_samples = static_cast<int>(sample_rate * duration_sec);
        std::vector<int16_t> tone(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            float t = (float)i / sample_rate;
            float sample = sinf(2.0f * 3.14159f * freq_hz * t);
            // 放大音量（int16 max = 32767）
            tone[i] = static_cast<int16_t>(sample * 30000.0f);
        }

        fprintf(stdout, "[TEST] Playing 440Hz sine tone for %.1f seconds...\n", duration_sec);
        player.play(tone);
        player.wait_for_finish();
        fprintf(stdout, "[PASS] Sine tone playback finished\n");
    }

    // ── Step 1: 初始化 TTS 引擎 ──
    fprintf(stdout, "\n--- Step 1: TtsEngine init ---\n");
    TtsEngine tts;
    if (!tts.init(model_dir, python_home, bridge_dir)) {
        fprintf(stderr, "[FAIL] TtsEngine init failed!\n");
        return 1;
    }
    fprintf(stdout, "[PASS] TtsEngine initialized\n");

    // ── Step 2: 合成测试 ──
    fprintf(stdout, "\n--- Step 2: TTS synthesis test ---\n");
    const char* test_texts[] = {
        "你好，我是你的语音助手。",
        "今天天气真好。",
        "测试一下语音合成效果。",
    };
    const int num_tests = sizeof(test_texts) / sizeof(test_texts[0]);

    for (int i = 0; i < num_tests; ++i) {
        std::string text = test_texts[i];
        std::vector<int16_t> pcm;

        fprintf(stdout, "[TEST %d] Synthesizing: \"%s\"\n", i+1, text.c_str());

        bool ok = tts.synthesize_sync(text, pcm, "中文女");
        if (!ok) {
            fprintf(stderr, "[FAIL] Synthesis failed for: %s\n", text.c_str());
            continue;
        }

        if (pcm.empty()) {
            fprintf(stderr, "[FAIL] Synthesis returned empty PCM!\n");
            continue;
        }

        // 打印 PCM 统计数据
        int16_t min_val = 32767, max_val = -32768;
        double sum = 0;
        int nonzero = 0;
        for (size_t s = 0; s < pcm.size(); ++s) {
            if (pcm[s] < min_val) min_val = pcm[s];
            if (pcm[s] > max_val) max_val = pcm[s];
            sum += std::abs(pcm[s]);
            if (pcm[s] != 0) nonzero++;
        }
        double avg_abs = sum / pcm.size();

        float duration_ms = (float)pcm.size() * 1000.0f / (float)sample_rate;
        fprintf(stdout, "[PASS] Synthesized %zu samples (%.0f ms, %dHz)\n",
                pcm.size(), duration_ms, sample_rate);
        fprintf(stdout, "       PCM stats: min=%d max=%d avg_abs=%.1f nonzero=%d/%zu (%.1f%%)\n",
                min_val, max_val, avg_abs, nonzero, pcm.size(),
                (float)nonzero * 100.0f / pcm.size());

        // ── Step 3: 播放测试 ──
        fprintf(stdout, "\n--- Step 3: AudioPlayer playback of TTS result ---\n");

        // 每次测试重新创建 AudioPlayer（避免前面播放器已 stop/dispose 问题）
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

    fprintf(stdout, "========================================\n");
    fprintf(stdout, "  TTS Test Complete!\n");
    fprintf(stdout, "========================================\n");

    return 0;
}

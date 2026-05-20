#include "whisper.h"
#include "common-whisper.h"

#include <iostream>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <cmath>
#include <chrono>

// ANSI colors
#define COLOR_CYAN    "\033[36m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_RED     "\033[31m"
#define COLOR_RESET   "\033[0m"

static void cb_log_disable(enum ggml_log_level, const char*, void*) {}

// ★ 改进：高通滤波器，去除 80Hz 以下低频噪声
static void highpass_filter(std::vector<float>& data, float cutoff, int sampleRate) {
    const float dt = 1.0f / sampleRate;
    const float rc = 1.0f / (2.0f * M_PI * cutoff);
    const float alpha = rc / (rc + dt);

    if (data.empty()) return;

    float y_prev = data[0];
    for (size_t i = 1; i < data.size(); ++i) {
        float y = alpha * (y_prev + data[i] - data[i-1]);
        data[i] = y;
        y_prev = y;
    }
}

int main(int argc, char ** argv) {
    whisper_log_set(cb_log_disable, nullptr);

    std::string modelPath  = "models/ggml-base.bin";
    std::string audioPath  = "test_speech.wav";

    if (argc > 1) modelPath = argv[1];
    if (argc > 2) audioPath = argv[2];

    // 1. 加载模型
    std::cout << COLOR_CYAN "Loading Whisper model: " << modelPath << " ..." COLOR_RESET << std::endl;

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;  // ★ 启用 GPU 加速

    struct whisper_context * ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to load Whisper model!" << std::endl;
        return 1;
    }
    std::cout << COLOR_GREEN "Model loaded successfully." COLOR_RESET << std::endl;

    // 2. 读取音频文件
    std::cout << COLOR_CYAN "Reading audio: " << audioPath << " ..." COLOR_RESET << std::endl;

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;

    if (!read_audio_data(audioPath, pcmf32, pcmf32s, false)) {
        std::cerr << "Failed to read audio file!" << std::endl;
        whisper_free(ctx);
        return 1;
    }

    double durSec = pcmf32.size() / 16000.0;
    printf("  Audio duration: %.2fs, samples: %zu\n", durSec, pcmf32.size());

    // ★ 改进：对音频进行高通滤波预处理，去除低频噪声
    highpass_filter(pcmf32, 80.0f, 16000);

    // 3. 转写
    std::cout << COLOR_YELLOW "\n--- Transcribing with Whisper (language=zh) ---" COLOR_RESET << std::endl;

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    wparams.print_progress   = true;
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = true;
    wparams.language         = "zh";
    wparams.n_threads        = std::min(4, (int)std::thread::hardware_concurrency());
    wparams.no_context       = false;
    wparams.suppress_blank   = true;
    wparams.suppress_nst     = true;
    wparams.single_segment   = false;
    wparams.temperature      = 0.0f;
    wparams.temperature_inc  = 0.4f;          // ★ 更快回退，覆盖更广
    wparams.entropy_thold    = 1.2f;           // ★ 降低，更严格抑制重复
    wparams.logprob_thold    = -1.0f;          // 对数概率阈值
    wparams.no_speech_thold  = 0.5f;           // ★ 降低，更敏感捕捉语音
    wparams.max_len          = 40;             // ★ 提高，允许更长的句子

    // ★ 改进：Beam search 从 5-beam 提升到 7-beam
    wparams.beam_search.beam_size = 7;

    // ★ 改进：更丰富的中文初始提示
    const char * chinese_prompt_v2 =
        "你好世界欢迎使用语音识别系统"
        "一二三四五六七八九十百千万亿"
        "今天天气很好请问有什么可以帮助你的"
        "打开关闭保存删除复制粘贴撤回发送"
        "他说她说他们说我们在你们他们在"
        "能不能要不要会不会可以不可以";

    wparams.initial_prompt  = chinese_prompt_v2;
    wparams.carry_initial_prompt = false;

    printf("\n");

    if (whisper_full(ctx, wparams, pcmf32.data(), (int)pcmf32.size()) == 0) {
        const int nSeg = whisper_full_n_segments(ctx);
        printf("\n%s=== Transcription Result (%d segments) ===%s\n", COLOR_GREEN, nSeg, COLOR_RESET);

        for (int i = 0; i < nSeg; ++i) {
            const char* text = whisper_full_get_segment_text(ctx, i);
            const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
            const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

            if (text && text[0] != '\0') {
                printf("  [%s --> %s]  %s\n",
                       to_timestamp(t0, false).c_str(),
                       to_timestamp(t1, false).c_str(),
                       text);
            }
        }

        printf("%s==============================%s\n", COLOR_GREEN, COLOR_RESET);
    } else {
        printf("  %s[transcription failed]%s\n", COLOR_RED, COLOR_RESET);
    }

    // 4. 清理
    whisper_free(ctx);
    return 0;
}
// ============================================================
// test_whisper — Whisper 语音识别独立测试程序
//
// 用法: ./test_whisper [模型路径] [音频文件]
// 默认: models/ggml-base.bin 和 test_speech.wav
//
// 功能：
//   1. 加载 Whisper 模型（支持 GPU 加速）
//   2. 读取 WAV 音频文件
//   3. 高通滤波预处理（去除 80Hz 以下低频噪声）
//   4. Beam Search 转写（中文优化参数）
//   5. 打印带时间戳的识别结果
// ============================================================

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

// ANSI 终端颜色代码（用于美化输出）
#define COLOR_CYAN    "\033[36m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_RED     "\033[31m"
#define COLOR_RESET   "\033[0m"

// 禁用 ggml 日志输出（避免干扰测试结果）
static void cb_log_disable(enum ggml_log_level, const char*, void*) {}

// 一阶高通 IIR 滤波器（一阶无限脉冲响应滤波器）
// 作用：去除 cutoff 频率以下的低频噪声，提升语音清晰度
// 公式：y[n] = α * (y[n-1] + x[n] - x[n-1])，其中 α = RC/(RC + dt)
// @param data       输入的音频样本（float，16kHz 归一化）
// @param cutoff     截止频率（Hz），默认 80Hz
// @param sampleRate 采样率（Hz），默认 16000
static void highpass_filter(std::vector<float>& data, float cutoff, int sampleRate) {
    // 计算滤波器系数
    const float dt = 1.0f / sampleRate;           // 采样间隔
    const float rc = 1.0f / (2.0f * M_PI * cutoff); // RC 时间常数
    const float alpha = rc / (rc + dt);            // 平滑系数

    if (data.empty()) return;

    float y_prev = data[0];  // 初始输出值
    for (size_t i = 1; i < data.size(); ++i) {
        // 一阶差分方程
        float y = alpha * (y_prev + data[i] - data[i-1]);
        data[i] = y;
        y_prev = y;
    }
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char ** argv) {
    // 禁用 ggml 日志，避免干扰测试输出
    whisper_log_set(cb_log_disable, nullptr);

    // 默认模型和音频路径
    std::string modelPath  = "models/ggml-base.bin";
    std::string audioPath  = "test_speech.wav";

    // 支持命令行参数覆盖默认路径
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) audioPath = argv[2];

    // ── 步骤 1：加载 Whisper 模型 ──
    std::cout << COLOR_CYAN "Loading Whisper model: " << modelPath << " ..." COLOR_RESET << std::endl;

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;  // 启用 GPU 加速（macOS Metal / CUDA）

    struct whisper_context * ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to load Whisper model!" << std::endl;
        return 1;
    }
    std::cout << COLOR_GREEN "Model loaded successfully." COLOR_RESET << std::endl;

    // ── 步骤 2：读取音频文件 ──
    std::cout << COLOR_CYAN "Reading audio: " << audioPath << " ..." COLOR_RESET << std::endl;

    std::vector<float> pcmf32;                // 单声道 16kHz float PCM
    std::vector<std::vector<float>> pcmf32s;  // 多声道（未使用，保持兼容）

    if (!read_audio_data(audioPath, pcmf32, pcmf32s, false)) {
        std::cerr << "Failed to read audio file!" << std::endl;
        whisper_free(ctx);
        return 1;
    }

    double durSec = pcmf32.size() / 16000.0;
    printf("  Audio duration: %.2fs, samples: %zu\n", durSec, pcmf32.size());

    // 音频预处理：高通滤波去除 80Hz 以下低频噪声
    highpass_filter(pcmf32, 80.0f, 16000);

    // ── 步骤 3：配置转写参数并执行识别 ──
    std::cout << COLOR_YELLOW "\n--- Transcribing with Whisper (language=zh) ---" COLOR_RESET << std::endl;

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    // 输出控制
    wparams.print_progress   = true;   // 打印进度信息
    wparams.print_special    = false;  // 不打印特殊 token
    wparams.print_realtime   = false;  // 不打印实时输出
    wparams.print_timestamps = true;   // 打印时间戳
    // 语言和线程
    wparams.language         = "zh";   // 中文识别
    wparams.n_threads        = std::min(4, (int)std::thread::hardware_concurrency());
    // 上下文和抑制
    wparams.no_context       = true;              // 每段语音独立识别，避免上一轮文本污染短指令
    wparams.suppress_blank   = true;              // 抑制空白输出
    wparams.suppress_nst     = true;              // 抑制非语音 token
    wparams.single_segment   = true;              // 测试单条指令时减少拆段和补字
    // 温度和采样
    wparams.temperature      = 0.0f;              // 贪婪解码（确定性输出）
    wparams.temperature_inc  = 0.4f;              // 温度递增步长（更快回退）
    wparams.entropy_thold    = 1.2f;              // 熵阈值（严格抑制重复）
    wparams.logprob_thold    = -1.0f;             // 对数概率阈值
    wparams.no_speech_thold  = 0.5f;              // 无语音检测阈值（低=更灵敏）
    wparams.max_len          = 40;                // 最大段长度

    // Beam Search 参数：7-beam 提供更好的中文识别质量
    wparams.beam_search.beam_size = 7;

    // 中文初始提示文本（提高低资源场景下的识别准确性）
    const char * chinese_prompt_v2 =
        "语音助手常用指令："
        "播放周杰伦的歌，播放周杰伦歌曲，播放稻香，播放晴天，播放网易云音乐。"
        "暂停音乐，继续播放，下一首，上一首，打开网易云音乐。"
        "北京天气怎么样，上海天气怎么样，打开计算器，现在几点了。"
        "你好，请问有什么可以帮助你的。";

    wparams.initial_prompt  = chinese_prompt_v2;
    wparams.carry_initial_prompt = false;  // 不将初始提示带入后续段

    printf("\n");

    // 执行转写
    if (whisper_full(ctx, wparams, pcmf32.data(), (int)pcmf32.size()) == 0) {
        const int nSeg = whisper_full_n_segments(ctx);
        printf("\n%s=== Transcription Result (%d segments) ===%s\n", COLOR_GREEN, nSeg, COLOR_RESET);

        // 遍历每个识别段，打印文本和时间戳
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

    // ── 步骤 4：释放资源 ──
    whisper_free(ctx);
    return 0;
}

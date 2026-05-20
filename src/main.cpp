#include "audio_recorder.h"

#include "whisper.h"

#include <iostream>
#include <cstdio>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>
#include <atomic>
#include <algorithm>
#include <ctime>
#include <cmath>

// ANSI colors
#define COLOR_CYAN    "\033[36m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_RED     "\033[31m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_RESET   "\033[0m"
#define CLEAR_SCREEN  "\033[2J\033[H"

constexpr int SAMPLES_PER_VAD_FRAME = 512;   // 32ms @16kHz
constexpr int SAMPLE_RATE = 16000;

// 线程安全的环形缓冲
class RingBuffer {
public:
    void push(const float* data, size_t n) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffer.insert(m_buffer.end(), data, data + n);
    }

    size_t consume(std::vector<float>& out, size_t n) {
        std::lock_guard<std::mutex> lock(m_mutex);
        n = std::min(n, m_buffer.size());
        if (n == 0) return 0;
        out.assign(m_buffer.begin(), m_buffer.begin() + n);
        m_buffer.erase(m_buffer.begin(), m_buffer.begin() + n);
        return n;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_buffer.size();
    }

private:
    std::vector<float> m_buffer;
    std::mutex m_mutex;
};

// 抑制 whisper/VAD 内部 debug 日志
static void cb_log_disable(enum ggml_log_level, const char*, void*) {}

// 获取当前时间字符串
static std::string current_time_str() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

// 计算 RMS 能量
static float compute_rms(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum / n);
}

// 计算峰值幅度
static float compute_peak(const float* data, size_t n) {
    float peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float absv = std::fabs(data[i]);
        if (absv > peak) peak = absv;
    }
    return peak;
}

// ★ 改进：简单高通 FIR 滤波器，滤除 80Hz 以下低频噪声（空调/风扇/风声）
// 显著提升 VAD 准确率和 Whisper 识别率
static void highpass_filter(std::vector<float>& data, float cutoff, int sampleRate) {
    // 一阶 IIR 高通：y[n] = 0.5 * (x[n] - x[n-1]) + 0.5 * y[n-1] * alpha
    // 但更简单有效：使用 RC 高通滤波器
    // alpha = RC / (RC + dt)，其中 RC = 1/(2*pi*cutoff)
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
    // 抑制 whisper VAD 内部 debug 日志
    whisper_log_set(cb_log_disable, nullptr);

    std::string modelPath    = "models/ggml-base.bin";
    std::string vadModelPath = "models/ggml-silero-v6.2.0.bin";

    if (argc > 1) modelPath    = argv[1];
    if (argc > 2) vadModelPath = argv[2];

    // ============================================================
    // 1. 初始化 Whisper
    // ============================================================
    printf("%s[%s] Loading Whisper model: %s ...%s\n",
           COLOR_CYAN, current_time_str().c_str(), modelPath.c_str(), COLOR_RESET);

    // 启用 GPU（Metal）加速，Apple Silicon 上约 3-5x 加速
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    // Metal 设备数量（Apple Silicon 统一内存）
    cparams.gpu_device = 0;

    struct whisper_context * ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to load Whisper model!" << std::endl;
        return 1;
    }
    printf("%s[%s] Whisper model loaded.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    // ============================================================
    // 2. 初始化 VAD
    // ============================================================
    printf("%s[%s] Loading VAD model: %s ...%s\n",
           COLOR_CYAN, current_time_str().c_str(), vadModelPath.c_str(), COLOR_RESET);

    struct whisper_vad_context_params vparams = whisper_vad_default_context_params();
    vparams.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    vparams.use_gpu   = false;

    struct whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(
            vadModelPath.c_str(), vparams);
    if (!vctx) {
        std::cerr << "Failed to load VAD model!" << std::endl;
        whisper_free(ctx);
        return 1;
    }
    printf("%s[%s] VAD model loaded.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    // VAD 参数
    struct whisper_vad_params vad_params = whisper_vad_default_params();
    vad_params.threshold               = 0.5f;
    vad_params.min_speech_duration_ms  = 200;
    vad_params.min_silence_duration_ms = 300;   // ⬇ 从 600ms 降到 300ms，减少结束等待
    vad_params.speech_pad_ms           = 200;

    printf("%s[%s] VAD params: thr=%.1f, min_speech=%dms, min_silence=%dms, pad=%dms%s\n",
           COLOR_CYAN, current_time_str().c_str(),
           vad_params.threshold,
           vad_params.min_speech_duration_ms,
           vad_params.min_silence_duration_ms,
           vad_params.speech_pad_ms,
           COLOR_RESET);

    // ============================================================
    // 3. 启动录音（流式模式）
    // ============================================================
    RingBuffer ringBuffer;
    std::atomic<bool> running{true};

    AudioRecorder recorder;
    recorder.setDataCallback([&](const float* data, size_t n) {
        ringBuffer.push(data, n);
    });

    if (!recorder.start(true)) {
        std::cerr << "Failed to start recording!" << std::endl;
        whisper_vad_free(vctx);
        whisper_free(ctx);
        return 1;
    }

    // ============================================================
    // 程序启动完毕
    // ============================================================
    printf(CLEAR_SCREEN);
    printf("%s╔════════════════════════════════════════════════════╗%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("%s║       🎙  Real-time ASR (Whisper + VAD)          ║%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("%s║       Press Ctrl+C to exit                       ║%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("%s╚════════════════════════════════════════════════════╝%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("\n");
    printf("%s[%s] 🔴 Recording started (16kHz, 32-bit float)%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);
    printf("%s──────────────────────────────────────────────────%s\n", COLOR_CYAN, COLOR_RESET);

    // ============================================================
    // 4. VAD + ASR 主循环 - 增强版状态机
    // ============================================================
    //
    // 改进要点：
    //   1) 语音开始前预留 200ms 静音作为上下文（pad）
    //   2) 语音结束后追加 200ms 静音（pad）
    //   3) 说话中判断条件放宽：VAD 或 能量，任一触发即认为有语音
    //   4) 去除 no_context，让 Whisper 利用上下文提升准确率
    //   5) 动态阈值自适应
    //
    // ============================================================

    enum State {
        STATE_IDLE,
        STATE_SPEECHING,    // 正在说话
    };

    State state = STATE_IDLE;

    std::vector<float> speechBuffer;
    int speechCount = 0;
    int vadStartCount = 0;

    // 语音段累计帧数
    int speechFrames = 0;

    // 连续静音帧计数器（用于检测语音结束）
    const int SILENCE_FRAMES_THRESHOLD = 8;   // ⬇ 8帧 * 32ms = 256ms 静音判定结束（从640ms降低）
    int silenceFrames = 0;

    // 最小语音帧数（避免过短触发）
    const int MIN_SPEECH_FRAMES = 6;   // 6帧 * 32ms = 192ms，短词也能识别

    // 最大语音帧数（强制转写）
    const int MAX_SPEECH_FRAMES = 30 * 1000 / 32;  // ~30s

    // RMS 阈值 - 降低到 0.01，适应麦克风增益较小的场景
    const float SPEECH_RMS_THRESHOLD = 0.01f;

    // 底噪自适应
    float noiseFloor = 0.01f;

    // 用于语音段 padding 的环形缓冲（保留最近 500ms 音频）
    const int PAD_FRAMES = 8;  // 8帧 ≈ 256ms 的 padding
    std::vector<float> padBuffer;

    // 先向 VAD 输入一些静音帧做校准
    printf("%s[%s] Calibrating VAD (300ms of silence)...%s\n",
           COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);

    // 给 VAD 喂静音帧做校准
    std::vector<float> silenceInput(SAMPLES_PER_VAD_FRAME, 0.0f);
    for (int i = 0; i < 5; ++i) {
        whisper_vad_detect_speech_no_reset(vctx, silenceInput.data(), SAMPLES_PER_VAD_FRAME);
    }
    whisper_vad_reset_state(vctx);

    printf("%s[%s] VAD calibrated.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    while (running) {
        // 从环形缓冲取数据
        std::vector<float> chunk;
        size_t got = ringBuffer.consume(chunk, SAMPLES_PER_VAD_FRAME);

        if (got == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        // 计算 RMS 能量和峰值
        float rms = compute_rms(chunk.data(), (int)chunk.size());
        float peak = compute_peak(chunk.data(), (int)chunk.size());

        // 更新底噪估计（只在空闲时更新更准确）
        if (state == STATE_IDLE && rms < SPEECH_RMS_THRESHOLD * 3.0f) {
            noiseFloor = 0.995f * noiseFloor + 0.005f * rms;
        }

        // 运行 VAD
        bool vadSpeech = whisper_vad_detect_speech_no_reset(vctx,
                chunk.data(), (int)chunk.size());

        // ---- ★ 关键改进：综合决策逻辑 ★ ----
        // 使用能量检测（但用自适应阈值 = max(noiseFloor*2, SPEECH_RMS_THRESHOLD)）
        float adaptiveThreshold = std::max(noiseFloor * 3.0f, SPEECH_RMS_THRESHOLD);
        bool energyHasSpeech = (rms >= adaptiveThreshold);

        // IDLE 状态：需要 VAD 和能量同时触发（严格，防止误触发）
        // SPEECHING 状态：同样用 AND - 必须 VAD 和能量都认为是语音才保持
        // 修复前用了 OR，导致任一检测不断重置 silenceFrames，永远结束不了
        bool isSpeech;
        if (state == STATE_IDLE) {
            // ★ 严格模式：VAD + 能量双确认才判定为语音开始
            isSpeech = vadSpeech && energyHasSpeech;
        } else {
            // ★ 修复：改为 AND 判断
            //   isSpeech=true:  VAD和能量都认为有语音 → 继续说话，重置静音计数
            //   isSpeech=false: 只要有一方认为无语音 → 不重置静音计数
            //   这样即使偶发噪声/误判也不会让 silenceFrames 永远归零
            isSpeech = vadSpeech && energyHasSpeech;
        }

        // ★ 调试日志：打印每帧的 VAD/能量/判定结果
        static int logCounter = 0;
        if (++logCounter % 30 == 0 || state == STATE_SPEECHING) {
            printf("%s[DEBUG] frame: vad=%d rms=%.4f peak=%.4f thr=%.4f noise=%.4f energy=%d isSpeech=%d state=%s silence=%d%s\n",
                   COLOR_CYAN,
                   (int)vadSpeech, rms, peak, adaptiveThreshold, noiseFloor,
                   (int)energyHasSpeech, (int)isSpeech,
                   (state == STATE_IDLE) ? "IDLE" : "SPEECH",
                   silenceFrames,
                   COLOR_RESET);
        }

        // ---- 更新 padding 环形缓冲 ----
        padBuffer.insert(padBuffer.end(), chunk.begin(), chunk.end());
        if (padBuffer.size() > (size_t)PAD_FRAMES * SAMPLES_PER_VAD_FRAME) {
            padBuffer.erase(padBuffer.begin(), padBuffer.begin() + SAMPLES_PER_VAD_FRAME);
        }

        // 状态机
        switch (state) {
            case STATE_IDLE: {
                if (isSpeech) {
                    // 检测到语音，开始累积
                    state = STATE_SPEECHING;
                    speechFrames = 0;
                    silenceFrames = 0;
                    vadStartCount++;

                    // ★ 改进：把 padding 中的最近音频作为语音前导上下文加入
                    speechBuffer.clear();
                    speechBuffer.insert(speechBuffer.end(), padBuffer.begin(), padBuffer.end());
                    speechFrames += (int)(padBuffer.size() / SAMPLES_PER_VAD_FRAME);

                    // 当前帧也要入 buffer
                    speechBuffer.insert(speechBuffer.end(), chunk.begin(), chunk.end());
                    speechFrames++;

                    printf("\n%s▶ [%s] Voice activity detected (rms=%.4f, peak=%.4f, noise=%.4f)%s\n",
                           COLOR_YELLOW, current_time_str().c_str(),
                           rms, peak, noiseFloor, COLOR_RESET);
                }
                break;
            }

            case STATE_SPEECHING: {
                // 累积当前帧
                speechBuffer.insert(speechBuffer.end(), chunk.begin(), chunk.end());
                speechFrames++;

                if (isSpeech) {
                    // 还在说话，重置静音计数器
                    silenceFrames = 0;
                } else {
                    // VAD 和能量都不够了，开始计数静音帧
                    silenceFrames++;
                }

                // 检查条件
                if (silenceFrames >= SILENCE_FRAMES_THRESHOLD) {
                    // 连续静音达到阈值 -> 语音结束
                    if (speechFrames >= MIN_SPEECH_FRAMES) {
                        // ★ 改进：语音结束后追加 padding 帧作为上下文
                        speechBuffer.insert(speechBuffer.end(),
                            padBuffer.begin(), padBuffer.end());

                        goto do_transcribe;
                    } else {
                        // 太短了，丢弃
                        float durMs = speechFrames * 32.0f;
                        state = STATE_IDLE;
                        speechBuffer.clear();
                        speechFrames = 0;
                        silenceFrames = 0;
                        printf("%s↩ [%s] Too short (%.0fms), discarding%s\n",
                               COLOR_YELLOW, current_time_str().c_str(),
                               durMs, COLOR_RESET);
                    }
                } else if (speechFrames >= MAX_SPEECH_FRAMES) {
                    // 语音太长，强制转写
                    printf("%s⏰ [%s] Speech too long (%.0fs), force-transcribing...%s\n",
                           COLOR_YELLOW, current_time_str().c_str(),
                           speechFrames * 32.0f / 1000.0f, COLOR_RESET);
                    whisper_vad_reset_state(vctx);
                    goto do_transcribe;
                }

                // 正常流程：继续累积
                break;
            }
        }

        // 跳过转写
        continue;

    do_transcribe:
        {
            // ★ 改进：对语音段应用高通滤波，去除低频噪声，提升 Whisper 识别率
            highpass_filter(speechBuffer, 80.0f, SAMPLE_RATE);

            double durSec = speechBuffer.size() / (double)SAMPLE_RATE;
            printf("%s📏 [%s] Speech segment (%.1fs, %zu samples)%s\n",
                   COLOR_CYAN, current_time_str().c_str(), durSec,
                   speechBuffer.size(), COLOR_RESET);
            printf("%s⏳ [%s] Transcribing #%d...%s\n",
                   COLOR_YELLOW, current_time_str().c_str(), ++speechCount, COLOR_RESET);

            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
            wparams.print_progress   = false;
            wparams.print_special    = false;
            wparams.print_realtime   = false;
            wparams.print_timestamps = false;
            wparams.language         = "zh";
            wparams.n_threads        = std::min(4, (int)std::thread::hardware_concurrency());

    // ---------- 关键改进（v2）：提升识别准确率 ----------
    wparams.no_context       = false;        // 利用上下文提升准确率
    wparams.single_segment   = false;        // 允许 Whisper 内部自动分段和对齐

    // suppress_blank: 抑制空输出
    wparams.suppress_blank   = true;
    // suppress_nst: 抑制非语音 token（如背景噪声标记）
    wparams.suppress_nst     = true;

    // ★ 改进：多级温度回退策略 + 更细粒度的控制
    // 依次尝试 0.0(贪婪) → 0.2 → 0.4 → 0.6 → 0.8，用最佳结果
    wparams.temperature      = 0.0f;
    wparams.temperature_inc  = 0.4f;         // 更快回退到更高温度，覆盖更广
    wparams.entropy_thold    = 1.2f;         // ⬇ 从1.5降到1.2，更严格抑制重复文本
    wparams.logprob_thold    = -1.0f;        // 对数概率阈值
    wparams.no_speech_thold  = 0.5f;         // ⬇ 从0.6降到0.5，更敏感捕捉语音
    wparams.max_len          = 40;           // ⬆ 从30提高到40，允许更长的句子

    // ★ 改进：Beam search 从 5-beam 提升到 7-beam
    // 更大的搜索空间显著提升准确率（尤其对长句和同音字），
    // Apple Silicon GPU 加速下速度影响很小
    wparams.beam_search.beam_size = 7;

    // ★ 改进：更丰富的中文初始提示，帮助模型激活更广泛的中文词汇
    // 覆盖日常对话、数字、指令等常见场景
    const char * chinese_prompt_v2 =
        "你好世界欢迎使用语音识别系统"
        "一二三四五六七八九十百千万亿"
        "今天天气很好请问有什么可以帮助你的"
        "打开关闭保存删除复制粘贴撤回发送"
        "他说她说他们说我们在你们他们在"
        "能不能要不要会不会可以不可以";

    wparams.initial_prompt        = chinese_prompt_v2;
    wparams.carry_initial_prompt  = false;

    // ★ 改进：音频上下文自动，禁用 token-level 调试
    wparams.audio_ctx            = 0;         // 自动
    wparams.tdrz_enable          = false;     // 禁用 token-level 调试

            auto t0 = std::chrono::high_resolution_clock::now();

            int ret = whisper_full(ctx, wparams,
                    speechBuffer.data(), (int)speechBuffer.size());

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (ret == 0) {
                const int nSeg = whisper_full_n_segments(ctx);
                printf("%s✅ [%s] Transcription done in %.0fms (%d segments)%s\n",
                       COLOR_GREEN, current_time_str().c_str(),
                       elapsedMs, nSeg, COLOR_RESET);

                printf("%s  ┌─────────────────────────────────────────────┐%s\n",
                       COLOR_BOLD COLOR_GREEN, COLOR_RESET);
                printf("%s  │ 📝 RESULT: ", COLOR_BOLD COLOR_GREEN);
                for (int i = 0; i < nSeg; ++i) {
                    const char* text = whisper_full_get_segment_text(ctx, i);
                    if (text && text[0] != '\0') {
                        printf("%s", text);
                    }
                }
                printf("%s\n", COLOR_RESET);
                printf("%s  └─────────────────────────────────────────────┘%s\n",
                       COLOR_BOLD COLOR_GREEN, COLOR_RESET);
                printf("\n");
            } else {
                printf("%s❌ [%s] Transcription failed (ret=%d)%s\n",
                       COLOR_RED, current_time_str().c_str(), ret, COLOR_RESET);
            }

            // 重置 VAD 状态和所有变量
            whisper_vad_reset_state(vctx);
            speechBuffer.clear();
            speechFrames = 0;
            silenceFrames = 0;
            state = STATE_IDLE;

            printf("%s──────────────────────────────────────────────────%s\n",
                   COLOR_CYAN, COLOR_RESET);
        }
        // end do_transcribe
    }

    // ============================================================
    // 5. 清理
    // ============================================================
    printf("\n%s[%s] Shutting down...%s\n", COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);
    printf("%s[%s] Total detections: %d, transcriptions: %d%s\n",
           COLOR_CYAN, current_time_str().c_str(), vadStartCount, speechCount, COLOR_RESET);
    recorder.stop();
    whisper_vad_free(vctx);
    whisper_free(ctx);

    return 0;
}
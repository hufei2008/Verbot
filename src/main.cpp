#include "audio_recorder.h"
#include "semantic_engine.h"

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
#include <cstdlib>
#include <future>
#include <csignal>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>

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
constexpr bool VAD_DEBUG_LOG = false;

static volatile std::sig_atomic_t g_stop_requested = 0;
static volatile std::sig_atomic_t g_graceful_sigint_enabled = 0;

    static void handle_sigint(int) {
        if (!g_graceful_sigint_enabled || g_stop_requested) {
            _exit(130);
        }
        g_stop_requested = 1;
    }

    // 安全退出辅助函数：跳过 Python/MLX 的 atexit/finalizer 链。
    // 嵌入式 Python 加载 MLX、tokenizers 后，正常 return/exit 可能在
    // 进程收尾时触发 C 扩展析构并 crash。这里先 flush 日志，再直接
    // _exit，保留正常退出码。
    static void safe_exit(int code) {
        fflush(nullptr);
        _exit(code);
        __builtin_unreachable();
    }

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

static bool should_skip_low_quality_segment(double durSec, float rms, float peak) {
    if (durSec < 0.45) return true;
    if (durSec < 0.85 && rms < 0.016f && peak < 0.055f) return true;
    return false;
}

// ★ 改进：简单高通 FIR 滤波器，滤除 80Hz 以下低频噪声（空调/风扇/风声）
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

static void execute_action(const Action& action) {
    switch (action.type) {
        case ActionType::OPEN_APP: {
            std::string cmd = "open -a \"" + action.target + "\"";
            printf("%s  ↪ Opening app: %s%s\n",
                   COLOR_YELLOW, cmd.c_str(), COLOR_RESET);
            std::system(cmd.c_str());
            break;
        }
        case ActionType::SEARCH_WEB: {
            std::string url = "https://www.google.com/search?q=" + action.target;
            for (auto& c : url) { if (c == ' ') c = '+'; }
            std::string cmd = "open \"" + url + "\"";
            printf("%s  ↪ Searching: %s%s\n",
                   COLOR_YELLOW, url.c_str(), COLOR_RESET);
            std::system(cmd.c_str());
            break;
        }
        case ActionType::GET_TIME: {
            printf("%s  ↪ Current time: %s%s\n",
                   COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);
            break;
        }
        case ActionType::GET_WEATHER: {
            printf("%s  ↪ Weather API result: %s%s\n",
                   COLOR_YELLOW,
                   (!action.params.empty() ? action.params : action.response_text).c_str(),
                   COLOR_RESET);
            break;
        }
        default: {
            if (action.type != ActionType::NONE) {
                printf("%s  ↪ (No safe handler for %s)%s\n",
                       COLOR_YELLOW, action.action_name.c_str(), COLOR_RESET);
            }
            break;
        }
    }
}

static bool looks_like_asr_prompt_artifact(const std::string& text) {
    if (text.empty()) return true;
    if (text == "," || text == "." || text == "?" || text == "!" ||
        text == "，" || text == "。" || text == "？" || text == "！") return true;
    if (text == "嗯" || text == "嗯嗯" || text == "啊" || text == "哦") return true;
    if (text == "咳" || text == "咳咳" || text == "咳咳咳") return true;
    if (text == "哼" || text == "哼哼") return true;
    if (text.find("打开关闭保存删除复制粘贴撤回发送") != std::string::npos) return true;
    if (text.find("请问有什么可以帮助你的") != std::string::npos &&
        text.find("他说他们说") != std::string::npos) return true;
    if (text.size() > 90 &&
        text.find("今天天气很好") != std::string::npos &&
        text.find("一二三四") != std::string::npos) return true;
    return false;
}

int main(int argc, char ** argv) {
    std::signal(SIGINT, handle_sigint);

    // 抑制 whisper VAD 内部 debug 日志
    whisper_log_set(cb_log_disable, nullptr);
    llama_log_set(cb_log_disable, nullptr);

    std::string whisperModel  = "models/ggml-medium.bin";
    std::string vadModelPath  = "models/ggml-silero-v6.2.0.bin";
    std::string llmModelPath  = "models/gemma4-26b-a4b-it-q4_k_m.gguf";
    std::string textInput;

    bool textMode = false;
    bool argError = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--text") {
            textMode = true;
            if (i + 1 >= argc) {
                argError = true;
                break;
            }
            textInput = argv[++i];
        } else if (arg == "--llm") {
            if (i + 1 >= argc) {
                argError = true;
                break;
            }
            llmModelPath = argv[++i];
        } else if (!textMode) {
            if (i == 1) whisperModel = arg;
            else if (i == 2) vadModelPath = arg;
            else if (i == 3) llmModelPath = arg;
        }
    }

    if (argError || (textMode && textInput.empty())) {
        fprintf(stderr, "Usage: %s --text \"打开计算器\" [--llm path/to/model.gguf]\n", argv[0]);
        return 1;
    }

    // ============================================================
    // 1. 初始化语义引擎（LLM + TTS）
    // ============================================================
    printf("%s[%s] Loading LLM model: %s ...%s\n",
           COLOR_CYAN, current_time_str().c_str(), llmModelPath.c_str(), COLOR_RESET);

    SemanticEngine semanticEngine;
    if (!semanticEngine.init(llmModelPath, 4096, 4, true)) {
        fprintf(stderr, "Failed to initialize semantic engine!\n");
        return 1;
    }
    printf("%s[%s] Semantic engine ready.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    // TTS 状态
    if (semanticEngine.tts_ready()) {
        const char* ttsBackend = std::getenv("TTS_BACKEND");
        printf("%s[%s] TTS ready (%s backend via Python bridge)%s\n",
               COLOR_GREEN, current_time_str().c_str(),
               ttsBackend ? ttsBackend : "macos", COLOR_RESET);
    } else {
        printf("%s[%s] TTS not available. "
               "To enable, set:\n"
               "  export TTS_BACKEND=macos\n"
               "  export MACOS_TTS_VOICE=Tingting\n"
               "  export QWEN_TTS_PYTHON_HOME=/path/to/conda/envs/cosyvoice\n"
               "  export QWEN_TTS_BRIDGE_DIR=/path/to/study2/python%s\n",
               COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);
    }

    if (textMode) {
        std::promise<void> done;
        auto future = done.get_future();
        bool doneSet = false;
        g_graceful_sigint_enabled = 1;

        printf("%s🧠 [%s] Text mode input: \"%s\"%s\n",
               COLOR_MAGENTA, current_time_str().c_str(), textInput.c_str(), COLOR_RESET);

        semanticEngine.process_asr_result(textInput, [&](const Action& action) {
            if (!action.response_text.empty()) {
                printf("\n%s🤖 %s%s\n",
                       COLOR_BOLD COLOR_CYAN, action.response_text.c_str(), COLOR_RESET);
                // TTS 由 SemanticEngine 内部自动处理
                if (!doneSet) {
                    doneSet = true;
                    done.set_value();
                }
            }

            if (action.type != ActionType::NONE) {
                printf("%s⚡ [Action] %s: %s%s\n",
                       COLOR_BOLD COLOR_GREEN,
                       action.action_name.c_str(),
                       action.target.c_str(),
                       COLOR_RESET);
                execute_action(action);
            }
        });

        while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
            if (g_stop_requested) {
                fprintf(stderr, "\nInterrupted.\n");
                safe_exit(130);
            }
        }

        if (g_stop_requested) {
            fprintf(stderr, "\nInterrupted.\n");
            safe_exit(130);
        }

        // 等待 TTS 播放完成
        printf("%s[%s] Waiting for TTS playback to finish...%s\n",
               COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);
        while (!g_stop_requested && !semanticEngine.wait_for_tts_for(100)) {}
        if (g_stop_requested) {
            fprintf(stderr, "\nInterrupted.\n");
            safe_exit(130);
        }

        safe_exit(0);
    }

    // ============================================================
    // 2. 初始化 Whisper
    // ============================================================
    printf("%s[%s] Loading Whisper model: %s ...%s\n",
           COLOR_CYAN, current_time_str().c_str(), whisperModel.c_str(), COLOR_RESET);

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    cparams.gpu_device = 0;

    struct whisper_context * ctx = whisper_init_from_file_with_params(whisperModel.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to load Whisper model!" << std::endl;
        safe_exit(1);
    }
    printf("%s[%s] Whisper model loaded.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    // ============================================================
    // 3. 初始化 VAD
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
        safe_exit(1);
    }
    printf("%s[%s] VAD model loaded.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    // VAD 参数
    struct whisper_vad_params vad_params = whisper_vad_default_params();
    vad_params.threshold               = 0.7f;   // 提高概率阈值，减少噪音误触发
    vad_params.min_speech_duration_ms  = 300;    // 增加最小语音持续时长
    vad_params.min_silence_duration_ms = 400;    // 增加沉默判定时长
    vad_params.speech_pad_ms           = 200;

    printf("%s[%s] VAD params: thr=%.1f, min_speech=%dms, min_silence=%dms, pad=%dms%s\n",
           COLOR_CYAN, current_time_str().c_str(),
           vad_params.threshold,
           vad_params.min_speech_duration_ms,
           vad_params.min_silence_duration_ms,
           vad_params.speech_pad_ms,
           COLOR_RESET);

    // ============================================================
    // 4. 启动录音
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
        safe_exit(1);
    }

    // ============================================================
    // 程序启动完毕
    // ============================================================
    printf(CLEAR_SCREEN);
    printf("%s╔════════════════════════════════════════════════════╗%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("%s║    🎙  Real-time ASR + Semantic Understanding     ║%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("%s║    Press Ctrl+C to exit                           ║%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("%s╚════════════════════════════════════════════════════╝%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("\n");
    printf("%s[%s] 🔴 Recording started (16kHz, 32-bit float)%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);
    printf("%s──────────────────────────────────────────────────%s\n", COLOR_CYAN, COLOR_RESET);
    g_graceful_sigint_enabled = 1;

    // ============================================================
    // 5. VAD + ASR 主循环
    // ============================================================

    enum State {
        STATE_IDLE,
        STATE_SPEECHING,
    };

    State state = STATE_IDLE;

    std::vector<float> speechBuffer;
    int speechCount = 0;
    int vadStartCount = 0;
    int speechFrames = 0;

    const int SILENCE_FRAMES_THRESHOLD = 10;
    int silenceFrames = 0;

    const int MIN_SPEECH_FRAMES = 12;            // 需要更多连续帧才触发
    const int MAX_SPEECH_FRAMES = 30 * 1000 / 32;

    const float SPEECH_RMS_THRESHOLD = 0.025f;   // 提高能量门槛，小环境音不触发
    float noiseFloor = 0.025f;

    const int PAD_FRAMES = 8;
    std::vector<float> padBuffer;

    // 先向 VAD 输入静音帧做校准
    printf("%s[%s] Calibrating VAD (300ms of silence)...%s\n",
           COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);

    std::vector<float> silenceInput(SAMPLES_PER_VAD_FRAME, 0.0f);
    for (int i = 0; i < 5; ++i) {
        whisper_vad_detect_speech_no_reset(vctx, silenceInput.data(), SAMPLES_PER_VAD_FRAME);
    }
    whisper_vad_reset_state(vctx);

    printf("%s[%s] VAD calibrated.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    while (running && !g_stop_requested) {
        std::vector<float> chunk;
        size_t got = ringBuffer.consume(chunk, SAMPLES_PER_VAD_FRAME);

        if (got == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        float rms = compute_rms(chunk.data(), (int)chunk.size());
        float peak = compute_peak(chunk.data(), (int)chunk.size());

        if (state == STATE_IDLE && rms < noiseFloor) {
            noiseFloor = 0.999f * noiseFloor + 0.001f * rms;
        }

        bool vadSpeech = whisper_vad_detect_speech_no_reset(vctx,
                chunk.data(), (int)chunk.size());

        float adaptiveThreshold = std::max(noiseFloor * 5.0f, SPEECH_RMS_THRESHOLD);
        bool energyHasSpeech = (rms >= adaptiveThreshold);

        bool isSpeech = vadSpeech && energyHasSpeech;

        // VAD 调试日志（每 30 帧或说话时打印）
        static int logCounter = 0;
        if (VAD_DEBUG_LOG && (++logCounter % 30 == 0 || state == STATE_SPEECHING)) {
            printf("%s[DEBUG] frame: vad=%d rms=%.4f peak=%.4f thr=%.4f noise=%.4f energy=%d isSpeech=%d state=%s silence=%d%s\n",
                   COLOR_CYAN,
                   (int)vadSpeech, rms, peak, adaptiveThreshold, noiseFloor,
                   (int)energyHasSpeech, (int)isSpeech,
                   (state == STATE_IDLE) ? "IDLE" : "SPEECH",
                   silenceFrames,
                   COLOR_RESET);
        }

        // 更新 padding 环形缓冲
        padBuffer.insert(padBuffer.end(), chunk.begin(), chunk.end());
        if (padBuffer.size() > (size_t)PAD_FRAMES * SAMPLES_PER_VAD_FRAME) {
            padBuffer.erase(padBuffer.begin(), padBuffer.begin() + SAMPLES_PER_VAD_FRAME);
        }

        switch (state) {
            case STATE_IDLE: {
                if (isSpeech) {
                    state = STATE_SPEECHING;
                    speechFrames = 0;
                    silenceFrames = 0;
                    vadStartCount++;

                    speechBuffer.clear();
                    speechBuffer.insert(speechBuffer.end(), padBuffer.begin(), padBuffer.end());
                    speechFrames += (int)(padBuffer.size() / SAMPLES_PER_VAD_FRAME);

                    speechBuffer.insert(speechBuffer.end(), chunk.begin(), chunk.end());
                    speechFrames++;

                    printf("\n%s▶ [%s] Voice activity detected (rms=%.4f, peak=%.4f, noise=%.4f)%s\n",
                           COLOR_YELLOW, current_time_str().c_str(),
                           rms, peak, noiseFloor, COLOR_RESET);
                }
                break;
            }

            case STATE_SPEECHING: {
                speechBuffer.insert(speechBuffer.end(), chunk.begin(), chunk.end());
                speechFrames++;

                if (isSpeech) {
                    silenceFrames = 0;
                } else {
                    silenceFrames++;
                }

                if (silenceFrames >= SILENCE_FRAMES_THRESHOLD) {
                    if (speechFrames >= MIN_SPEECH_FRAMES) {
                        speechBuffer.insert(speechBuffer.end(),
                            padBuffer.begin(), padBuffer.end());
                        goto do_transcribe;
                    } else {
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
                    printf("%s⏰ [%s] Speech too long (%.0fs), force-transcribing...%s\n",
                           COLOR_YELLOW, current_time_str().c_str(),
                           speechFrames * 32.0f / 1000.0f, COLOR_RESET);
                    whisper_vad_reset_state(vctx);
                    goto do_transcribe;
                }
                break;
            }
        }

        continue;

    do_transcribe:
        {
            // 高通滤波
            highpass_filter(speechBuffer, 80.0f, SAMPLE_RATE);

            double durSec = speechBuffer.size() / (double)SAMPLE_RATE;
            float segmentRms = compute_rms(speechBuffer.data(), speechBuffer.size());
            float segmentPeak = compute_peak(speechBuffer.data(), speechBuffer.size());

            if (should_skip_low_quality_segment(durSec, segmentRms, segmentPeak)) {
                printf("%s⚠ [%s] Ignoring low-quality speech segment (%.1fs, rms=%.4f, peak=%.4f)%s\n",
                       COLOR_YELLOW, current_time_str().c_str(), durSec,
                       segmentRms, segmentPeak, COLOR_RESET);
                speechBuffer.clear();
                speechFrames = 0;
                silenceFrames = 0;
                state = STATE_IDLE;
                printf("%s──────────────────────────────────────────────────%s\n",
                       COLOR_CYAN, COLOR_RESET);
                continue;
            }

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

            wparams.no_context       = true;
            wparams.single_segment   = false;
            wparams.suppress_blank   = true;
            wparams.suppress_nst     = true;

            wparams.temperature      = 0.0f;
            wparams.temperature_inc  = 0.4f;
            wparams.entropy_thold    = 1.2f;
            wparams.logprob_thold    = -1.0f;
            wparams.no_speech_thold  = 0.5f;
            wparams.max_len          = 40;

            wparams.beam_search.beam_size = 7;

            const char * chinese_prompt_v2 =
                "你好世界欢迎使用语音识别系统"
                "一二三四五六七八九十百千万亿"
                "今天天气很好请问有什么可以帮助你的"
                "打开关闭保存删除复制粘贴撤回发送"
                "他说她说他们说我们在你们他们在"
                "能不能要不要会不会可以不可以";

            wparams.initial_prompt        = chinese_prompt_v2;
            wparams.carry_initial_prompt  = false;
            wparams.audio_ctx             = 0;
            wparams.tdrz_enable           = false;

            whisper_vad_reset_state(vctx);

            auto t0 = std::chrono::high_resolution_clock::now();

            int ret = whisper_full(ctx, wparams,
                    speechBuffer.data(), (int)speechBuffer.size());

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::string asr_text;

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
                        asr_text += text;
                    }
                }
                printf("%s\n", COLOR_RESET);
                printf("%s  └─────────────────────────────────────────────┘%s\n",
                       COLOR_BOLD COLOR_GREEN, COLOR_RESET);
                printf("\n");

                // ============================================================
                // ★ 语义理解：将 ASR 结果送入 LLM 处理
                // ============================================================
                if (!asr_text.empty()) {
                    // 移除首尾空白
                    while (!asr_text.empty() && (asr_text.back() == ' ' || asr_text.back() == '\n'))
                        asr_text.pop_back();
                    while (!asr_text.empty() && (asr_text.front() == ' ' || asr_text.front() == '\n'))
                        asr_text.erase(0, 1);

                    if (looks_like_asr_prompt_artifact(asr_text)) {
                        printf("%s⚠ [%s] Ignoring non-command/noisy ASR: \"%s\"%s\n",
                               COLOR_YELLOW, current_time_str().c_str(),
                               asr_text.c_str(), COLOR_RESET);
                        goto after_semantic;
                    }

                    static std::string lastSubmittedText;
                    static auto lastSubmittedAt = std::chrono::steady_clock::now() - std::chrono::seconds(60);
                    auto now = std::chrono::steady_clock::now();
                    if (asr_text == lastSubmittedText &&
                        now - lastSubmittedAt < std::chrono::seconds(20)) {
                        printf("%s⚠ [%s] Ignoring duplicate ASR: \"%s\"%s\n",
                               COLOR_YELLOW, current_time_str().c_str(),
                               asr_text.c_str(), COLOR_RESET);
                        goto after_semantic;
                    }
                    lastSubmittedText = asr_text;
                    lastSubmittedAt = now;

                    printf("%s🧠 [%s] Sending to semantic engine: \"%s\"%s\n",
                           COLOR_MAGENTA, current_time_str().c_str(),
                           asr_text.c_str(), COLOR_RESET);

                    semanticEngine.process_asr_result(asr_text, [](const Action& action) {
                        // Action 处理完成后的回调
                        if (action.type != ActionType::NONE) {
                            printf("%s⚡ [Action] %s: %s%s\n",
                                   COLOR_BOLD COLOR_GREEN,
                                   action.action_name.c_str(),
                                   action.target.c_str(),
                                   COLOR_RESET);

                            execute_action(action);
                        }

                        // 显示 LLM 回复
                        if (!action.response_text.empty()) {
                            printf("\n%s🤖 %s%s%s\n",
                                   COLOR_BOLD COLOR_CYAN, action.response_text.c_str(),
                                   COLOR_RESET, COLOR_RESET);

                            // ★ TTS 合成由 SemanticEngine 内部自动完成
                        }
                    });
                }
            after_semantic:
                ;
            } else {
                printf("%s❌ [%s] Transcription failed (ret=%d)%s\n",
                       COLOR_RED, current_time_str().c_str(), ret, COLOR_RESET);
            }

            // 重置 VAD 状态和所有变量
            speechBuffer.clear();
            speechFrames = 0;
            silenceFrames = 0;
            state = STATE_IDLE;

            printf("%s──────────────────────────────────────────────────%s\n",
                   COLOR_CYAN, COLOR_RESET);
        }
    }

    // ============================================================
    // 6. 清理
    // ============================================================
    printf("\n%s[%s] Shutting down...%s\n", COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);
    printf("%s[%s] Total detections: %d, transcriptions: %d%s\n",
           COLOR_CYAN, current_time_str().c_str(), vadStartCount, speechCount, COLOR_RESET);
    recorder.stop();
    whisper_vad_free(vctx);
    whisper_free(ctx);

    // 使用 safe_exit 而非 exit()/return，避免 Python finalization crash。
    // 详见 safe_exit() 注释。
    if (g_stop_requested) {
        safe_exit(130);
    }

    safe_exit(0);
}

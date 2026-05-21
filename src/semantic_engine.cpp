#include "semantic_engine.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <vector>
#include <thread>
#include <cstdlib>
#include <chrono>

namespace {

std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    };
    while (!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
    return s;
}

std::string find_json_object(const std::string& text) {
    size_t start = text.find('{');
    if (start == std::string::npos) return "";

    bool in_string = false;
    bool escaped = false;
    int depth = 0;

    for (size_t i = start; i < text.size(); ++i) {
        char c = text[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\' && in_string) {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;

        if (c == '{') {
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0) {
                return text.substr(start, i - start + 1);
            }
        }
    }

    return "";
}

std::string json_string_value(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) return "";

    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return "";

    size_t quote = json.find('"', colon + 1);
    if (quote == std::string::npos) return "";

    std::string out;
    bool escaped = false;
    for (size_t i = quote + 1; i < json.size(); ++i) {
        char c = json[i];
        if (escaped) {
            switch (c) {
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                default: out.push_back(c); break;
            }
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            return out;
        }
        out.push_back(c);
    }

    return "";
}

float json_float_value(const std::string& json, const std::string& key, float fallback) {
    std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) return fallback;

    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return fallback;

    size_t start = json.find_first_of("-0123456789", colon + 1);
    if (start == std::string::npos) return fallback;

    size_t end = json.find_first_not_of("0123456789.-", start);
    try {
        return std::stof(json.substr(start, end - start));
    } catch (...) {
        return fallback;
    }
}

ActionType action_type_from_name(const std::string& action_name) {
    if (action_name == "open_app") return ActionType::OPEN_APP;
    if (action_name == "search_web") return ActionType::SEARCH_WEB;
    if (action_name == "send_message") return ActionType::SEND_MESSAGE;
    if (action_name == "get_time") return ActionType::GET_TIME;
    if (action_name == "set_reminder") return ActionType::SET_REMINDER;
    if (action_name == "play_music") return ActionType::PLAY_MUSIC;
    if (action_name == "custom") return ActionType::CUSTOM;
    return ActionType::NONE;
}

}

SemanticEngine::SemanticEngine()
    : m_conversation(8) {
    // 初始化 Action 处理器为默认
    for (int i = 0; i < 10; ++i) {
        m_action_handlers[i] = nullptr;
    }
}

SemanticEngine::~SemanticEngine() {
    shutdown();
}

bool SemanticEngine::init(const std::string& model_path,
                          int n_ctx, int n_threads, bool use_gpu) {
    fprintf(stdout, "[SemanticEngine] Initializing, model=%s, ctx=%d, threads=%d, gpu=%d\n",
            model_path.c_str(), n_ctx, n_threads, (int)use_gpu);

    // 1. 加载 LLM 模型
    if (!m_llm.load_model(model_path, n_ctx, n_threads, use_gpu)) {
        fprintf(stderr, "[SemanticEngine] Failed to load LLM model\n");
        return false;
    }

    fprintf(stdout, "[SemanticEngine] LLM loaded: %s\n", m_llm.model_info().c_str());

    // 2. 设置系统提示词
    std::string system_prompt =
        "你是一个智能语音助手，运行在 macOS 系统上。\n"
        "你的任务是根据用户的语音输入理解语义，给出简短回复，并在安全时给出动作。\n"
        "\n"
        "你必须只输出一个 JSON 对象，不要使用 Markdown，不要输出代码块，不要额外解释。\n"
        "JSON 字段固定为：reply, action, target, params, confidence。\n"
        "action 只能是 none、open_app、search_web、get_time、custom。\n"
        "不允许输出 system_cmd。\n"
        "reply 使用中文，简洁自然。\n"
        "params 如果没有额外参数就填空字符串。\n"
        "confidence 是 0 到 1 的数字。\n"
        "\n"
        "例如：\n"
        "用户说\"打开计算器\"，输出 {\"reply\":\"好的，正在为你打开计算器。\",\"action\":\"open_app\",\"target\":\"Calculator\",\"params\":\"\",\"confidence\":0.95}\n"
        "用户说\"帮我搜索 Python 教程\"，输出 {\"reply\":\"好的，正在搜索 Python 教程。\",\"action\":\"search_web\",\"target\":\"Python 教程\",\"params\":\"\",\"confidence\":0.95}\n"
        "用户说\"现在几点了\"，输出 {\"reply\":\"我来看看当前时间。\",\"action\":\"get_time\",\"target\":\"\",\"params\":\"\",\"confidence\":0.95}\n"
        "用户说\"你好\"，输出 {\"reply\":\"你好，有什么我可以帮你的吗？\",\"action\":\"none\",\"target\":\"\",\"params\":\"\",\"confidence\":0.9}\n"
        "\n"
        "注意：\n"
        "1. 必须准确理解用户意图，不要过度猜测\n"
        "2. 如果不确定，回复用户询问确认\n"
        "3. 只有用户明确要求打开应用、搜索网页、询问时间时才设置动作\n"
        "4. 如果用户说\"结束\"或\"退出\"，只回复告别，action 为 none";

    m_conversation.set_system_prompt(system_prompt);

    // 3. 启动任务队列
    if (!m_task_queue.start(&m_llm)) {
        fprintf(stderr, "[SemanticEngine] Failed to start task queue\n");
        m_llm.unload_model();
        return false;
    }

    // 4. 初始化 TTS 引擎（嵌入式 Qwen3-TTS / MLX-Audio）
    {
        const char* env_model_dir = std::getenv("QWEN_TTS_MODEL");
        const char* env_python_home = std::getenv("QWEN_TTS_PYTHON_HOME");
        const char* env_bridge_dir = std::getenv("QWEN_TTS_BRIDGE_DIR");

        std::string tts_model_dir = env_model_dir
            ? env_model_dir
            : "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16";
        std::string tts_python_home = env_python_home
            ? env_python_home
            : "/opt/homebrew/Caskroom/miniforge/base/envs/cosyvoice";
        std::string tts_bridge_dir = env_bridge_dir
            ? env_bridge_dir
            : "/Users/boby/work/study2/python";

        if (init_tts(tts_model_dir, tts_python_home, tts_bridge_dir)) {
            fprintf(stdout, "[SemanticEngine] TTS engine initialized (Embedded Qwen3-TTS)\n");
        } else {
            fprintf(stderr, "[SemanticEngine] TTS engine init failed (non-fatal). "
                    "Set QWEN_TTS_MODEL, QWEN_TTS_PYTHON_HOME, QWEN_TTS_BRIDGE_DIR if your paths differ.\n");
        }
    }

    fprintf(stdout, "[SemanticEngine] Initialization complete\n");
    return true;
}

void SemanticEngine::shutdown() {
    if (!m_task_queue.is_running() && !m_llm.is_loaded()) {
        return;
    }
    m_task_queue.stop();
    m_llm.unload_model();

    // TTS 引擎在调用析构函数时自动清理
    m_tts_initialized = false;

    fprintf(stdout, "[SemanticEngine] Shutdown\n");
}

void SemanticEngine::process_asr_result(const std::string& asr_text,
                                         ActionCallback callback) {
    if (asr_text.empty()) return;

    fprintf(stdout, "[SemanticEngine] Processing ASR: \"%s\"\n", asr_text.c_str());

    // 1. 将用户输入加入对话历史
    m_conversation.add_turn("user", asr_text);

    // 2. 构建 prompt
    std::string prompt = build_prompt(asr_text);

    // 3. 提交 LLM 推理任务
    m_task_queue.push_task(prompt, [this, callback](const LlmResult& result) {
        if (!result.success) {
            fprintf(stderr, "[SemanticEngine] LLM inference failed: %s\n",
                    result.error_msg.c_str());
            if (callback) {
                Action action;
                action.response_text = "语义模型这次没有生成有效回复。";
                callback(action);
            }
            return;
        }

        fprintf(stdout, "[SemanticEngine] LLM response (%d tokens, %.0fms):\n%s\n",
                result.token_count, result.elapsed_ms,
                result.text.c_str());

        // 4. 解析 LLM 输出
        Action action = parse_llm_response(result.text);

        // 5. 将助手回复加入对话历史
        if (!action.response_text.empty()) {
            m_conversation.add_turn("assistant", action.response_text);
        } else if (!result.text.empty()) {
            m_conversation.add_turn("assistant", result.text);
        }

        // 6. 分发 Action
        if (action.type != ActionType::NONE) {
            dispatch_action(action);
        }

        // 7. TTS 语音合成并播放
        if (m_auto_speak && m_tts_initialized && !action.response_text.empty()) {
            speak(action.response_text, m_default_spk_id);
        }

        // 8. 回调
        if (callback) {
            callback(action);
        }
    });

}

void SemanticEngine::set_action_handler(ActionType type, ActionCallback handler) {
    std::lock_guard<std::mutex> lock(m_handler_mutex);
    int idx = static_cast<int>(type);
    if (idx >= 0 && idx < 10) {
        m_action_handlers[idx] = handler;
    }
}

void SemanticEngine::dispatch_action(const Action& action) {
    // 默认处理
    default_action_handler(action);

    // 自定义处理器
    std::lock_guard<std::mutex> lock(m_handler_mutex);
    int idx = static_cast<int>(action.type);
    if (idx >= 0 && idx < 10 && m_action_handlers[idx]) {
        m_action_handlers[idx](action);
    }
}

Action SemanticEngine::parse_llm_response(const std::string& llm_output) {
    Action action;

    std::string json_str = find_json_object(llm_output);
    if (json_str.empty()) {
        action.response_text = trim_copy(llm_output);
        return action;
    }

    fprintf(stdout, "[SemanticEngine] Parsing JSON: %s\n", json_str.c_str());

    action.response_text = trim_copy(json_string_value(json_str, "reply"));
    action.action_name = trim_copy(json_string_value(json_str, "action"));
    action.target = trim_copy(json_string_value(json_str, "target"));
    action.params = trim_copy(json_string_value(json_str, "params"));
    action.confidence = json_float_value(json_str, "confidence", 0.0f);

    if (action.action_name.empty()) {
        action.action_name = "none";
    }

    action.type = action_type_from_name(action.action_name);
    if (action.action_name == "system_cmd") {
        action.type = ActionType::NONE;
        action.target.clear();
        action.params.clear();
        if (action.response_text.empty()) {
            action.response_text = "这个操作不在当前安全白名单里。";
        }
    }

    return action;
}

std::string SemanticEngine::build_prompt(const std::string& user_input) {
    (void)user_input;

    std::vector<std::pair<std::string, std::string>> messages;
    std::string system_prompt = m_conversation.get_system_prompt();
    if (!system_prompt.empty()) {
        messages.push_back({"system", system_prompt});
    }

    for (const auto& turn : m_conversation.get_turns()) {
        if (turn.role == "user" || turn.role == "assistant") {
            messages.push_back({turn.role, turn.content});
        }
    }

    std::string prompt = m_llm.format_chat_prompt(messages);
    if (!prompt.empty()) {
        return prompt;
    }

    return m_conversation.get_context();
}

void SemanticEngine::default_action_handler(const Action& action) {
    const char* type_names[] = {
        "NONE", "OPEN_APP", "SEARCH_WEB", "SEND_MESSAGE",
        "GET_TIME", "SET_REMINDER", "PLAY_MUSIC", "SYSTEM_CMD", "CUSTOM"
    };

    int idx = static_cast<int>(action.type);
    const char* type_name = (idx >= 0 && idx < 9) ? type_names[idx] : "UNKNOWN";

    fprintf(stdout, "[Action] type=%s action=\"%s\" target=\"%s\" params=\"%s\" response=\"%s\"\n",
            type_name,
            action.action_name.c_str(),
            action.target.c_str(),
            action.params.c_str(),
            action.response_text.c_str());
}

// ============================================================================
// TTS — Embedded Python + Qwen3-TTS
// ============================================================================

bool SemanticEngine::init_tts(const std::string& model_dir,
                               const std::string& python_home,
                               const std::string& bridge_script_dir) {
    if (!m_tts_engine.init(model_dir, python_home, bridge_script_dir)) {
        fprintf(stderr, "[TTS] m_tts_engine.init() failed\n");
        return false;
    }

    m_tts_sample_rate = 24000;
    if (const char* env_sample_rate = std::getenv("TTS_SAMPLE_RATE")) {
        int configured_sample_rate = std::atoi(env_sample_rate);
        if (configured_sample_rate > 0) {
            m_tts_sample_rate = configured_sample_rate;
        }
    }

    // 初始化音频播放器（macOS AudioQueue）
    if (!m_audio_player.init(m_tts_sample_rate)) {
        fprintf(stderr, "[TTS] AudioPlayer init failed\n");
        return false;
    }

    m_tts_initialized = true;
    return true;
}

bool SemanticEngine::synthesize_text(const std::string& text,
                                      std::vector<int16_t>& out_pcm,
                                      const std::string& spk_id) {
    if (!m_tts_initialized) {
        fprintf(stderr, "[TTS] Engine not initialized\n");
        return false;
    }

    // 调用 TTS 引擎合成（同步接口 synthesize_sync）
    if (!m_tts_engine.synthesize_sync(text, out_pcm, spk_id)) {
        fprintf(stderr, "[TTS] Synthesis failed for: \"%s\"\n", text.c_str());
        return false;
    }

    return true;
}

void SemanticEngine::speak(const std::string& text, const std::string& spk_id) {
    {
        std::lock_guard<std::mutex> lock(m_tts_mutex);
        m_tts_active_jobs++;
    }

    // 在独立线程中播放，不阻塞 LLM 推理；串行化 Python TTS 调用和 AudioQueue 播放
    std::thread([this, text, spk_id]() {
        {
            std::lock_guard<std::mutex> serial_lock(m_tts_serial_mutex);

            fprintf(stdout, "[TTS] Streaming reply: \"%s\"\n", text.c_str());

            size_t total_samples = 0;
            size_t buffered_samples = 0;
            int chunk_count = 0;
            bool playback_started = false;
            std::vector<int16_t> pending_pcm;

            double start_buffer_sec = 0.35;
            if (const char* env_buffer = std::getenv("TTS_STREAM_START_BUFFER_SEC")) {
                try {
                    start_buffer_sec = std::max(0.0, std::atof(env_buffer));
                } catch (...) {
                    start_buffer_sec = 0.35;
                }
            }
            const size_t start_buffer_samples = (size_t)(start_buffer_sec * (double)m_tts_sample_rate);

            auto start_playback = [&]() {
                if (playback_started || pending_pcm.empty()) return;
                m_audio_player.start_stream();
                m_audio_player.play(pending_pcm);
                playback_started = true;
                fprintf(stdout, "[TTS] Playback started after buffering %.2f sec (%zu samples)\n",
                        (double)pending_pcm.size() / (double)m_tts_sample_rate, pending_pcm.size());
                pending_pcm.clear();
            };

            bool ok = m_tts_engine.synthesize_stream(text, spk_id,
                [this, &total_samples, &buffered_samples, &chunk_count,
                 &pending_pcm, &playback_started, start_buffer_samples, &start_playback]
                (const std::vector<int16_t>& pcm) {
                    total_samples += pcm.size();
                    chunk_count++;
                    fprintf(stdout, "[TTS] Streaming chunk #%d: %zu samples (total %.2f sec)\n",
                            chunk_count, pcm.size(), (double)total_samples / (double)m_tts_sample_rate);

                    if (!playback_started) {
                        pending_pcm.insert(pending_pcm.end(), pcm.begin(), pcm.end());
                        buffered_samples = pending_pcm.size();
                        if (buffered_samples >= start_buffer_samples) {
                            start_playback();
                        }
                    } else {
                        m_audio_player.play(pcm);
                    }
                    return true;
                });

            if (!playback_started) {
                start_playback();
            }
            m_audio_player.finish_stream();
            if (ok) {
                fprintf(stdout, "[TTS] Stream playback queued: %zu samples in %d chunks (%.2f sec)\n",
                        total_samples, chunk_count, (double)total_samples / (double)m_tts_sample_rate);
                m_audio_player.wait_for_finish();
            } else {
                fprintf(stderr, "[TTS] Stream synthesis failed for: \"%s\"\n", text.c_str());
            }
        }

        {
            std::lock_guard<std::mutex> lock(m_tts_mutex);
            m_tts_active_jobs--;
        }
        m_tts_cv.notify_all();
    }).detach();
}

void SemanticEngine::wait_for_tts() {
    std::unique_lock<std::mutex> lock(m_tts_mutex);
    m_tts_cv.wait(lock, [this]() {
        return m_tts_active_jobs == 0;
    });
}

bool SemanticEngine::wait_for_tts_for(int timeout_ms) {
    std::unique_lock<std::mutex> lock(m_tts_mutex);
    return m_tts_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
        return m_tts_active_jobs == 0;
    });
}

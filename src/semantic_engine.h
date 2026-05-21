#ifndef SEMANTIC_ENGINE_H
#define SEMANTIC_ENGINE_H

#include "llm_client.h"
#include "llm_task.h"
#include "conversation.h"
#include "tts_engine.h"
#include "audio_player.h"

#include <string>
#include <functional>
#include <mutex>
#include <memory>
#include <vector>
#include <cstdint>

// ============================================================
// 语义引擎
// 1) 接收 ASR 识别结果（语音转文字）
// 2) 用 LLM 理解语义
// 3) 解析 LLM 输出中的 Action JSON
// 4) 分发 Action 到对应执行器
// 5) TTS 语音合成 + 音频播放
// ============================================================

// ---- Action 定义 ----

enum class ActionType {
    NONE,           // 无操作（日常聊天等）
    OPEN_APP,       // 打开应用
    SEARCH_WEB,     // 搜索网页
    SEND_MESSAGE,   // 发送消息
    GET_TIME,       // 获取时间
    SET_REMINDER,   // 设置提醒
    PLAY_MUSIC,     // 播放音乐
    SYSTEM_CMD,     // 系统命令
    CUSTOM,         // 自定义操作
};

struct Action {
    ActionType type = ActionType::NONE;
    std::string action_name;     // 操作名称
    std::string target;          // 操作目标（如应用名、搜索词）
    std::string params;          // 额外参数（JSON 字符串）
    std::string response_text;   // LLM 生成的回复文本
    float confidence = 0.0f;     // 置信度
};

using ActionCallback = std::function<void(const Action&)>;

// ---- 语义引擎类 ----

class SemanticEngine {
public:
    SemanticEngine();
    ~SemanticEngine();

    // 初始化（加载 LLM 模型）
    bool init(const std::string& model_path,
              int n_ctx = 4096,
              int n_threads = 4,
              bool use_gpu = true);

    // 关闭
    void shutdown();

    // 处理 ASR 结果（语音转文字输入）
    // asr_text: 语音识别的文本
    // callback: 处理完成后的回调（可选，在主线程调用）
    void process_asr_result(const std::string& asr_text, ActionCallback callback = nullptr);

    // 设置自定义 Action 处理器
    void set_action_handler(ActionType type, ActionCallback handler);

    // 获取当前对话管理器引用
    ConversationManager& conversation() { return m_conversation; }

    // 获取 LLM 客户端引用
    LlmClient& llm_client() { return m_llm; }

    // LLM 是否已加载
    bool is_loaded() const { return m_llm.is_loaded(); }

    // 任务队列状态
    bool is_processing() const { return m_task_queue.is_running(); }

    // 设置 TTS 自动朗读（默认开启）
    void set_auto_speak(bool enable) { m_auto_speak = enable; }

    /// 等待当前 TTS 播放完成（text mode 优雅退出用）
    void wait_for_tts();

    /// TTS 引擎是否就绪
    bool tts_ready() const { return m_tts_initialized; }

private:
    // ──────────────────────────────────────────────────────
    // TTS 内部方法
    // ──────────────────────────────────────────────────────

    /// 初始化 TTS 引擎（嵌入式 CosyVoice）
    bool init_tts(const std::string& model_dir,
                  const std::string& python_home,
                  const std::string& bridge_script_dir);

    /// 合成文本为 PCM 音频数据
    bool synthesize_text(const std::string& text,
                         std::vector<int16_t>& out_pcm,
                         const std::string& spk_id = "中文女");

    /// 合成并播放（异步，不阻塞 LLM 推理）
    void speak(const std::string& text, const std::string& spk_id = "中文女");

    // ──────────────────────────────────────────────────────
    // 成员
    // ──────────────────────────────────────────────────────

    // 解析 LLM 输出中的 Action JSON
    // 解析 LLM 输出中的 Action JSON
    Action parse_llm_response(const std::string& llm_output);

    // 默认 Action 处理器（显示日志和打印）
    void default_action_handler(const Action& action);

    // 构建 prompt（含系统提示+对话历史+当前输入）
    std::string build_prompt(const std::string& user_input);

    // 分发 Action
    void dispatch_action(const Action& action);

    LlmClient              m_llm;
    LlmTaskQueue           m_task_queue;
    ConversationManager    m_conversation;

    // Action 处理器映射
    ActionCallback m_action_handlers[10];
    std::mutex     m_handler_mutex;

    // ──────────────────────────────────────────────────────
    // TTS 相关
    // ──────────────────────────────────────────────────────
    TtsEngine              m_tts_engine;
    AudioPlayer            m_audio_player;
    bool                   m_tts_initialized{false};
    bool                   m_auto_speak{true};     // 自动朗读回复
    std::string            m_default_spk_id{"中文女"};
    mutable std::mutex     m_tts_mutex;
    std::condition_variable m_tts_cv;
    int                    m_tts_active_jobs{0};
    std::mutex             m_tts_serial_mutex;
};

#endif // SEMANTIC_ENGINE_H

#ifndef LLM_CLIENT_H
#define LLM_CLIENT_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <utility>

// llama.h 由 vendor/llama.cpp/include 提供
#include "llama.h"

// ============================================================
// LLM 推理结果回调
// ============================================================
struct LlmResult {
    bool   success = false;
    std::string text;
    double elapsed_ms  = 0.0;
    int    token_count = 0;
    std::string error_msg;
};

using LlmCallback = std::function<void(const LlmResult&)>;

// ============================================================
// LLM 推理参数
// ============================================================
struct LlmInferenceParams {
    std::string prompt;
    int max_tokens        = 192;
    float temperature     = 0.2f;
    float top_p           = 0.8f;
    float top_k           = 40.0f;
    float repeat_penalty  = 1.1f;
    int   n_threads       = 4;
    bool  use_gpu         = true;

    // 回调（可选）
    LlmCallback callback = nullptr;

    // 唯一 ID（用于追踪）
    uint64_t id = 0;
};

// ============================================================
// LLM 客户端封装
// 封装 llama.cpp 的模型加载/卸载/推理
// 线程安全（仅推理本身是线程安全的，由调用者保证串行访问）
// ============================================================
class LlmClient {
public:
    LlmClient();
    ~LlmClient();

    // 加载模型（线程安全）
    bool load_model(const std::string& model_path, int n_ctx = 4096, int n_threads = 4, bool use_gpu = true);

    // 卸载模型（线程安全）
    void unload_model();

    // 是否已加载
    bool is_loaded() const;

    // 获取模型信息
    std::string model_info() const;

    // 同步推理（在当前线程阻塞执行）
    // 注意：同一时刻只能有一个推理进行；调用者需保证线程安全
    LlmResult infer_sync(const LlmInferenceParams& params);

    // 获取模型描述
    std::string model_name() const;

    // 使用模型自带 chat template 构建 prompt
    std::string format_chat_prompt(const std::vector<std::pair<std::string, std::string>>& messages) const;

private:
    // 实际的推理逻辑
    LlmResult run_inference(const LlmInferenceParams& params);

    void unload_model_locked();

    mutable std::mutex m_mutex;

    // llama 对象
    struct llama_model    * m_model    = nullptr;
    struct llama_context  * m_ctx      = nullptr;
    struct llama_vocab    * m_vocab    = nullptr;
    struct llama_sampler  * m_sampler  = nullptr;

    std::string m_model_path;
    int  m_n_ctx     = 4096;
    int  m_n_threads = 4;
    bool m_use_gpu   = true;

    // 统计
    int  m_n_loaded_tokens = 0;
};

#endif // LLM_CLIENT_H

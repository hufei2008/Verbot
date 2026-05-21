#ifndef LLM_TASK_H
#define LLM_TASK_H

#include "llm_client.h"

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <memory>

// ============================================================
// 异步 LLM 任务队列
// 1) 后台线程串行处理推理请求
// 2) 调用者 push_task 后立即返回
// 3) 完成时通过回调通知调用者
// ============================================================

struct LlmTaskItem {
    LlmInferenceParams params;
    LlmCallback        callback;   // 推理完成的回调
};

class LlmTaskQueue {
public:
    LlmTaskQueue();
    ~LlmTaskQueue();

    // 启动后台推理线程（必须先加载模型）
    bool start(LlmClient* client);

    // 停止后台推理线程
    void stop();

    // 提交推理任务（线程安全）
    // prompt: 用户输入的文本
    // callback: 推理完成后的回调（可选）
    // 返回任务 ID
    uint64_t push_task(const std::string& prompt, LlmCallback callback = nullptr);

    // 是否正在运行
    bool is_running() const { return m_running; }

    // 队列中的任务数
    size_t queue_size() const;

    // 清空队列
    void clear_queue();

private:
    // 后台线程主循环
    void worker_loop();

    LlmClient* m_client = nullptr;

    std::queue<LlmTaskItem> m_queue;
    mutable std::mutex      m_mutex;
    std::condition_variable m_cv;
    std::thread             m_worker;
    std::atomic<bool>       m_running{false};
    std::atomic<uint64_t>   m_next_id{1};
};

#endif // LLM_TASK_H
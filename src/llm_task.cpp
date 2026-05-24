#include "llm_task.h"
#include <cstdio>

// ============================================================
// LlmTaskQueue — LLM 异步推理任务队列
//
// 生产者-消费者模型：主线程 push_task 提交推理请求，
// 后台 worker 线程串行执行 LLM 推理并通过回调返回结果。
// 线程安全，支持启动/停止/清空队列。
// ============================================================

LlmTaskQueue::LlmTaskQueue() {}

LlmTaskQueue::~LlmTaskQueue() {
    stop();
}

// 启动任务队列，创建后台 worker 线程
bool LlmTaskQueue::start(LlmClient* client) {
    if (m_running) {
        fprintf(stderr, "[TaskQueue] Already running\n");
        return false;
    }
    if (!client || !client->is_loaded()) {
        fprintf(stderr, "[TaskQueue] LLM client not loaded\n");
        return false;
    }

    m_client  = client;
    m_running = true;
    // 创建后台线程：循环等待并执行推理任务
    m_worker  = std::thread(&LlmTaskQueue::worker_loop, this);

    fprintf(stdout, "[TaskQueue] Worker thread started\n");
    return true;
}

// 停止任务队列：通知 worker 退出，join 等待线程结束
void LlmTaskQueue::stop() {
    if (!m_running) return;

    m_running = false;
    m_cv.notify_one();  // 唤醒 worker 让其检查 m_running 标志

    if (m_worker.joinable()) {
        m_worker.join();
    }

    // 清空队列（丢弃未处理的任务）
    std::lock_guard<std::mutex> lock(m_mutex);
    while (!m_queue.empty()) {
        m_queue.pop();
    }

    fprintf(stdout, "[TaskQueue] Worker thread stopped\n");
}

// 提交推理任务到队列，返回任务 ID
uint64_t LlmTaskQueue::push_task(const std::string& prompt, LlmCallback callback) {
    LlmTaskItem item;
    item.params.prompt    = prompt;
    item.params.callback  = callback;
    item.params.id        = m_next_id++;  // 自增任务 ID

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(item);
    }

    m_cv.notify_one();  // 唤醒 worker 线程
    return item.params.id;
}

// 获取当前队列中等待执行的任务数量
size_t LlmTaskQueue::queue_size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.size();
}

// 清空队列中所有未执行的任务
void LlmTaskQueue::clear_queue() {
    std::lock_guard<std::mutex> lock(m_mutex);
    while (!m_queue.empty()) {
        m_queue.pop();
    }
}

// 后台工作线程主循环：阻塞等待任务 → 执行推理 → 回调通知
void LlmTaskQueue::worker_loop() {
    while (m_running) {
        LlmTaskItem item;

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            // 阻塞等待：有新任务 或 停止信号
            m_cv.wait(lock, [this]() {
                return !m_queue.empty() || !m_running;
            });

            if (!m_running) break;

            if (m_queue.empty()) continue;

            // 从队首取出一个任务
            item = m_queue.front();
            m_queue.pop();
        }

        // 执行 LLM 推理（同步，阻塞直到完成）
        fprintf(stdout, "[TaskQueue] Running inference (id=%llu)...\n",
                (unsigned long long)item.params.id);

        LlmResult result = m_client->infer_sync(item.params);

        fprintf(stdout, "[TaskQueue] Inference done (id=%llu, success=%d, tokens=%d, elapsed=%.0fms)\n",
                (unsigned long long)item.params.id,
                (int)result.success,
                result.token_count,
                result.elapsed_ms);

        // 通过 Lambda 回调通知调用者
        if (item.callback) {
            item.callback(result);
        }

        // 如果 params 中有 callback 也调用（兼容两种回调方式）
        if (item.params.callback) {
            item.params.callback(result);
        }
    }
}

#include "llm_task.h"
#include <cstdio>

LlmTaskQueue::LlmTaskQueue() {}

LlmTaskQueue::~LlmTaskQueue() {
    stop();
}

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
    m_worker  = std::thread(&LlmTaskQueue::worker_loop, this);

    fprintf(stdout, "[TaskQueue] Worker thread started\n");
    return true;
}

void LlmTaskQueue::stop() {
    if (!m_running) return;

    m_running = false;
    m_cv.notify_one();

    if (m_worker.joinable()) {
        m_worker.join();
    }

    // 清空队列
    std::lock_guard<std::mutex> lock(m_mutex);
    while (!m_queue.empty()) {
        m_queue.pop();
    }

    fprintf(stdout, "[TaskQueue] Worker thread stopped\n");
}

uint64_t LlmTaskQueue::push_task(const std::string& prompt, LlmCallback callback) {
    LlmTaskItem item;
    item.params.prompt    = prompt;
    item.params.callback  = callback;
    item.params.id        = m_next_id++;

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(item);
    }

    m_cv.notify_one();
    return item.params.id;
}

size_t LlmTaskQueue::queue_size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.size();
}

void LlmTaskQueue::clear_queue() {
    std::lock_guard<std::mutex> lock(m_mutex);
    while (!m_queue.empty()) {
        m_queue.pop();
    }
}

void LlmTaskQueue::worker_loop() {
    while (m_running) {
        LlmTaskItem item;

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this]() {
                return !m_queue.empty() || !m_running;
            });

            if (!m_running) break;

            if (m_queue.empty()) continue;

            item = m_queue.front();
            m_queue.pop();
        }

        // 执行推理
        fprintf(stdout, "[TaskQueue] Running inference (id=%llu)...\n",
                (unsigned long long)item.params.id);

        LlmResult result = m_client->infer_sync(item.params);

        fprintf(stdout, "[TaskQueue] Inference done (id=%llu, success=%d, tokens=%d, elapsed=%.0fms)\n",
                (unsigned long long)item.params.id,
                (int)result.success,
                result.token_count,
                result.elapsed_ms);

        // 回调
        if (item.callback) {
            item.callback(result);
        }

        // 如果 params 中有 callback 也调用
        if (item.params.callback) {
            item.params.callback(result);
        }
    }
}
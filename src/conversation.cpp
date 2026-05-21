#include "conversation.h"
#include <sstream>

ConversationManager::ConversationManager(int max_turns)
    : m_max_turns(max_turns) {}

void ConversationManager::add_turn(const std::string& role, const std::string& content) {
    std::lock_guard<std::mutex> lock(m_mutex);

    // 超过最大轮数时删除最早的用户/助手轮次（保留系统提示）
    while ((int)m_turns.size() >= m_max_turns) {
        // 从最早的非系统提示开始删除
        bool removed = false;
        for (auto it = m_turns.begin(); it != m_turns.end(); ++it) {
            if (it->role != "system") {
                m_turns.erase(it);
                removed = true;
                break;
            }
        }
        if (!removed) break;
    }

    m_turns.push_back({role, content});
}

std::string ConversationManager::get_context() const {
    std::lock_guard<std::mutex> lock(m_mutex);

    std::ostringstream oss;

    // 系统提示在最前面
    if (!m_system_prompt.empty()) {
        oss << "<|system|>\n" << m_system_prompt << "\n";
    }

    // 对话历史
    for (const auto& turn : m_turns) {
        if (turn.role == "user") {
            oss << "<|user|>\n" << turn.content << "\n";
        } else if (turn.role == "assistant") {
            oss << "<|assistant|>\n" << turn.content << "\n";
        } else if (turn.role == "system") {
            // 系统提示已经在上面预置了，这里跳过（避免重复）
            continue;
        }
    }

    // 最后添加上助手标记，等待模型完成
    oss << "<|assistant|>\n";

    return oss.str();
}

void ConversationManager::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_turns.clear();
}

void ConversationManager::set_system_prompt(const std::string& prompt) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_system_prompt = prompt;
}

int ConversationManager::turn_count() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return (int)m_turns.size();
}

std::vector<ConversationTurn> ConversationManager::get_turns() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_turns;
}
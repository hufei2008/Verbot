#ifndef CONVERSATION_H
#define CONVERSATION_H

#include <string>
#include <vector>
#include <mutex>

// ============================================================
// 对话上下文管理器
// 维护多轮对话历史，用于构建 LLM prompt 上下文
// ============================================================

struct ConversationTurn {
    std::string role;      // "user" | "assistant" | "system"
    std::string content;
};

class ConversationManager {
public:
    ConversationManager(int max_turns = 20);

    // 添加一条对话记录
    void add_turn(const std::string& role, const std::string& content);

    // 获取完整对话历史（格式化为 LLM prompt 上下文）
    std::string get_context() const;

    // 清空对话历史
    void clear();

    // 设置系统提示词
    void set_system_prompt(const std::string& prompt);

    // 获取系统提示词
    std::string get_system_prompt() const { return m_system_prompt; }

    // 获取对话轮数
    int turn_count() const;

    // 获取所有轮次
    std::vector<ConversationTurn> get_turns() const;

private:
    std::string m_system_prompt;
    int m_max_turns;

    std::vector<ConversationTurn> m_turns;
    mutable std::mutex m_mutex;
};

#endif // CONVERSATION_H
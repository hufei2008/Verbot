#include "asr_rejector.h"

#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <vector>

namespace {

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    };
    while (!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
    return s;
}

std::string strip_terminal_punctuation(std::string text) {
    text = trim_copy(text);
    const std::vector<std::string> punct = {
        "。", "，", "？", "！", ".", ",", "?", "!", "～", "~"
    };

    bool removed = true;
    while (removed) {
        removed = false;
        text = trim_copy(text);
        for (const auto& p : punct) {
            if (ends_with(text, p)) {
                text = text.substr(0, text.size() - p.size());
                removed = true;
                break;
            }
        }
    }
    return trim_copy(text);
}

bool contains_any(const std::string& text, const std::vector<std::string>& needles) {
    for (const auto& needle : needles) {
        if (text.find(needle) != std::string::npos) return true;
    }
    return false;
}

bool has_supported_signal(const std::string& text) {
    static const std::vector<std::string> signals = {
        "天气", "温度", "几点", "时间", "打开", "播放", "放一下", "放一首",
        "我要听", "我想听", "想听", "听一下", "听一首", "暂停", "继续",
        "下一首", "上一首", "网易云", "计算器", "设置", "搜索", "查一下",
        "查询", "什么", "为什么", "怎么", "哪里", "有哪些", "你好"
    };
    if (contains_any(text, signals)) return true;

    static const std::vector<std::string> cities = {
        "北京", "上海", "深圳", "天津", "杭州", "广州", "南京", "苏州",
        "成都", "重庆", "武汉", "西安"
    };
    return contains_any(text, cities) && contains_any(text, {"的呢", "呢"});
}

bool is_filler_or_ack(const std::string& text) {
    static const std::set<std::string> fillers = {
        "嗯", "嗯嗯", "啊", "啊啊", "哦", "喔", "哼", "哼哼",
        "咳", "咳咳", "咳咳咳", "呵", "呵呵", "呵呵呵", "呵呵呵呵",
        "哈", "哈哈", "哈哈哈", "哈哈哈哈", "嘿", "嘿嘿", "嘿嘿嘿",
        "嘻", "嘻嘻", "嘻嘻嘻", "啦", "啦啦", "啦啦啦",
        "好", "好的", "可以", "行"
    };
    return fillers.count(text) > 0;
}

bool is_prompt_artifact(const std::string& text) {
    if (text.find("打开关闭保存删除复制粘贴撤回发送") != std::string::npos) return true;
    if (text.find("请问有什么可以帮助你的") != std::string::npos) return true;
    if (text.find("请问有什么可以帮助您") != std::string::npos) return true;
    if (text.find("MING PAO CANADA") != std::string::npos) return true;
    if (text.find("BORN TO BE") != std::string::npos) return true;

    if (text.size() > 80 &&
        (text.find("能不能要不要会不会可以不可以") != std::string::npos ||
         text.find("今天天气很好请问有什么可以帮助你的") != std::string::npos)) {
        if (text.find("一二三四") != std::string::npos ||
            text.find("他说他们说") != std::string::npos) return true;
    }
    return false;
}

bool is_mostly_latin_noise(const std::string& text) {
    int latin = 0;
    int alnum = 0;
    int uppercase = 0;
    int chinese_bytes = 0;
    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            alnum++;
            if (std::isalpha(c)) latin++;
            if (std::isupper(c)) uppercase++;
        }
        if (c >= 0x80) chinese_bytes++;
    }
    if (alnum < 8) return false;
    if (chinese_bytes > 0) return false;
    return latin >= 8 && uppercase * 2 >= latin;
}

bool has_excessive_repetition(const std::string& text) {
    if (text.find("啊啊啊") != std::string::npos) return true;
    if (text.find("得得得") != std::string::npos) return true;
    if (text.find("天天天天") != std::string::npos) return true;
    if (text.find("呵呵呵") != std::string::npos) return true;
    if (text.find("哈哈哈") != std::string::npos) return true;
    if (text.find("嘿嘿嘿") != std::string::npos) return true;
    if (text.find("嘻嘻嘻") != std::string::npos) return true;
    if (text.find("啦啦啦") != std::string::npos) return true;
    if (text.find("他说他们说") != std::string::npos) return true;
    return false;
}

} // namespace

AsrRejectDecision reject_asr_text(const std::string& text) {
    AsrRejectFeatures features;
    features.text = text;
    return reject_asr_result(features);
}

AsrRejectDecision reject_asr_result(const AsrRejectFeatures& features) {
    std::string clean = strip_terminal_punctuation(features.text);

    if (clean.empty()) return {true, "empty"};
    if (is_filler_or_ack(clean)) return {true, "filler"};
    if (is_prompt_artifact(clean)) return {true, "prompt_artifact"};
    if (is_mostly_latin_noise(clean)) return {true, "latin_noise"};
    if (has_excessive_repetition(clean)) return {true, "repetition"};

    const bool has_signal = has_supported_signal(clean);
    if (features.no_speech_prob > 0.75f && !has_signal) {
        return {true, "no_speech"};
    }
    if (features.n_tokens > 0 && features.avg_token_p < 0.32f && !has_signal) {
        return {true, "low_avg_token_p"};
    }
    if (features.n_tokens > 0 && features.min_token_p < 0.05f &&
        features.avg_token_p < 0.45f && !has_signal) {
        return {true, "low_token_p"};
    }

    if (clean.size() <= 3 && !has_signal) {
        return {true, "too_short"};
    }

    return {false, ""};
}

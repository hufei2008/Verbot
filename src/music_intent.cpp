#include "music_intent.h"

#include <algorithm>
#include <vector>

// ============================================================
// 匿名 namespace 内的文本预处理工具函数
// 负责清洗 ASR 文本，提取音乐控制关键信息
// ============================================================
namespace {

// 去除字符串首尾空白字符（空格、换行、制表符等）
std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    };
    while (!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
    return s;
}

// 判断字符串是否以 prefix 开头
bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

// 判断字符串是否以 suffix 结尾
bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// 去除文本末尾的标点符号（支持中英文标点如 。，？！等）
// 循环去除直到尾部不再有标点
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

// 只在音乐意图解析内部修正 ASR 常见同音误识别：
// “播放周杰伦的歌”容易被识别成“播放周杰伦的哥/哥哥”。
std::string normalize_music_asr_text(std::string text) {
    text = strip_terminal_punctuation(text);
    const std::vector<std::pair<std::string, std::string>> suffix_rewrites = {
        {"的哥哥", "的歌"},
        {"的哥", "的歌"},
        {"的歌歌", "的歌"},
    };
    for (const auto& rewrite : suffix_rewrites) {
        if (ends_with(text, rewrite.first) && text.size() > rewrite.first.size()) {
            return text.substr(0, text.size() - rewrite.first.size()) + rewrite.second;
        }
    }
    return text;
}

// 从目标文本中剥离歌手后缀（如"周杰伦的歌" => "周杰伦"）
// 处理的模式：XX的歌、XX的歌曲、XX的音乐、XX歌曲、XX音乐
std::string strip_music_artist_suffix(std::string target) {
    target = normalize_music_asr_text(target);
    const std::vector<std::string> suffixes = {
        "的歌", "的歌曲", "的音乐", "歌曲", "音乐"
    };
    for (const auto& suffix : suffixes) {
        if (ends_with(target, suffix) && target.size() > suffix.size()) {
            return strip_terminal_punctuation(target.substr(0, target.size() - suffix.size()));
        }
    }
    return target;
}

// 判断文本是否为「歌手点播」请求（如"播放周杰伦的歌"）
// 通过检测是否包含"的歌""的歌曲""的音乐"等模式识别
bool is_artist_music_request(const std::string& text) {
    std::string clean = normalize_music_asr_text(text);
    return clean.find("的歌") != std::string::npos ||
           clean.find("的歌曲") != std::string::npos ||
           clean.find("的音乐") != std::string::npos ||
           ends_with(clean, "歌曲") ||
           ends_with(clean, "音乐");
}

} // namespace

// ============================================================
// 音乐意图快速匹配入口
// 绕过 LLM，直接在 ASR 文本层面进行规则匹配，
// 生成 PLAY_MUSIC 类型的 Action 填充到 TaskPlan
// ============================================================
bool build_fast_music_plan(const std::string& asr_text, TaskPlan& plan) {
    // 清洗标点，空文本直接返回不匹配
    std::string text = normalize_music_asr_text(asr_text);
    if (text.empty()) return false;

    // 构建默认的音乐播放 Action，高置信度 0.95
    Action action;
    action.type = ActionType::PLAY_MUSIC;
    action.action_name = "play_music";
    action.confidence = 0.95f;

    // ---- 精确匹配：应用控制指令 ----
    if (text == "打开网易云" || text == "打开网易云音乐") {
        action.target = "网易云音乐";
        action.params = "command=open;provider=netease_app";
        plan.reply = "已打开网易云音乐。";
    } else if (text == "暂停音乐" || text == "暂停播放" || text == "音乐暂停") {
        action.params = "command=pause;provider=netease_app";
        plan.reply = "已暂停音乐。";
    } else if (text == "继续播放" || text == "恢复播放") {
        action.params = "command=resume;provider=netease_app";
        plan.reply = "继续播放音乐。";
    } else if (text == "下一首" || text == "下一曲") {
        action.params = "command=next;provider=netease_app";
        plan.reply = "已切到下一首。";
    } else if (text == "上一首" || text == "上一曲") {
        action.params = "command=previous;provider=netease_app";
        plan.reply = "已切到上一首。";
    } else if (text == "播放音乐" || text == "放音乐") {
        action.params = "command=play;provider=netease_app";
        plan.reply = "开始播放音乐。";
    } else if (starts_with(text, "播放") || starts_with(text, "放一下") || starts_with(text, "放一首")) {
        // 提取播放目标（歌手名或歌曲名）
        std::string target = text;
        if (starts_with(target, "播放")) target = target.substr(std::string("播放").size());
        else if (starts_with(target, "放一下")) target = target.substr(std::string("放一下").size());
        else if (starts_with(target, "放一首")) target = target.substr(std::string("放一首").size());

        const bool artist_request = is_artist_music_request(text);
        target = strip_music_artist_suffix(target);
        if (target.empty()) return false;

        action.target = target;
        if (artist_request) {
            // 歌手点播：command=artist_play，下游拉取该歌手热门歌曲
            action.params = "command=artist_play;provider=netease_app";
            plan.reply = "开始播放" + target + "的歌。";
        } else {
            // 歌曲搜索：command=play，下游按歌名搜索播放
            action.params = "command=play;provider=netease_app";
            plan.reply = "开始播放" + target + "。";
        }
    } else {
        // 无法匹配任何音乐意图，返回 false
        return false;
    }

    plan.actions.push_back(action);
    plan.confidence = action.confidence;
    return true;
}

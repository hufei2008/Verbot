#include "music_intent.h"

#include <algorithm>
#include <vector>

namespace {

std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    };
    while (!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
    return s;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
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

std::string strip_music_artist_suffix(std::string target) {
    target = strip_terminal_punctuation(target);
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

bool is_artist_music_request(const std::string& text) {
    std::string clean = strip_terminal_punctuation(text);
    return clean.find("的歌") != std::string::npos ||
           clean.find("的歌曲") != std::string::npos ||
           clean.find("的音乐") != std::string::npos ||
           ends_with(clean, "歌曲") ||
           ends_with(clean, "音乐");
}

} // namespace

bool build_fast_music_plan(const std::string& asr_text, TaskPlan& plan) {
    std::string text = strip_terminal_punctuation(asr_text);
    if (text.empty()) return false;

    Action action;
    action.type = ActionType::PLAY_MUSIC;
    action.action_name = "play_music";
    action.confidence = 0.95f;

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
        std::string target = text;
        if (starts_with(target, "播放")) target = target.substr(std::string("播放").size());
        else if (starts_with(target, "放一下")) target = target.substr(std::string("放一下").size());
        else if (starts_with(target, "放一首")) target = target.substr(std::string("放一首").size());

        const bool artist_request = is_artist_music_request(text);
        target = strip_music_artist_suffix(target);
        if (target.empty()) return false;

        action.target = target;
        if (artist_request) {
            action.params = "command=artist_play;provider=netease_app";
            plan.reply = "开始播放" + target + "的歌。";
        } else {
            action.params = "command=play;provider=netease_app";
            plan.reply = "开始播放" + target + "。";
        }
    } else {
        return false;
    }

    plan.actions.push_back(action);
    plan.confidence = action.confidence;
    return true;
}

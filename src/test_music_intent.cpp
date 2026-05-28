#include "music_intent.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Case {
    std::string input;
    std::string target;
    std::string params_contains;
    std::string reply;
};

void require_true(bool ok, const std::string& message) {
    if (!ok) {
        std::fprintf(stderr, "[FAIL] %s\n", message.c_str());
        std::exit(1);
    }
}

} // namespace

int main() {
    const std::vector<Case> cases = {
        {"播放周杰伦的歌", "周杰伦", "command=artist_play", "开始播放周杰伦的歌。"},
        {"播放周杰伦的歌。", "周杰伦", "command=artist_play", "开始播放周杰伦的歌。"},
        {"播放周杰伦歌曲", "周杰伦", "command=artist_play", "开始播放周杰伦的歌。"},
        {"播放周杰伦歌曲。", "周杰伦", "command=artist_play", "开始播放周杰伦的歌。"},
        {"播放稻香。", "稻香", "command=play", "开始播放稻香。"},
        {"下一首。", "", "command=next", "已切到下一首。"},
    };

    for (const auto& c : cases) {
        TaskPlan plan;
        require_true(build_fast_music_plan(c.input, plan), c.input + " should match music intent");
        require_true(plan.actions.size() == 1, c.input + " should produce one action");
        const Action& action = plan.actions.front();
        require_true(action.type == ActionType::PLAY_MUSIC, c.input + " should be PLAY_MUSIC");
        require_true(action.target == c.target,
                     c.input + " target expected \"" + c.target + "\", got \"" + action.target + "\"");
        require_true(action.params.find(c.params_contains) != std::string::npos,
                     c.input + " params should contain " + c.params_contains + ", got " + action.params);
        require_true(plan.reply == c.reply,
                     c.input + " reply expected \"" + c.reply + "\", got \"" + plan.reply + "\"");
        std::printf("[PASS] %-24s -> target=%s params=%s\n",
                    c.input.c_str(), action.target.c_str(), action.params.c_str());
    }

    return 0;
}

// ============================================================
// test_music_intent — music_intent 模块的单元测试
// 验证 build_fast_music_plan 对各种音乐控制输入的正确性
// 编译：cmake --build build --target test_music_intent
// 运行：./build/test_music_intent
// ============================================================

#include "music_intent.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// 匿名 namespace，测试用例结构体和辅助断言函数
namespace {

// 单个测试用例：input 语义文本，期望的 target/params/reply
struct Case {
    std::string input;            // 模拟 ASR 识别文本
    std::string target;          // 期望的 Action.target
    std::string params_contains; // Action.params 应包含的子串
    std::string reply;           // 期望的 plan.reply
};

// 断言辅助：ok 为 false 时打印失败信息并退出
void require_true(bool ok, const std::string& message) {
    if (!ok) {
        std::fprintf(stderr, "[FAIL] %s\n", message.c_str());
        std::exit(1);
    }
}

} // namespace

int main() {
    // 定义测试用例集：覆盖歌手点播、歌曲搜索、播放控制等场景
    const std::vector<Case> cases = {
        // 歌手点播：周杰伦
        {"播放周杰伦的歌", "周杰伦", "command=artist_play", "开始播放周杰伦的歌。"},
        {"播放周杰伦的歌。", "周杰伦", "command=artist_play", "开始播放周杰伦的歌。"},
        {"播放周杰伦的哥哥", "周杰伦", "command=artist_play", "开始播放周杰伦的歌。"},
        {"播放周杰伦的哥。", "周杰伦", "command=artist_play", "开始播放周杰伦的歌。"},
        {"播放周杰伦歌曲", "周杰伦", "command=artist_play", "开始播放周杰伦的歌。"},
        {"播放周杰伦歌曲。", "周杰伦", "command=artist_play", "开始播放周杰伦的歌。"},
        // 歌曲搜索：稻香
        {"播放稻香。", "稻香", "command=play", "开始播放稻香。"},
        // 播放控制：下一首
        {"下一首。", "", "command=next", "已切到下一首。"},
    };

    for (const auto& c : cases) {
        TaskPlan plan;
        // 1. 必须能匹配音乐意图
        require_true(build_fast_music_plan(c.input, plan), c.input + " should match music intent");
        // 2. 必须只产生一个 Action
        require_true(plan.actions.size() == 1, c.input + " should produce one action");
        const Action& action = plan.actions.front();
        // 3. Action 类型必须是 PLAY_MUSIC
        require_true(action.type == ActionType::PLAY_MUSIC, c.input + " should be PLAY_MUSIC");
        // 4. target 必须与期望一致
        require_true(action.target == c.target,
                     c.input + " target expected \"" + c.target + "\", got \"" + action.target + "\"");
        // 5. params 必须包含预期子串（如 command=artist_play）
        require_true(action.params.find(c.params_contains) != std::string::npos,
                     c.input + " params should contain " + c.params_contains + ", got " + action.params);
        // 6. 回复文案必须与期望一致
        require_true(plan.reply == c.reply,
                     c.input + " reply expected \"" + c.reply + "\", got \"" + plan.reply + "\"");
        std::printf("[PASS] %-24s -> target=%s params=%s\n",
                    c.input.c_str(), action.target.c_str(), action.params.c_str());
    }

    return 0;
}

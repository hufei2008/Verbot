#include "asr_rejector.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Case {
    std::string text;
    bool rejected;
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
        {"北京天气怎么样，", false},
        {"上海天气，", false},
        {"深圳的呢?", false},
        {"我要听周杰伦的歌。", false},
        {"打开计算器", false},
        {"现在几点了", false},

        {",", true},
        {"好，。", true},
        {"咳咳咳", true},
        {"请问有什么可以帮助你的。", true},
        {"打开关闭保存删除复制粘贴撤回发送他说他们说", true},
        {"MING PAO CANADA .COM.BORN TO BE", true},
        {"啊啊啊啊", true},
        {"呵呵呵呵，", true},
        {"哈哈哈哈", true},
        {"嘿嘿嘿", true},
        {"嘻嘻嘻", true},
        {"啦啦啦", true},
        {"不一定得得得", true},
    };

    for (const auto& c : cases) {
        auto decision = reject_asr_text(c.text);
        require_true(decision.rejected == c.rejected,
                     c.text + " expected rejected=" + (c.rejected ? "true" : "false") +
                     ", got " + (decision.rejected ? "true" : "false") +
                     " reason=" + decision.reason);
        std::printf("[PASS] %-48s -> %s\n",
                    c.text.c_str(), decision.rejected ? "reject" : "pass");
    }

    {
        AsrRejectFeatures features;
        features.text = "随机无关内容";
        features.n_tokens = 6;
        features.avg_token_p = 0.25f;
        features.min_token_p = 0.03f;
        features.no_speech_prob = 0.2f;
        auto decision = reject_asr_result(features);
        require_true(decision.rejected, "low confidence unsupported text should reject");
        std::printf("[PASS] %-48s -> reject\n", "low confidence unsupported text");
    }

    {
        AsrRejectFeatures features;
        features.text = "北京天气怎么样";
        features.n_tokens = 6;
        features.avg_token_p = 0.30f;
        features.min_token_p = 0.04f;
        features.no_speech_prob = 0.2f;
        auto decision = reject_asr_result(features);
        require_true(!decision.rejected, "supported signal should survive low token confidence");
        std::printf("[PASS] %-48s -> pass\n", "supported signal with low confidence");
    }

    {
        AsrRejectFeatures features;
        features.text = "旁边有点声音";
        features.n_tokens = 5;
        features.avg_token_p = 0.8f;
        features.min_token_p = 0.4f;
        features.no_speech_prob = 0.8f;
        auto decision = reject_asr_result(features);
        require_true(decision.rejected, "high no-speech unsupported text should reject");
        std::printf("[PASS] %-48s -> reject\n", "high no-speech unsupported text");
    }

    return 0;
}

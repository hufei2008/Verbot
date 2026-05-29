#pragma once

#include <string>

struct AsrRejectDecision {
    bool rejected = false;
    std::string reason;
};

struct AsrRejectFeatures {
    std::string text;
    int n_tokens = 0;
    float avg_token_p = 1.0f;
    float min_token_p = 1.0f;
    float no_speech_prob = 0.0f;
};

AsrRejectDecision reject_asr_text(const std::string& text);
AsrRejectDecision reject_asr_result(const AsrRejectFeatures& features);

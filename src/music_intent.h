#ifndef MUSIC_INTENT_H
#define MUSIC_INTENT_H

#include "semantic_engine.h"

#include <string>

bool build_fast_music_plan(const std::string& asr_text, TaskPlan& plan);

#endif

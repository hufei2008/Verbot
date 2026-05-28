// ============================================================
// music_intent.h — 音乐点播语义快速匹配模块
// 绕过 LLM，直接在 ASR 文本层面识别音乐控制意图
// （播放/暂停/下一首/上一首/歌手点播/歌曲搜索等），
// 生成对应的 TaskPlan 供 downstream 执行。
// ============================================================

#ifndef MUSIC_INTENT_H
#define MUSIC_INTENT_H

#include "semantic_engine.h"     // TaskPlan / Action / ActionType

#include <string>

// 尝试从 ASR 识别文本中提取音乐意图，填充 TaskPlan
// 返回 true 表示成功匹配（即使文本不属于音乐意图则返回 false）
// 匹配范围包括：播放控制、歌手点播、歌曲搜索、打开网易云等
bool build_fast_music_plan(const std::string& asr_text, TaskPlan& plan);

#endif

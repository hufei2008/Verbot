#!/usr/bin/env zsh
# ============================================================
# test_music_e2e.sh — 语音音乐点播端到端测试（Tahoe/CI）
# 测试链路：ASR 文本 → music_intent → NetEase 搜索 → 播放
# 用法：zsh scripts/test_music_e2e.sh
# ============================================================
set -euo pipefail

# 切换到项目根目录
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Step 1: 编译项目（并发 4 线程）
cmake --build build -j4 >/tmp/verbot_music_build.log

# Step 2: 运行 music_intent 单元测试
./build/test_music_intent

# Step 3: 模拟 ASR 文本 → Verbot 音乐播放
LOG="${TMPDIR:-/tmp}/verbot_music_e2e.$$.log"
./build/Verbot --text "播放周杰伦的歌。" >"$LOG" 2>&1

# Step 4: 验证关键日志输出
grep -q 'Fast music path: target="周杰伦" params="command=artist_play;provider=netease_app"' "$LOG"
grep -q 'NetEase song id:' "$LOG"
grep -q 'NetEase playable URL resolved' "$LOG"
grep -q 'MusicPlayer] Queue loaded:' "$LOG"
grep -q 'MusicPlayer] Playing 1/' "$LOG"

# Step 5: 提取播放的 song_id，验证本地缓存文件存在
SONG_ID="$(grep 'MusicPlayer] Playing 1/' "$LOG" | sed -E 's/.*id=([0-9]+).*/\1/' | tail -1)"
[[ -n "$SONG_ID" ]]
MUSIC_FILE="/tmp/verbot_netease_${SONG_ID}.mp3"

# Step 6: 等待播放进程启动（最多 6 秒）
for _ in {1..30}; do
    if pgrep -f 'ffplay .*music.126.net|mpg123 .*music.126.net' >/dev/null ||
       pgrep -f "afplay $MUSIC_FILE" >/dev/null; then
        break
    fi
    sleep 0.2
done

# Step 7: 验证播放器进程仍在运行
if ! pgrep -f 'ffplay .*music.126.net|mpg123 .*music.126.net' >/dev/null; then
    pgrep -f "afplay $MUSIC_FILE" >/dev/null
fi

# 输出最终结果
echo "[PASS] Verbot music E2E: 播放周杰伦的歌。"
echo "[INFO] song_id: $SONG_ID"
echo "[INFO] file: $MUSIC_FILE"
echo "[INFO] log: $LOG"
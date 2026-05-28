#!/usr/bin/env zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cmake --build build -j4 >/tmp/verbot_music_build.log

./build/test_music_intent

LOG="${TMPDIR:-/tmp}/verbot_music_e2e.$$.log"
./build/Verbot --text "播放周杰伦的歌。" >"$LOG" 2>&1

grep -q 'Fast music path: target="周杰伦" params="command=artist_play;provider=netease_app"' "$LOG"
grep -q 'NetEase song id:' "$LOG"
grep -q 'NetEase playable URL resolved' "$LOG"
grep -q 'MusicPlayer] Queue loaded:' "$LOG"
grep -q 'MusicPlayer] Playing 1/' "$LOG"

SONG_ID="$(grep 'MusicPlayer] Playing 1/' "$LOG" | sed -E 's/.*id=([0-9]+).*/\1/' | tail -1)"
[[ -n "$SONG_ID" ]]
MUSIC_FILE="/tmp/verbot_netease_${SONG_ID}.mp3"
for _ in {1..30}; do
    if pgrep -f 'ffplay .*music.126.net|mpg123 .*music.126.net' >/dev/null ||
       pgrep -f "afplay $MUSIC_FILE" >/dev/null; then
        break
    fi
    sleep 0.2
done

if ! pgrep -f 'ffplay .*music.126.net|mpg123 .*music.126.net' >/dev/null; then
    pgrep -f "afplay $MUSIC_FILE" >/dev/null
fi

echo "[PASS] Verbot music E2E: 播放周杰伦的歌。"
echo "[INFO] song_id: $SONG_ID"
echo "[INFO] file: $MUSIC_FILE"
echo "[INFO] log: $LOG"

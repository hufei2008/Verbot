// ============================================================
// Verbot - 实时语音识别 + 语义理解系统入口
// 功能：录音 → VAD语音活动检测 → Whisper语音识别 → LLM语义理解 → TTS语音合成
// ============================================================

#include "audio_recorder.h"     // 音频采集模块
#include "semantic_engine.h"    // 语义引擎（LLM + TTS）
#include "asr_rejector.h"       // ASR 文本拒识

#include "whisper.h"            // Whisper 语音识别库

#include <iostream>
#include <cstdio>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>
#include <atomic>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <future>
#include <csignal>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <set>
#include <unordered_map>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <curl/curl.h>

// ANSI 终端颜色宏定义，用于彩色日志输出
#define COLOR_CYAN    "\033[36m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_RED     "\033[31m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_RESET   "\033[0m"
#define CLEAR_SCREEN  "\033[2J\033[H"

// 音频处理常量：每帧512采样点（16kHz下约32ms），采样率16kHz
constexpr int SAMPLES_PER_VAD_FRAME = 512;   // 32ms @16kHz
constexpr int SAMPLE_RATE = 16000;
constexpr bool VAD_DEBUG_LOG = false;        // VAD 调试日志开关

// 全局退出标志：1 表示收到 SIGINT 信号，请求停止
static volatile std::sig_atomic_t g_stop_requested = 0;
// 优雅退出开关：仅在初始化完成后才启用，防止启动阶段误触发强制退出
static volatile std::sig_atomic_t g_graceful_sigint_enabled = 0;

    // SIGINT 信号处理函数（Ctrl+C）：首次按下优雅退出，再次按下强制退出
    static void handle_sigint(int) {
        if (!g_graceful_sigint_enabled || g_stop_requested) {
            _exit(130);  // 未启用优雅退出或已请求过，直接强制退出
        }
        g_stop_requested = 1;  // 设置退出标志
    }

    // 安全退出辅助函数：跳过 Python/MLX 的 atexit/finalizer 链。
    // 嵌入式 Python 加载 MLX、tokenizers 后，正常 return/exit 可能在
    // 进程收尾时触发 C 扩展析构并 crash。这里先 flush 日志，再直接
    // _exit，保留正常退出码。
    static void safe_exit(int code) {
        fflush(nullptr);          // 刷新所有输出缓冲区
        _exit(code);              // 直接系统调用退出，跳过 atexit
        __builtin_unreachable();  // 标记不可达，辅助编译器优化
    }

// ============================================================
// 线程安全的环形缓冲区（RingBuffer）
// 用于录音线程与主线程之间的音频数据传输
// push() 由录音回调线程调用，consume() 由主线程调用
// ============================================================
class RingBuffer {
public:
    // 向缓冲区尾部追加音频数据
    void push(const float* data, size_t n) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffer.insert(m_buffer.end(), data, data + n);
    }

    // 从缓冲区头部取出 n 个采样点数据
    size_t consume(std::vector<float>& out, size_t n) {
        std::lock_guard<std::mutex> lock(m_mutex);
        n = std::min(n, m_buffer.size());
        if (n == 0) return 0;
        out.assign(m_buffer.begin(), m_buffer.begin() + n);
        m_buffer.erase(m_buffer.begin(), m_buffer.begin() + n);
        return n;
    }

    // 获取当前缓冲区中的数据量（采样点数）
    size_t size() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_buffer.size();
    }

private:
    std::vector<float> m_buffer;  // 音频数据存储
    std::mutex m_mutex;           // 互斥锁，保证线程安全
};

// 抑制 whisper/llama 内部 debug 日志输出的回调函数
static void cb_log_disable(enum ggml_log_level, const char*, void*) {}

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static std::string applescript_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out;
}

static bool text_contains(const std::string& s, const std::string& needle) {
    return s.find(needle) != std::string::npos;
}

static size_t curl_write_string(void* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

static std::string url_escape(const std::string& text) {
    CURL* curl = curl_easy_init();
    if (!curl) return text;
    char* escaped = curl_easy_escape(curl, text.c_str(), (int)text.size());
    std::string out = escaped ? escaped : text;
    if (escaped) curl_free(escaped);
    curl_easy_cleanup(curl);
    return out;
}

static std::string http_get(const std::string& url, long timeout_sec = 8) {
    CURL* curl = curl_easy_init();
    if (!curl) return "";

    std::string body;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "User-Agent: Mozilla/5.0");
    headers = curl_slist_append(headers, "Referer: https://music.163.com/");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_COOKIEFILE, "");
    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || http_code < 200 || http_code >= 300) {
        fprintf(stderr, "[NetEaseMusic] HTTP failed: code=%ld err=%s url=%s\n",
                http_code, curl_easy_strerror(res), url.c_str());
        return "";
    }
    return body;
}

static std::string http_redirect_location(const std::string& url, long timeout_ms = 1500) {
    CURL* curl = curl_easy_init();
    if (!curl) return "";

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "User-Agent: Mozilla/5.0");
    headers = curl_slist_append(headers, "Referer: https://music.163.com/");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 800L);
    CURLcode res = curl_easy_perform(curl);

    char* redirect = nullptr;
    curl_easy_getinfo(curl, CURLINFO_REDIRECT_URL, &redirect);
    std::string out = (res == CURLE_OK && redirect) ? redirect : "";
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return out;
}

static std::vector<std::string> json_array_objects(const std::string& body,
                                                   const std::string& key) {
    std::vector<std::string> objects;
    size_t songs_pos = body.find("\"" + key + "\"");
    if (songs_pos == std::string::npos) return objects;

    size_t array_start = body.find('[', songs_pos);
    if (array_start == std::string::npos) return objects;

    int array_depth = 0;
    int object_depth = 0;
    bool in_string = false;
    bool escape = false;
    size_t object_start = std::string::npos;

    for (size_t i = array_start; i < body.size(); ++i) {
        char c = body[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '[') {
            array_depth++;
            continue;
        }
        if (c == ']') {
            array_depth--;
            if (array_depth == 0) break;
            continue;
        }
        if (array_depth != 1) continue;

        if (c == '{') {
            if (object_depth == 0) object_start = i;
            object_depth++;
        } else if (c == '}') {
            object_depth--;
            if (object_depth == 0 && object_start != std::string::npos) {
                objects.push_back(body.substr(object_start, i - object_start + 1));
                object_start = std::string::npos;
            }
        }
    }

    return objects;
}

static std::string top_level_json_number(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (size_t i = 0; i < json.size(); ++i) {
        char c = json[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            if (depth == 1 && json.compare(i, needle.size(), needle) == 0) {
                size_t colon = json.find(':', i + needle.size());
                if (colon == std::string::npos) return "";
                size_t begin = json.find_first_of("0123456789", colon + 1);
                if (begin == std::string::npos) return "";
                size_t end = json.find_first_not_of("0123456789", begin);
                return json.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
            }
            in_string = true;
        } else if (c == '{') {
            depth++;
        } else if (c == '}') {
            depth--;
        }
    }
    return "";
}

static std::string top_level_json_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (size_t i = 0; i < json.size(); ++i) {
        char c = json[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            if (depth == 1 && json.compare(i, needle.size(), needle) == 0) {
                size_t colon = json.find(':', i + needle.size());
                if (colon == std::string::npos) return "";
                size_t quote = json.find('"', colon + 1);
                if (quote == std::string::npos) return "";

                std::string out;
                bool value_escape = false;
                for (size_t j = quote + 1; j < json.size(); ++j) {
                    char vc = json[j];
                    if (value_escape) {
                        switch (vc) {
                            case 'n': out.push_back('\n'); break;
                            case 'r': out.push_back('\r'); break;
                            case 't': out.push_back('\t'); break;
                            case '"': out.push_back('"'); break;
                            case '\\': out.push_back('\\'); break;
                            case '/': out.push_back('/'); break;
                            default: out.push_back(vc); break;
                        }
                        value_escape = false;
                    } else if (vc == '\\') {
                        value_escape = true;
                    } else if (vc == '"') {
                        return out;
                    } else {
                        out.push_back(vc);
                    }
                }
                return out;
            }
            in_string = true;
        } else if (c == '{') {
            depth++;
        } else if (c == '}') {
            depth--;
        }
    }
    return "";
}

static bool netease_song_has_artist(const std::string& song_object, const std::string& artist) {
    if (artist.empty()) return false;
    // cloudsearch API 用 "ar"，旧 search API 用 "artists"
    return (song_object.find("\"ar\"") != std::string::npos ||
            song_object.find("\"artists\"") != std::string::npos) &&
           song_object.find("\"name\":\"" + artist + "\"") != std::string::npos;
}

static std::string json_unescape_slashes(std::string s) {
    size_t pos = 0;
    while ((pos = s.find("\\/", pos)) != std::string::npos) {
        s.replace(pos, 2, "/");
        pos += 1;
    }
    return s;
}

static std::string netease_player_url(const std::string& song_id);

static std::string first_playable_song_id(const std::vector<std::string>& songs,
                                          const std::string& preferred_artist = "") {
    for (const auto& song : songs) {
        if (!preferred_artist.empty() && !netease_song_has_artist(song, preferred_artist)) {
            continue;
        }
        std::string id = top_level_json_number(song, "id");
        if (!id.empty() && !netease_player_url(id).empty()) return id;
    }

    if (!preferred_artist.empty()) {
        for (const auto& song : songs) {
            std::string id = top_level_json_number(song, "id");
            if (!id.empty() && !netease_player_url(id).empty()) return id;
        }
    }

    return "";
}

static std::string first_netease_artist_id(const std::string& artist) {
    std::string url = "https://music.163.com/api/cloudsearch/pc?s=" + url_escape(artist) +
        "&type=100&offset=0&limit=5";
    std::string body = http_get(url);
    auto artists = json_array_objects(body, "artists");
    if (artists.empty()) return "";

    for (const auto& candidate : artists) {
        if (candidate.find("\"name\":\"" + artist + "\"") != std::string::npos) {
            std::string id = top_level_json_number(candidate, "id");
            if (!id.empty()) return id;
        }
    }

    return top_level_json_number(artists.front(), "id");
}

static std::string first_netease_artist_hot_song_id(const std::string& artist) {
    std::string artist_id = first_netease_artist_id(artist);
    if (artist_id.empty()) return "";

    std::string url = "https://music.163.com/api/artist/" + artist_id;
    std::string body = http_get(url);
    auto songs = json_array_objects(body, "hotSongs");
    if (songs.empty()) return "";

    std::vector<std::string> non_live;
    for (const auto& song : songs) {
        if (song.find("(Live)") == std::string::npos && song.find("Live") == std::string::npos) {
            non_live.push_back(song);
        }
    }

    std::string id = first_playable_song_id(non_live, artist);
    if (!id.empty()) return id;
    return first_playable_song_id(songs, artist);
}

static std::string first_netease_song_id(const std::string& query,
                                         const std::string& preferred_artist = "") {
    std::string url = "https://music.163.com/api/cloudsearch/pc?s=" + url_escape(query) +
        "&type=1&offset=0&limit=20";
    std::string body = http_get(url);
    fprintf(stdout, "[NetEaseMusic] Search response len=%zu\n", body.size());
    if (!body.empty()) {
        fprintf(stdout, "[NetEaseMusic] Response preview: %.400s\n", body.c_str());
    }
    auto songs = json_array_objects(body, "songs");
    fprintf(stdout, "[NetEaseMusic] Found %zu songs for \"%s\"\n", songs.size(), query.c_str());
    for (size_t i = 0; i < songs.size() && i < 5; ++i) {
        std::string id = top_level_json_number(songs[i], "id");
        fprintf(stdout, "[NetEaseMusic]   #%zu: id=%s raw=%.200s\n", i+1, id.c_str(), songs[i].c_str());
    }
    if (songs.empty()) return "";

    // 优先匹配指定歌手的歌
    if (!preferred_artist.empty()) {
        for (const auto& song : songs) {
            if (netease_song_has_artist(song, preferred_artist)) {
                std::string id = top_level_json_number(song, "id");
                if (!id.empty()) return id;
            }
        }
    }

    // 返回第一个结果
    return top_level_json_number(songs.front(), "id");
}

struct MusicTrack {
    std::string id;
    std::string title;
    std::string artist;
    std::string audio_url;
};

static std::mutex g_music_mutex;
static std::vector<MusicTrack> g_music_queue;
static size_t g_music_index = 0;

static std::mutex g_netease_cache_mutex;
static std::unordered_map<std::string, std::string> g_netease_url_cache;
static std::unordered_map<std::string, std::vector<MusicTrack>> g_netease_track_cache;

static std::string netease_player_url(const std::string& song_id) {
    if (song_id.empty()) return "";

    {
        std::lock_guard<std::mutex> lock(g_netease_cache_mutex);
        auto it = g_netease_url_cache.find(song_id);
        if (it != g_netease_url_cache.end()) return it->second;
    }

    std::string outer_url = "https://music.163.com/song/media/outer/url?id=" + song_id + ".mp3";
    std::string redirect = http_redirect_location(outer_url, 1500);
    std::string playable = (!redirect.empty() && redirect.find("/404") == std::string::npos)
        ? redirect
        : "";

    {
        std::lock_guard<std::mutex> lock(g_netease_cache_mutex);
        g_netease_url_cache[song_id] = playable;
    }
    return playable;
}

static std::vector<MusicTrack> playable_tracks_from_songs(const std::vector<std::string>& songs,
                                                          const std::string& preferred_artist,
                                                          size_t limit = 10,
                                                          int max_playable_probes = 12) {
    std::vector<MusicTrack> tracks;
    std::vector<std::pair<std::string, std::string>> candidates;
    std::set<std::string> seen_ids;

    auto add_candidate = [&](const std::string& song) {
        if ((int)candidates.size() >= max_playable_probes) return;
        std::string id = top_level_json_number(song, "id");
        if (id.empty() || seen_ids.count(id)) return;
        candidates.push_back({song, id});
        seen_ids.insert(id);
    };

    if (!preferred_artist.empty()) {
        for (const auto& song : songs) {
            if (netease_song_has_artist(song, preferred_artist)) {
                add_candidate(song);
            }
        }
    }

    for (const auto& song : songs) {
        add_candidate(song);
    }

    std::vector<std::future<std::string>> futures;
    futures.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        futures.push_back(std::async(std::launch::async, [id = candidate.second]() {
            return netease_player_url(id);
        }));
    }

    for (size_t i = 0; i < candidates.size() && tracks.size() < limit; ++i) {
        std::string audio_url = futures[i].get();
        if (audio_url.empty()) continue;

        MusicTrack track;
        track.id = candidates[i].second;
        track.title = top_level_json_string(candidates[i].first, "name");
        track.artist = preferred_artist;
        track.audio_url = audio_url;
        tracks.push_back(track);
    }

    return tracks;
}

static std::vector<MusicTrack> search_netease_tracks(const std::string& query,
                                                     const std::string& preferred_artist = "",
                                                     int max_playable_probes = 12) {
    std::string cache_key = query + "\n" + preferred_artist + "\n" + std::to_string(max_playable_probes);
    {
        std::lock_guard<std::mutex> lock(g_netease_cache_mutex);
        auto it = g_netease_track_cache.find(cache_key);
        if (it != g_netease_track_cache.end()) {
            fprintf(stdout, "[NetEaseMusic] Search cache hit: \"%s\" -> %zu tracks\n",
                    query.c_str(), it->second.size());
            return it->second;
        }
    }

    auto started = std::chrono::steady_clock::now();
    std::string url = "https://music.163.com/api/cloudsearch/pc?s=" + url_escape(query) +
        "&type=1&offset=0&limit=30";
    std::string body = http_get(url);
    auto songs = json_array_objects(body, "songs");
    auto tracks = playable_tracks_from_songs(songs, preferred_artist, 10, max_playable_probes);
    double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    fprintf(stdout, "[NetEaseMusic] Search \"%s\" returned %zu songs, %zu playable tracks in %.0fms\n",
            query.c_str(), songs.size(), tracks.size(), elapsed_ms);

    {
        std::lock_guard<std::mutex> lock(g_netease_cache_mutex);
        g_netease_track_cache[cache_key] = tracks;
    }
    return tracks;
}

static std::vector<MusicTrack> artist_hot_tracks(const std::string& artist) {
    // 网易云公开外链对很多大版权歌手官方曲库会直接返回 404。
    // 歌手播放先找“歌手 + 翻唱/歌曲”的可播放音源，避免只打开 App 当前队列。
    auto tracks = search_netease_tracks(artist + " 翻唱", "", 8);
    if (!tracks.empty()) return tracks;

    tracks = search_netease_tracks(artist + " 歌曲", "", 12);
    if (!tracks.empty()) return tracks;

    std::string artist_id = first_netease_artist_id(artist);
    if (artist_id.empty()) return {};

    std::string url = "https://music.163.com/api/artist/" + artist_id;
    std::string body = http_get(url);
    auto songs = json_array_objects(body, "hotSongs");
    if (songs.empty()) return {};

    std::vector<std::string> non_live;
    for (const auto& song : songs) {
        if (song.find("(Live)") == std::string::npos && song.find("Live") == std::string::npos) {
            non_live.push_back(song);
        }
    }

    tracks = playable_tracks_from_songs(non_live, artist, 10, 16);
    if (tracks.empty()) {
        tracks = playable_tracks_from_songs(songs, artist, 10, 16);
    }
    if (tracks.empty()) {
        tracks = search_netease_tracks(artist, artist, 20);
    }
    return tracks;
}

static std::string music_file_path(const std::string& song_id) {
    return "/tmp/verbot_netease_" + song_id + ".mp3";
}

static void stop_music_player_process() {
    std::system("pkill -TERM -f 'ffplay .*music.126.net' >/dev/null 2>&1 || true");
    std::system("pkill -TERM -f 'mpg123 .*music.126.net' >/dev/null 2>&1 || true");
    std::system("pkill -TERM -f 'afplay /tmp/verbot_netease_' >/dev/null 2>&1 || true");
}

static void pause_music_player_process() {
    std::system("pkill -STOP -f 'ffplay .*music.126.net' >/dev/null 2>&1 || true");
    std::system("pkill -STOP -f 'mpg123 .*music.126.net' >/dev/null 2>&1 || true");
    std::system("pkill -STOP -f 'afplay /tmp/verbot_netease_' >/dev/null 2>&1 || true");
    fprintf(stdout, "[MusicPlayer] Paused\n");
}

static void resume_music_player_process() {
    std::system("pkill -CONT -f 'ffplay .*music.126.net' >/dev/null 2>&1 || true");
    std::system("pkill -CONT -f 'mpg123 .*music.126.net' >/dev/null 2>&1 || true");
    std::system("pkill -CONT -f 'afplay /tmp/verbot_netease_' >/dev/null 2>&1 || true");
    fprintf(stdout, "[MusicPlayer] Resumed\n");
}

static void play_music_track(const MusicTrack& track, size_t index, size_t total) {
    if (track.id.empty() || track.audio_url.empty()) return;
    std::string out_path = music_file_path(track.id);

    stop_music_player_process();
    std::string headers = "User-Agent: Mozilla/5.0\r\nReferer: https://music.163.com/\r\n";
    std::string cache_cmd =
        "curl -L --max-time 60 -A 'Mozilla/5.0' -e 'https://music.163.com/' " +
        shell_quote(track.audio_url) + " -o " + shell_quote(out_path) +
        " >/tmp/verbot_music_cache.log 2>&1 &";
    std::string player_cmd =
        "if command -v ffplay >/dev/null 2>&1; then "
        "exec ffplay -nodisp -autoexit -loglevel error -headers " + shell_quote(headers) + " " + shell_quote(track.audio_url) + "; "
        "elif command -v mpg123 >/dev/null 2>&1; then "
        "exec mpg123 -q " + shell_quote(track.audio_url) + "; "
        "else "
        "curl -L --max-time 60 -A 'Mozilla/5.0' -e 'https://music.163.com/' " + shell_quote(track.audio_url) +
        " -o " + shell_quote(out_path) + " && exec afplay " + shell_quote(out_path) + "; "
        "fi";
    std::string cmd =
        "nohup sh -c " + shell_quote(cache_cmd + " " + player_cmd) +
        " >/tmp/verbot_music_playback.log 2>&1 &";
    std::system(cmd.c_str());
    fprintf(stdout, "[MusicPlayer] Playing %zu/%zu: id=%s title=\"%s\" artist=\"%s\" file=%s\n",
            index + 1, total, track.id.c_str(), track.title.c_str(), track.artist.c_str(), out_path.c_str());
}

static void set_music_queue_and_play(std::vector<MusicTrack> tracks) {
    std::lock_guard<std::mutex> lock(g_music_mutex);
    g_music_queue = std::move(tracks);
    g_music_index = 0;
    fprintf(stdout, "[MusicPlayer] Queue loaded: %zu tracks\n", g_music_queue.size());
    if (!g_music_queue.empty()) {
        play_music_track(g_music_queue[g_music_index], g_music_index, g_music_queue.size());
    }
}

static void play_next_music_track() {
    std::lock_guard<std::mutex> lock(g_music_mutex);
    if (g_music_queue.empty()) {
        fprintf(stderr, "[MusicPlayer] No queue for next track\n");
        return;
    }
    g_music_index = (g_music_index + 1) % g_music_queue.size();
    play_music_track(g_music_queue[g_music_index], g_music_index, g_music_queue.size());
}

static void play_previous_music_track() {
    std::lock_guard<std::mutex> lock(g_music_mutex);
    if (g_music_queue.empty()) {
        fprintf(stderr, "[MusicPlayer] No queue for previous track\n");
        return;
    }
    g_music_index = (g_music_index + g_music_queue.size() - 1) % g_music_queue.size();
    play_music_track(g_music_queue[g_music_index], g_music_index, g_music_queue.size());
}

static std::string music_command_from_action(const Action& action) {
    std::string p = action.params;
    std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });

    if (text_contains(p, "pause") || text_contains(p, "暂停")) return "pause";
    if (text_contains(p, "resume") || text_contains(p, "continue") || text_contains(p, "继续")) return "resume";
    if (text_contains(p, "next") || text_contains(p, "下一")) return "next";
    if (text_contains(p, "previous") || text_contains(p, "prev") || text_contains(p, "上一")) return "previous";
    if (text_contains(p, "stop") || text_contains(p, "停止") || text_contains(p, "关闭")) return "stop";
    if (text_contains(p, "open") || text_contains(p, "打开")) return "open";
    if (text_contains(p, "artist") || text_contains(p, "歌手")) return "artist_play";
    return action.target.empty() ? "play" : "search_play";
}

static int run_applescript(const std::vector<std::string>& lines) {
    std::string cmd = "osascript";
    for (const auto& line : lines) {
        cmd += " -e " + shell_quote(line);
    }
    return std::system(cmd.c_str());
}

static void activate_netease_music() {
    std::system("open -a NeteaseMusic");
}

static void click_netease_control_menu(const std::string& item) {
    run_applescript({
        "tell application \"NeteaseMusic\" to activate",
        "delay 0.2",
        "tell application \"System Events\" to tell process \"NeteaseMusic\" to click menu item \"" +
            applescript_escape(item) + "\" of menu 1 of menu bar item \"控制\" of menu bar 1"
    });
}

static void click_netease_control_menu_any(const std::vector<std::string>& items) {
    std::string script =
        "tell application \"NeteaseMusic\" to activate\n"
        "delay 0.2\n"
        "tell application \"System Events\" to tell process \"NeteaseMusic\"\n";
    for (const auto& item : items) {
        script +=
            "try\n"
            "click menu item \"" + applescript_escape(item) + "\" of menu 1 of menu bar item \"控制\" of menu bar 1\n"
            "return\n"
            "end try\n";
    }
    script += "end tell";
    run_applescript({script});
}

static std::string netease_play_pause_menu_state() {
    std::string cmd =
        "osascript -e " +
        shell_quote("tell application \"System Events\" to tell process \"NeteaseMusic\" to get name of menu item 1 of menu 1 of menu bar item \"控制\" of menu bar 1");

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buf[256];
    std::string out;
    while (fgets(buf, sizeof(buf), pipe)) {
        out += buf;
    }
    pclose(pipe);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
        out.pop_back();
    }
    return out;
}

static void ensure_netease_playing() {
    std::string first_state = netease_play_pause_menu_state();
    fprintf(stdout, "[NetEaseMusic] control menu before ensure_playing: %s\n",
            first_state.empty() ? "(unknown)" : first_state.c_str());

    for (int attempt = 1; attempt <= 5; ++attempt) {
        std::string state = netease_play_pause_menu_state();
        if (state == "暂停") {
            fprintf(stdout, "[NetEaseMusic] control menu after ensure_playing: %s\n", state.c_str());
            return;
        }

        // Deep links sometimes need a moment before the player accepts Play.
        std::this_thread::sleep_for(std::chrono::milliseconds(300 * attempt));
        click_netease_control_menu_any({"播放"});
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::string final_state = netease_play_pause_menu_state();
    fprintf(stdout, "[NetEaseMusic] control menu after ensure_playing: %s\n",
            final_state.empty() ? "(unknown)" : final_state.c_str());
    if (final_state != "暂停") {
        fprintf(stderr, "[NetEaseMusic] Failed to enter playing state after retries\n");
    }
}

static bool netease_play_song_by_id(const std::string& song_id) {
    if (song_id.empty()) return false;
    std::string url = "orpheus://song/" + song_id;
    fprintf(stdout, "[NetEaseMusic] Telling NeteaseMusic to open location: %s\n", url.c_str());
    int ret = run_applescript({
        "tell application \"NeteaseMusic\"",
        "activate",
        "delay 0.5",
        "open location \"" + applescript_escape(url) + "\"",
        "end tell"
    });
    fprintf(stdout, "[NetEaseMusic] AppleScript result: %d\n", ret);

    if (ret != 0) {
        fprintf(stdout, "[NetEaseMusic] AppleScript failed, trying open command\n");
        std::system("open -a NeteaseMusic");
        std::string cmd = "open " + shell_quote(url);
        ret = std::system(cmd.c_str());
        fprintf(stdout, "[NetEaseMusic] open command result: %d\n", ret);
    }

    // 最后的回退：浏览器打开
    if (ret != 0) {
        std::string web_url = "https://music.163.com/#/song?id=" + song_id;
        fprintf(stdout, "[NetEaseMusic] orpheus failed, opening web: %s\n", web_url.c_str());
        std::string cmd = "open " + shell_quote(web_url);
        ret = std::system(cmd.c_str());
        fprintf(stdout, "[NetEaseMusic] web open result: %d\n", ret);
    }
    return ret == 0;
}

static void execute_netease_music_action(const Action& action) {
    std::string command = music_command_from_action(action);
    std::string target = action.target;
    if (target == "网易云音乐" || target == "NeteaseMusic" || target == "NetEase Cloud Music") {
        target.clear();
    }

    printf("%s  ↪ NetEase Music command: %s target=\"%s\"%s\n",
           COLOR_YELLOW, command.c_str(), target.c_str(), COLOR_RESET);

    if (command == "open") {
        activate_netease_music();
        return;
    }

    if ((command == "play" || command == "search_play") && target.empty()) {
        fprintf(stdout, "[MusicPlayer] Empty play command ignored; ask user for a song or artist\n");
        return;
    }

    if ((command == "search_play" || command == "artist_play") && !target.empty()) {
        std::vector<MusicTrack> tracks = command == "artist_play"
            ? artist_hot_tracks(target)
            : search_netease_tracks(target);
        if (tracks.empty() && command == "search_play") {
            fprintf(stdout, "[NetEaseMusic] Song search empty, fallback to artist_play: %s\n",
                    target.c_str());
            tracks = artist_hot_tracks(target);
        }
        if (!tracks.empty()) {
            const std::string& song_id = tracks.front().id;
            printf("%s  ↪ NetEase song id: %s%s\n",
                   COLOR_YELLOW, song_id.c_str(), COLOR_RESET);
            printf("%s  ↪ NetEase playable URL resolved%s\n", COLOR_YELLOW, COLOR_RESET);
            set_music_queue_and_play(std::move(tracks));
        } else {
            fprintf(stderr, "[NetEaseMusic] No playable song found for query: %s\n", target.c_str());
        }
        return;
    }

    if (command == "next") {
        play_next_music_track();
        return;
    }

    if (command == "previous") {
        play_previous_music_track();
        return;
    }

    if (command == "pause" || command == "stop") {
        if (command == "stop") {
            stop_music_player_process();
            fprintf(stdout, "[MusicPlayer] Stopped\n");
        } else {
            pause_music_player_process();
        }
        return;
    }

    resume_music_player_process();
}

// 获取当前时间的 HH:MM:SS 格式字符串，用于日志时间戳
static std::string current_time_str() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

static std::string debug_timestamp_for_filename() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
    return std::string(buf);
}

static std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    };
    while (!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
    return s;
}

static std::string sanitize_debug_filename(std::string text) {
    text = trim_copy(text);
    if (text.empty()) text = "empty";

    std::string out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        if (c < 0x20 || c == 0x7f || c == '/' || c == '\\' || c == ':' ||
            c == '*' || c == '?' || c == '"' || c == '<' || c == '>' ||
            c == '|' || c == '\n' || c == '\r' || c == '\t') {
            out += '_';
        } else {
            out += (char)c;
        }
    }

    while (!out.empty() && (out.back() == ' ' || out.back() == '.')) {
        out.pop_back();
    }
    if (out.empty()) out = "empty";
    if (out.size() > 96) out = out.substr(0, 96);
    return out;
}

static void write_le16(std::ofstream& out, uint16_t v) {
    char bytes[2] = {
        static_cast<char>(v & 0xff),
        static_cast<char>((v >> 8) & 0xff),
    };
    out.write(bytes, sizeof(bytes));
}

static void write_le32(std::ofstream& out, uint32_t v) {
    char bytes[4] = {
        static_cast<char>(v & 0xff),
        static_cast<char>((v >> 8) & 0xff),
        static_cast<char>((v >> 16) & 0xff),
        static_cast<char>((v >> 24) & 0xff),
    };
    out.write(bytes, sizeof(bytes));
}

static bool write_pcm16_wav(const std::filesystem::path& path,
                            const std::vector<float>& samples,
                            int sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * block_align);
    const uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    write_le32(out, riff_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_le32(out, 16);
    write_le16(out, 1);
    write_le16(out, channels);
    write_le32(out, static_cast<uint32_t>(sample_rate));
    write_le32(out, byte_rate);
    write_le16(out, block_align);
    write_le16(out, bits_per_sample);
    out.write("data", 4);
    write_le32(out, data_size);

    for (float sample : samples) {
        sample = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t pcm = static_cast<int16_t>(std::lrint(sample * 32767.0f));
        write_le16(out, static_cast<uint16_t>(pcm));
    }

    return out.good();
}

static void save_asr_debug_audio_if_needed(bool enabled,
                                           const std::filesystem::path& dir,
                                           int speech_count,
                                           const std::string& asr_text,
                                           const std::vector<float>& samples) {
    if (!enabled || samples.empty()) return;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        fprintf(stderr, "[ASR Debug] Failed to create dir %s: %s\n",
                dir.string().c_str(), ec.message().c_str());
        return;
    }

    char prefix[32];
    std::snprintf(prefix, sizeof(prefix), "%04d_", speech_count);
    std::filesystem::path path = dir / (std::string(prefix) +
        sanitize_debug_filename(asr_text) + "_" + debug_timestamp_for_filename() + ".wav");

    if (write_pcm16_wav(path, samples, SAMPLE_RATE)) {
        fprintf(stdout, "%s[ASR Debug] Saved ASR audio: %s%s\n",
                COLOR_CYAN, path.string().c_str(), COLOR_RESET);
    } else {
        fprintf(stderr, "[ASR Debug] Failed to save ASR audio: %s\n", path.string().c_str());
    }
}

static void save_asr_debug_audio_pair_if_needed(bool enabled,
                                                const std::filesystem::path& root_dir,
                                                int speech_count,
                                                const std::string& asr_text,
                                                const std::vector<float>& raw_samples,
                                                const std::vector<float>& asr_samples) {
    if (!enabled) return;
    save_asr_debug_audio_if_needed(true, root_dir / "raw_capture",
                                   speech_count, asr_text, raw_samples);
    save_asr_debug_audio_if_needed(true, root_dir / "asr_input",
                                   speech_count, asr_text, asr_samples);
}


// 计算音频数据的 RMS（均方根）能量，衡量音量大小
static float compute_rms(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum / n);
}

// 计算音频数据的峰值幅度（最大绝对值）
static float compute_peak(const float* data, size_t n) {
    float peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float absv = std::fabs(data[i]);
        if (absv > peak) peak = absv;
    }
    return peak;
}

// 判断是否应跳过低质量语音片段
// 规则：时长过短（<0.45s）或短且能量低（<0.85s, rms<0.016, peak<0.055）
static bool should_skip_low_quality_segment(double durSec, float, float) {
    // 只丢弃极短的片段（<0.3s），VAD 已经过滤了噪音，
    // 不再用能量门槛，避免丢弃短指令（如"打开"、"好的"）
    if (durSec < 0.30) return true;
    return false;
}

// 一阶高通 IIR 滤波器，滤除 cutoff Hz 以下的低频噪声（如空调、风扇、风声）
// 参数：data - 输入输出音频数据；cutoff - 截止频率（Hz）；sampleRate - 采样率
static void highpass_filter(std::vector<float>& data, float cutoff, int sampleRate) {
    const float dt = 1.0f / sampleRate;                    // 采样间隔
    const float rc = 1.0f / (2.0f * M_PI * cutoff);        // RC 时间常数
    const float alpha = rc / (rc + dt);                     // 滤波系数

    if (data.empty()) return;

    float y_prev = data[0];  // 前一输出值
    for (size_t i = 1; i < data.size(); ++i) {
        float y = alpha * (y_prev + data[i] - data[i-1]);  // 一阶高通差分方程
        data[i] = y;
        y_prev = y;
    }
}

// 执行语义引擎输出的动作（打开应用、搜索网页、获取时间/天气等）
static void execute_action(const Action& action) {
    switch (action.type) {
        case ActionType::OPEN_APP: {
            std::string cmd = "open -a \"" + action.target + "\"";
            printf("%s  ↪ Opening app: %s%s\n",
                   COLOR_YELLOW, cmd.c_str(), COLOR_RESET);
            std::system(cmd.c_str());
            break;
        }
        case ActionType::SEARCH_WEB: {
            std::string url = "https://www.google.com/search?q=" + action.target;
            for (auto& c : url) { if (c == ' ') c = '+'; }
            std::string cmd = "open \"" + url + "\"";
            printf("%s  ↪ Searching: %s%s\n",
                   COLOR_YELLOW, url.c_str(), COLOR_RESET);
            std::system(cmd.c_str());
            break;
        }
        case ActionType::GET_TIME: {
            printf("%s  ↪ Current time: %s%s\n",
                   COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);
            break;
        }
        case ActionType::GET_WEATHER: {
            printf("%s  ↪ Weather API result: %s%s\n",
                   COLOR_YELLOW,
                   (!action.params.empty() ? action.params : action.response_text).c_str(),
                   COLOR_RESET);
            break;
        }
        case ActionType::OPEN_DOMAIN_QA: {
            printf("%s  ↪ Open-domain QA result: %s%s\n",
                   COLOR_YELLOW,
                   (!action.params.empty() ? action.params : action.response_text).c_str(),
                   COLOR_RESET);
            break;
        }
        case ActionType::PLAY_MUSIC: {
            execute_netease_music_action(action);
            break;
        }
        default: {
            if (action.type != ActionType::NONE) {
                printf("%s  ↪ (No safe handler for %s)%s\n",
                       COLOR_YELLOW, action.action_name.c_str(), COLOR_RESET);
            }
            break;
        }
    }
}

int main(int argc, char ** argv) {
    std::signal(SIGINT, handle_sigint);

    // 抑制 whisper VAD 内部 debug 日志
    whisper_log_set(cb_log_disable, nullptr);
    llama_log_set(cb_log_disable, nullptr);

    std::string whisperModel  = "models/ggml-large-v3-turbo.bin";
    std::string vadModelPath  = "models/ggml-silero-v6.2.0.bin";
    std::string llmModelPath  = "models/gemma4-26b-a4b-it-q4_k_m.gguf";
    std::string textInput;
    bool debugMode = false;
    std::filesystem::path asrDebugDir = "asr_debug_audio";

    bool textMode = false;
    bool argError = false;
    int positionalModelArg = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--text") {
            textMode = true;
            if (i + 1 >= argc) {
                argError = true;
                break;
            }
            textInput = argv[++i];
        } else if (arg == "--llm") {
            if (i + 1 >= argc) {
                argError = true;
                break;
            }
            llmModelPath = argv[++i];
        } else if (arg == "debug" || arg == "--debug") {
            debugMode = true;
        } else if (arg == "--asr-debug-dir") {
            if (i + 1 >= argc) {
                argError = true;
                break;
            }
            asrDebugDir = argv[++i];
        } else if (!textMode) {
            if (positionalModelArg == 0) whisperModel = arg;
            else if (positionalModelArg == 1) vadModelPath = arg;
            else if (positionalModelArg == 2) llmModelPath = arg;
            positionalModelArg++;
        }
    }

    if (argError || (textMode && textInput.empty())) {
        fprintf(stderr,
                "Usage: %s [debug] [--asr-debug-dir dir] [whisper_model] [vad_model] [llm_model]\n"
                "       %s --text \"打开计算器\" [--llm path/to/model.gguf]\n",
                argv[0], argv[0]);
        return 1;
    }

    if (debugMode) {
        std::error_code ec;
        std::filesystem::create_directories(asrDebugDir, ec);
        std::filesystem::create_directories(asrDebugDir / "raw_capture", ec);
        std::filesystem::create_directories(asrDebugDir / "asr_input", ec);
        if (ec) {
            fprintf(stderr, "Failed to create ASR debug dir %s: %s\n",
                    asrDebugDir.string().c_str(), ec.message().c_str());
            return 1;
        }
        printf("%s[%s] Debug mode enabled. ASR audio dir: %s%s\n",
               COLOR_CYAN, current_time_str().c_str(),
               std::filesystem::absolute(asrDebugDir).string().c_str(), COLOR_RESET);
        printf("%s[%s] Debug audio stages: raw_capture/ before filtering, asr_input/ sent to Whisper%s\n",
               COLOR_CYAN, current_time_str().c_str(), COLOR_RESET);
    }

    // ============================================================
    // 1. 初始化语义引擎（LLM + TTS）
    // ============================================================
    printf("%s[%s] Loading LLM model: %s ...%s\n",
           COLOR_CYAN, current_time_str().c_str(), llmModelPath.c_str(), COLOR_RESET);

    SemanticEngine semanticEngine;
    if (!semanticEngine.init(llmModelPath, 4096, 4, true)) {
        fprintf(stderr, "Failed to initialize semantic engine!\n");
        return 1;
    }
    printf("%s[%s] Semantic engine ready.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    // TTS 状态
    if (semanticEngine.tts_ready()) {
        const char* ttsBackend = std::getenv("TTS_BACKEND");
        printf("%s[%s] TTS ready (%s backend via Python bridge)%s\n",
               COLOR_GREEN, current_time_str().c_str(),
               ttsBackend ? ttsBackend : "macos", COLOR_RESET);
    } else {
        printf("%s[%s] TTS not available. "
               "To enable, set:\n"
               "  export TTS_BACKEND=macos\n"
               "  export MACOS_TTS_VOICE=Tingting\n"
               "  export QWEN_TTS_PYTHON_HOME=/path/to/conda/envs/cosyvoice\n"
               "  export QWEN_TTS_BRIDGE_DIR=/path/to/study2/python%s\n",
               COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);
    }

    if (textMode) {
        std::promise<void> done;
        auto future = done.get_future();
        bool doneSet = false;
        g_graceful_sigint_enabled = 1;

        printf("%s🧠 [%s] Text mode input: \"%s\"%s\n",
               COLOR_MAGENTA, current_time_str().c_str(), textInput.c_str(), COLOR_RESET);

        semanticEngine.process_asr_result(textInput, [&](const Action& action) {
            if (!action.response_text.empty()) {
                printf("\n%s🤖 %s%s\n",
                       COLOR_BOLD COLOR_CYAN, action.response_text.c_str(), COLOR_RESET);
                // TTS 由 SemanticEngine 内部自动处理
                if (!doneSet) {
                    doneSet = true;
                    done.set_value();
                }
            }

            if (action.type != ActionType::NONE) {
                printf("%s⚡ [Action] %s: %s%s\n",
                       COLOR_BOLD COLOR_GREEN,
                       action.action_name.c_str(),
                       action.target.c_str(),
                       COLOR_RESET);
                execute_action(action);
            }
        });

        while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
            if (g_stop_requested) {
                fprintf(stderr, "\nInterrupted.\n");
                safe_exit(130);
            }
        }

        if (g_stop_requested) {
            fprintf(stderr, "\nInterrupted.\n");
            safe_exit(130);
        }

        // 等待 TTS 播放完成
        printf("%s[%s] Waiting for TTS playback to finish...%s\n",
               COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);
        while (!g_stop_requested && !semanticEngine.wait_for_tts_for(100)) {}
        if (g_stop_requested) {
            fprintf(stderr, "\nInterrupted.\n");
            safe_exit(130);
        }

        safe_exit(0);
    }

    // ============================================================
    // 2. 初始化 Whisper
    // ============================================================
    printf("%s[%s] Loading Whisper model: %s ...%s\n",
           COLOR_CYAN, current_time_str().c_str(), whisperModel.c_str(), COLOR_RESET);

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    cparams.gpu_device = 0;

    struct whisper_context * ctx = whisper_init_from_file_with_params(whisperModel.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to load Whisper model!" << std::endl;
        safe_exit(1);
    }
    printf("%s[%s] Whisper model loaded.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    // ============================================================
    // 3. 初始化 VAD
    // ============================================================
    printf("%s[%s] Loading VAD model: %s ...%s\n",
           COLOR_CYAN, current_time_str().c_str(), vadModelPath.c_str(), COLOR_RESET);

    struct whisper_vad_context_params vparams = whisper_vad_default_context_params();
    vparams.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    vparams.use_gpu   = false;

    struct whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(
            vadModelPath.c_str(), vparams);
    if (!vctx) {
        std::cerr << "Failed to load VAD model!" << std::endl;
        whisper_free(ctx);
        safe_exit(1);
    }
    printf("%s[%s] VAD model loaded.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    // VAD 参数
    struct whisper_vad_params vad_params = whisper_vad_default_params();
    vad_params.threshold               = 1.0f;   // 最高概率阈值，最大限度减少噪音误触发
    vad_params.min_speech_duration_ms  = 300;    // 增加最小语音持续时长
    vad_params.min_silence_duration_ms = 700;    // 增加沉默判定时长
    vad_params.speech_pad_ms           = 200;

    printf("%s[%s] VAD params: thr=%.1f, min_speech=%dms, min_silence=%dms, pad=%dms%s\n",
           COLOR_CYAN, current_time_str().c_str(),
           vad_params.threshold,
           vad_params.min_speech_duration_ms,
           vad_params.min_silence_duration_ms,
           vad_params.speech_pad_ms,
           COLOR_RESET);

    // ============================================================
    // 4. 启动录音
    // ============================================================
    RingBuffer ringBuffer;
    std::atomic<bool> running{true};

    AudioRecorder recorder;
    recorder.setDataCallback([&](const float* data, size_t n) {
        ringBuffer.push(data, n);
    });

    if (!recorder.start(true)) {
        std::cerr << "Failed to start recording!" << std::endl;
        whisper_vad_free(vctx);
        whisper_free(ctx);
        safe_exit(1);
    }

    // ============================================================
    // 程序启动完毕
    // ============================================================
    printf(CLEAR_SCREEN);
    printf("%s╔════════════════════════════════════════════════════╗%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("%s║    🎙  Real-time ASR + Semantic Understanding     ║%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("%s║    Press Ctrl+C to exit                           ║%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("%s╚════════════════════════════════════════════════════╝%s\n", COLOR_BOLD COLOR_MAGENTA, COLOR_RESET);
    printf("\n");
    printf("%s[%s] 🔴 Recording started (16kHz, 32-bit float)%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);
    printf("%s──────────────────────────────────────────────────%s\n", COLOR_CYAN, COLOR_RESET);
    g_graceful_sigint_enabled = 1;

    // ============================================================
    // 5. VAD + ASR 主循环
    // ============================================================

    enum State {
        STATE_IDLE,
        STATE_SPEECHING,
    };

    State state = STATE_IDLE;

    std::vector<float> speechBuffer;
    int speechCount = 0;
    int vadStartCount = 0;
    int speechFrames = 0;

    const int SILENCE_FRAMES_THRESHOLD = 10;
    int silenceFrames = 0;

    const int MIN_SPEECH_FRAMES = 12;            // 需要更多连续帧才触发
    const int MAX_SPEECH_FRAMES = 30 * 1000 / 32;

    const float SPEECH_RMS_THRESHOLD = 0.025f;   // 提高能量门槛，小环境音不触发
    float noiseFloor = 0.025f;

    const int PAD_FRAMES = 8;
    std::vector<float> padBuffer;

    // 先向 VAD 输入静音帧做校准
    printf("%s[%s] Calibrating VAD (300ms of silence)...%s\n",
           COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);

    std::vector<float> silenceInput(SAMPLES_PER_VAD_FRAME, 0.0f);
    for (int i = 0; i < 5; ++i) {
        whisper_vad_detect_speech_no_reset(vctx, silenceInput.data(), SAMPLES_PER_VAD_FRAME);
    }
    whisper_vad_reset_state(vctx);

    printf("%s[%s] VAD calibrated.%s\n",
           COLOR_GREEN, current_time_str().c_str(), COLOR_RESET);

    while (running && !g_stop_requested) {
        std::vector<float> chunk;
        size_t got = ringBuffer.consume(chunk, SAMPLES_PER_VAD_FRAME);

        if (got == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        float rms = compute_rms(chunk.data(), (int)chunk.size());
        float peak = compute_peak(chunk.data(), (int)chunk.size());

        if (state == STATE_IDLE && rms < noiseFloor) {
            noiseFloor = 0.999f * noiseFloor + 0.001f * rms;
        }

        bool vadSpeech = whisper_vad_detect_speech_no_reset(vctx,
                chunk.data(), (int)chunk.size());

        float adaptiveThreshold = std::max(noiseFloor * 5.0f, SPEECH_RMS_THRESHOLD);
        bool energyHasSpeech = (rms >= adaptiveThreshold);

        bool isSpeech = vadSpeech && energyHasSpeech;

        // VAD 调试日志（每 30 帧或说话时打印）
        static int logCounter = 0;
        if (VAD_DEBUG_LOG && (++logCounter % 30 == 0 || state == STATE_SPEECHING)) {
            printf("%s[DEBUG] frame: vad=%d rms=%.4f peak=%.4f thr=%.4f noise=%.4f energy=%d isSpeech=%d state=%s silence=%d%s\n",
                   COLOR_CYAN,
                   (int)vadSpeech, rms, peak, adaptiveThreshold, noiseFloor,
                   (int)energyHasSpeech, (int)isSpeech,
                   (state == STATE_IDLE) ? "IDLE" : "SPEECH",
                   silenceFrames,
                   COLOR_RESET);
        }

        switch (state) {
            case STATE_IDLE: {
                if (isSpeech) {
                    state = STATE_SPEECHING;
                    speechFrames = 0;
                    silenceFrames = 0;
                    vadStartCount++;

                    speechBuffer.clear();
                    speechBuffer.insert(speechBuffer.end(), padBuffer.begin(), padBuffer.end());

                    speechBuffer.insert(speechBuffer.end(), chunk.begin(), chunk.end());
                    speechFrames++;

                    printf("\n%s▶ [%s] Voice activity detected (rms=%.4f, peak=%.4f, noise=%.4f)%s\n",
                           COLOR_YELLOW, current_time_str().c_str(),
                           rms, peak, noiseFloor, COLOR_RESET);
                }
                break;
            }

            case STATE_SPEECHING: {
                speechBuffer.insert(speechBuffer.end(), chunk.begin(), chunk.end());
                speechFrames++;

                if (isSpeech) {
                    silenceFrames = 0;
                } else {
                    silenceFrames++;
                }

                if (silenceFrames >= SILENCE_FRAMES_THRESHOLD) {
                    if (speechFrames >= MIN_SPEECH_FRAMES) {
                        goto do_transcribe;
                    } else {
                        float durMs = speechFrames * 32.0f;
                        state = STATE_IDLE;
                        speechBuffer.clear();
                        speechFrames = 0;
                        silenceFrames = 0;
                        padBuffer.clear();
                        printf("%s↩ [%s] Too short (%.0fms), discarding%s\n",
                               COLOR_YELLOW, current_time_str().c_str(),
                               durMs, COLOR_RESET);
                    }
                } else if (speechFrames >= MAX_SPEECH_FRAMES) {
                    printf("%s⏰ [%s] Speech too long (%.0fs), force-transcribing...%s\n",
                           COLOR_YELLOW, current_time_str().c_str(),
                           speechFrames * 32.0f / 1000.0f, COLOR_RESET);
                    whisper_vad_reset_state(vctx);
                    goto do_transcribe;
                }
                break;
            }
        }

        // 更新语音前置 padding。必须放在状态机之后：
        // 1. IDLE->SPEECHING 时 padBuffer 只包含当前帧之前的音频，避免开头重复当前帧。
        // 2. 结束转写时不再把 padBuffer 追加到 speechBuffer，避免尾音重复。
        padBuffer.insert(padBuffer.end(), chunk.begin(), chunk.end());
        if (padBuffer.size() > (size_t)PAD_FRAMES * SAMPLES_PER_VAD_FRAME) {
            padBuffer.erase(padBuffer.begin(), padBuffer.begin() + SAMPLES_PER_VAD_FRAME);
        }

        continue;

    do_transcribe:
        {
            std::vector<float> rawSpeechBuffer;
            if (debugMode) {
                rawSpeechBuffer = speechBuffer;
            }

            // 高通滤波
            highpass_filter(speechBuffer, 80.0f, SAMPLE_RATE);

            double durSec = speechBuffer.size() / (double)SAMPLE_RATE;
            float segmentRms = compute_rms(speechBuffer.data(), speechBuffer.size());
            float segmentPeak = compute_peak(speechBuffer.data(), speechBuffer.size());

            if (should_skip_low_quality_segment(durSec, segmentRms, segmentPeak)) {
                printf("%s⚠ [%s] Ignoring low-quality speech segment (%.1fs, rms=%.4f, peak=%.4f)%s\n",
                       COLOR_YELLOW, current_time_str().c_str(), durSec,
                       segmentRms, segmentPeak, COLOR_RESET);
                speechBuffer.clear();
                speechFrames = 0;
                silenceFrames = 0;
                state = STATE_IDLE;
                printf("%s──────────────────────────────────────────────────%s\n",
                       COLOR_CYAN, COLOR_RESET);
                continue;
            }

            printf("%s📏 [%s] Speech segment (%.1fs, %zu samples)%s\n",
                   COLOR_CYAN, current_time_str().c_str(), durSec,
                   speechBuffer.size(), COLOR_RESET);
            printf("%s⏳ [%s] Transcribing #%d...%s\n",
                   COLOR_YELLOW, current_time_str().c_str(), ++speechCount, COLOR_RESET);

            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
            wparams.print_progress   = false;
            wparams.print_special    = false;
            wparams.print_realtime   = false;
            wparams.print_timestamps = false;
            wparams.language         = "zh";
            wparams.n_threads        = std::min(4, (int)std::thread::hardware_concurrency());

            wparams.no_context       = true;             // 每段语音独立识别，避免上一轮文本污染短指令
            wparams.single_segment   = true;             // VAD 已经切成单条指令，减少短句被拆段和补字
            wparams.suppress_blank   = true;
            wparams.suppress_nst     = true;

            wparams.temperature      = 0.0f;
            wparams.temperature_inc  = 0.4f;             // 允许低置信短句 fallback，减少重复解码
            wparams.entropy_thold    = 1.2f;
            wparams.logprob_thold    = -1.0f;
            wparams.no_speech_thold  = 0.5f;
            wparams.max_len          = 40;

            wparams.beam_search.beam_size = 7;           // large-v3-turbo 下优先保证短句准确率

            const char * chinese_prompt_v2 =
                "语音助手常用指令："
                "播放周杰伦的歌，播放周杰伦歌曲，播放稻香，播放晴天，播放网易云音乐。"
                "暂停音乐，继续播放，下一首，上一首，打开网易云音乐。"
                "北京天气怎么样，上海天气怎么样，打开计算器，现在几点了。"
                "你好，请问有什么可以帮助你的。";

            wparams.initial_prompt        = chinese_prompt_v2;
            wparams.carry_initial_prompt  = false;
            wparams.tdrz_enable           = false;

            whisper_vad_reset_state(vctx);

            auto t0 = std::chrono::high_resolution_clock::now();

            int ret = whisper_full(ctx, wparams,
                    speechBuffer.data(), (int)speechBuffer.size());

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::string asr_text;

            if (ret == 0) {
                const int nSeg = whisper_full_n_segments(ctx);
                printf("%s✅ [%s] Transcription done in %.0fms (%d segments)%s\n",
                       COLOR_GREEN, current_time_str().c_str(),
                       elapsedMs, nSeg, COLOR_RESET);

                printf("%s  ┌─────────────────────────────────────────────┐%s\n",
                       COLOR_BOLD COLOR_GREEN, COLOR_RESET);
                printf("%s  │ 📝 RESULT: ", COLOR_BOLD COLOR_GREEN);
                AsrRejectFeatures asr_features;
                asr_features.no_speech_prob = nSeg > 0
                    ? whisper_full_get_segment_no_speech_prob(ctx, 0)
                    : 1.0f;
                float tokenProbSum = 0.0f;
                float minTokenProb = 1.0f;
                int tokenCount = 0;
                for (int i = 0; i < nSeg; ++i) {
                    const char* text = whisper_full_get_segment_text(ctx, i);
                    if (text && text[0] != '\0') {
                        printf("%s", text);
                        asr_text += text;
                    }
                    const int nTokens = whisper_full_n_tokens(ctx, i);
                    for (int j = 0; j < nTokens; ++j) {
                        const float p = whisper_full_get_token_p(ctx, i, j);
                        tokenProbSum += p;
                        minTokenProb = std::min(minTokenProb, p);
                        tokenCount++;
                    }
                }
                asr_features.n_tokens = tokenCount;
                asr_features.avg_token_p = tokenCount > 0 ? tokenProbSum / (float)tokenCount : 0.0f;
                asr_features.min_token_p = tokenCount > 0 ? minTokenProb : 0.0f;
                if (debugMode) {
                    fprintf(stdout,
                            "%s[ASR Debug] no_speech=%.3f avg_token_p=%.3f min_token_p=%.3f tokens=%d%s\n",
                            COLOR_CYAN,
                            asr_features.no_speech_prob,
                            asr_features.avg_token_p,
                            asr_features.min_token_p,
                            asr_features.n_tokens,
                            COLOR_RESET);
                }
                printf("%s\n", COLOR_RESET);
                printf("%s  └─────────────────────────────────────────────┘%s\n",
                       COLOR_BOLD COLOR_GREEN, COLOR_RESET);
                printf("\n");

                while (!asr_text.empty() && (asr_text.back() == ' ' || asr_text.back() == '\n'))
                    asr_text.pop_back();
                while (!asr_text.empty() && (asr_text.front() == ' ' || asr_text.front() == '\n'))
                    asr_text.erase(0, 1);

                save_asr_debug_audio_pair_if_needed(debugMode, asrDebugDir,
                                                    speechCount, asr_text,
                                                    rawSpeechBuffer, speechBuffer);

                // ============================================================
                // ★ 语义理解：将 ASR 结果送入 LLM 处理
                // ============================================================
                if (!asr_text.empty()) {
                    asr_features.text = asr_text;
                    auto reject_decision = reject_asr_result(asr_features);
                    if (reject_decision.rejected) {
                        printf("%s⚠ [%s] 无效命令拒识%s\n",
                               COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);
                        goto after_semantic;
                    }

                    static std::string lastSubmittedText;
                    static auto lastSubmittedAt = std::chrono::steady_clock::now() - std::chrono::seconds(60);
                    auto now = std::chrono::steady_clock::now();
                    // 5 秒内连续相同文本才跳过（而非 20 秒），允许用户重复指令
                    if (asr_text == lastSubmittedText &&
                        now - lastSubmittedAt < std::chrono::seconds(5)) {
                        printf("%s⚠ [%s] Ignoring duplicate ASR: \"%s\"%s\n",
                               COLOR_YELLOW, current_time_str().c_str(),
                               asr_text.c_str(), COLOR_RESET);
                        goto after_semantic;
                    }
                    lastSubmittedText = asr_text;
                    lastSubmittedAt = now;

                    printf("%s🧠 [%s] Sending to semantic engine: \"%s\"%s\n",
                           COLOR_MAGENTA, current_time_str().c_str(),
                           asr_text.c_str(), COLOR_RESET);

                    semanticEngine.process_asr_result(asr_text, [](const Action& action) {
                        // Action 处理完成后的回调
                        if (action.type != ActionType::NONE) {
                            printf("%s⚡ [Action] %s: %s%s\n",
                                   COLOR_BOLD COLOR_GREEN,
                                   action.action_name.c_str(),
                                   action.target.c_str(),
                                   COLOR_RESET);

                            execute_action(action);
                        }

                        // 显示 LLM 回复
                        if (!action.response_text.empty()) {
                            printf("\n%s🤖 %s%s%s\n",
                                   COLOR_BOLD COLOR_CYAN, action.response_text.c_str(),
                                   COLOR_RESET, COLOR_RESET);

                            // ★ TTS 合成由 SemanticEngine 内部自动完成
                        }
                    });
                }
            after_semantic:
                ;
            } else {
                printf("%s❌ [%s] Transcription failed (ret=%d)%s\n",
                       COLOR_RED, current_time_str().c_str(), ret, COLOR_RESET);
                save_asr_debug_audio_pair_if_needed(debugMode, asrDebugDir,
                                                    speechCount,
                                                    "transcription_failed_ret_" + std::to_string(ret),
                                                    rawSpeechBuffer, speechBuffer);
            }

            // 重置 VAD 状态和所有变量
            speechBuffer.clear();
            speechFrames = 0;
            silenceFrames = 0;
            padBuffer.clear();
            state = STATE_IDLE;

            printf("%s──────────────────────────────────────────────────%s\n",
                   COLOR_CYAN, COLOR_RESET);
        }
    }

    // ============================================================
    // 6. 清理
    // ============================================================
    printf("\n%s[%s] Shutting down...%s\n", COLOR_YELLOW, current_time_str().c_str(), COLOR_RESET);
    printf("%s[%s] Total detections: %d, transcriptions: %d%s\n",
           COLOR_CYAN, current_time_str().c_str(), vadStartCount, speechCount, COLOR_RESET);
    recorder.stop();
    whisper_vad_free(vctx);
    whisper_free(ctx);

    // 使用 safe_exit 而非 exit()/return，避免 Python finalization crash。
    // 详见 safe_exit() 注释。
    if (g_stop_requested) {
        safe_exit(130);
    }

    safe_exit(0);
}

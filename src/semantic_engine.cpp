#include "semantic_engine.h"
#include "china_cities.h"
#include "music_intent.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <vector>
#include <thread>
#include <cstdlib>
#include <chrono>
#include <cctype>
#include <fstream>
#include <ctime>
#include <curl/curl.h>

namespace {

std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    };
    while (!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
    return s;
}

std::string find_json_object(const std::string& text) {
    size_t start = text.find('{');
    if (start == std::string::npos) return "";

    bool in_string = false;
    bool escaped = false;
    int depth = 0;

    for (size_t i = start; i < text.size(); ++i) {
        char c = text[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\' && in_string) {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;

        if (c == '{') {
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0) {
                return text.substr(start, i - start + 1);
            }
        }
    }

    return "";
}

std::vector<std::string> json_object_array_value(const std::string& json, const std::string& key) {
    std::vector<std::string> objects;
    std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) return objects;

    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return objects;

    size_t array_start = json.find('[', colon + 1);
    if (array_start == std::string::npos) return objects;

    bool in_string = false;
    bool escaped = false;
    int array_depth = 0;
    int object_depth = 0;
    size_t object_start = std::string::npos;

    for (size_t i = array_start; i < json.size(); ++i) {
        char c = json[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\' && in_string) {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;

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
                objects.push_back(json.substr(object_start, i - object_start + 1));
                object_start = std::string::npos;
            }
        }
    }

    return objects;
}

std::string json_string_value(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) return "";

    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return "";

    size_t quote = json.find('"', colon + 1);
    if (quote == std::string::npos) return "";

    std::string out;
    bool escaped = false;
    for (size_t i = quote + 1; i < json.size(); ++i) {
        char c = json[i];
        if (escaped) {
            switch (c) {
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                default: out.push_back(c); break;
            }
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            return out;
        }
        out.push_back(c);
    }

    return "";
}

float json_float_value(const std::string& json, const std::string& key, float fallback) {
    std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) return fallback;

    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return fallback;

    size_t start = json.find_first_of("-0123456789", colon + 1);
    if (start == std::string::npos) return fallback;

    size_t end = json.find_first_not_of("0123456789.-", start);
    try {
        return std::stof(json.substr(start, end - start));
    } catch (...) {
        return fallback;
    }
}

ActionType action_type_from_name(const std::string& action_name) {
    if (action_name == "open_app") return ActionType::OPEN_APP;
    if (action_name == "search_web") return ActionType::SEARCH_WEB;
    if (action_name == "open_domain_qa") return ActionType::OPEN_DOMAIN_QA;
    if (action_name == "send_message") return ActionType::SEND_MESSAGE;
    if (action_name == "get_time") return ActionType::GET_TIME;
    if (action_name == "set_reminder") return ActionType::SET_REMINDER;
    if (action_name == "play_music") return ActionType::PLAY_MUSIC;
    if (action_name == "get_weather") return ActionType::GET_WEATHER;
    if (action_name == "custom") return ActionType::CUSTOM;
    return ActionType::NONE;
}

Action action_from_json_object(const std::string& json, const std::string& default_reply = "") {
    Action action;
    action.response_text = trim_copy(json_string_value(json, "reply"));
    if (action.response_text.empty()) {
        action.response_text = default_reply;
    }
    action.action_name = trim_copy(json_string_value(json, "action"));
    action.target = trim_copy(json_string_value(json, "target"));
    action.params = trim_copy(json_string_value(json, "params"));
    action.confidence = json_float_value(json, "confidence", 0.0f);

    if (action.action_name.empty()) {
        action.action_name = "none";
    }

    action.type = action_type_from_name(action.action_name);
    if (action.action_name == "system_cmd") {
        action.type = ActionType::NONE;
        action.target.clear();
        action.params.clear();
        if (action.response_text.empty()) {
            action.response_text = "这个操作不在当前安全白名单里。";
        }
    }

    return action;
}

std::string spoken_text_for_action(const Action& action) {
    if (action.type == ActionType::SEARCH_WEB && !action.target.empty()) {
        return "查" + action.target + "。";
    }
    if (action.type == ActionType::OPEN_APP && !action.target.empty()) {
        return "打开" + action.target + "。";
    }
    if (action.type == ActionType::GET_TIME) {
        return "我看看时间。";
    }
    if (action.type == ActionType::GET_WEATHER && !action.target.empty()) {
        return "查" + action.target + "天气。";
    }
    if (action.type == ActionType::PLAY_MUSIC) {
        if (!action.target.empty() && action.target != "网易云音乐") return "播放" + action.target + "。";
        return "播放音乐。";
    }
    return action.response_text;
}

size_t curl_write_string(void* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    size_t total = size * nmemb;
    out->append(static_cast<const char*>(ptr), total);
    return total;
}

std::string http_get(const std::string& url, long timeout_sec = 8) {
    CURL* curl = curl_easy_init();
    if (!curl) return "";

    std::string body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Verbot/1.0");
    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || http_code < 200 || http_code >= 300) {
        fprintf(stderr, "[Weather] HTTP failed: code=%ld err=%s url=%s\n",
                http_code, curl_easy_strerror(res), url.c_str());
        return "";
    }
    return body;
}

std::string http_get_with_headers(const std::string& url,
                                  const std::vector<std::string>& headers,
                                  long timeout_sec = 8) {
    CURL* curl = curl_easy_init();
    if (!curl) return "";

    std::string body;
    struct curl_slist* header_list = nullptr;
    for (const auto& header : headers) {
        header_list = curl_slist_append(header_list, header.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Verbot/1.0");
    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || http_code < 200 || http_code >= 300) {
        fprintf(stderr, "[OpenDomainQA] HTTP GET failed: code=%ld err=%s url=%s body=%s\n",
                http_code, curl_easy_strerror(res), url.c_str(), body.c_str());
        return "";
    }
    return body;
}

std::string http_post_json(const std::string& url,
                           const std::string& body_json,
                           const std::vector<std::string>& headers,
                           long timeout_sec = 12) {
    CURL* curl = curl_easy_init();
    if (!curl) return "";

    std::string body;
    struct curl_slist* header_list = nullptr;
    for (const auto& header : headers) {
        header_list = curl_slist_append(header_list, header.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_json.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body_json.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Verbot/1.0");

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || http_code < 200 || http_code >= 300) {
        fprintf(stderr, "[OpenDomainQA] HTTP POST failed: code=%ld err=%s url=%s body=%s\n",
                http_code, curl_easy_strerror(res), url.c_str(), body.c_str());
        return "";
    }
    return body;
}

std::string url_escape(const std::string& text) {
    CURL* curl = curl_easy_init();
    if (!curl) return text;
    char* escaped = curl_easy_escape(curl, text.c_str(), (int)text.size());
    std::string out = escaped ? escaped : text;
    if (escaped) curl_free(escaped);
    curl_easy_cleanup(curl);
    return out;
}

double json_double_value(const std::string& json, const std::string& key, double fallback) {
    return (double)json_float_value(json, key, (float)fallback);
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 16);
    for (char c : text) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

std::string first_json_object_in_array(const std::string& json, const std::string& key) {
    std::vector<std::string> objects = json_object_array_value(json, key);
    return objects.empty() ? "" : objects.front();
}

std::string clean_weather_location(std::string target) {
    target = trim_copy(target);
    const std::string suffixes[] = {"天气怎么样", "的天气", "天气"};
    for (const auto& suffix : suffixes) {
        size_t pos = target.find(suffix);
        if (pos != std::string::npos) {
            target.erase(pos, suffix.size());
        }
    }
    target = trim_copy(target);
    return target.empty() ? "北京" : target;
}

std::string weather_code_text(int code) {
    if (code == 0) return "晴";
    if (code == 1 || code == 2 || code == 3) return "多云";
    if (code == 45 || code == 48) return "有雾";
    if ((code >= 51 && code <= 57) || (code >= 61 && code <= 67) || (code >= 80 && code <= 82)) return "有雨";
    if ((code >= 71 && code <= 77) || (code >= 85 && code <= 86)) return "有雪";
    if (code >= 95 && code <= 99) return "有雷雨";
    return "天气正常";
}

std::string current_time_summary() {
    std::time_t t = std::time(nullptr);
    std::tm local_tm{};
#if defined(_WIN32)
    localtime_s(&local_tm, &t);
#else
    localtime_r(&t, &local_tm);
#endif

    char buf[64];
    std::strftime(buf, sizeof(buf), "现在是%H点%M分。", &local_tm);
    return std::string(buf);
}

std::string completed_text_for_action(const Action& action) {
    if (action.type == ActionType::OPEN_APP && !action.target.empty()) {
        if (action.target == "Calculator") return "已打开计算器。";
        return "已打开" + action.target + "。";
    }
    if (action.type == ActionType::SEARCH_WEB && !action.target.empty()) {
        return "已搜索" + action.target + "。";
    }
    if (action.type == ActionType::GET_TIME) {
        return current_time_summary();
    }
    if (action.type == ActionType::PLAY_MUSIC) {
        std::string params = action.params;
        std::transform(params.begin(), params.end(), params.begin(), [](unsigned char c) {
            return (char)std::tolower(c);
        });
        if (params.find("pause") != std::string::npos || params.find("暂停") != std::string::npos) {
            return "已暂停音乐。";
        }
        if (params.find("resume") != std::string::npos || params.find("continue") != std::string::npos ||
            params.find("继续") != std::string::npos) {
            return "继续播放音乐。";
        }
        if (params.find("next") != std::string::npos || params.find("下一") != std::string::npos) {
            return "已切到下一首。";
        }
        if (params.find("previous") != std::string::npos || params.find("prev") != std::string::npos ||
            params.find("上一") != std::string::npos) {
            return "已切到上一首。";
        }
        if (params.find("open") != std::string::npos || params.find("打开") != std::string::npos) {
            return "已打开网易云音乐。";
        }
        if (!action.target.empty() && action.target != "网易云音乐") {
            return "开始播放" + action.target + "。";
        }
        return "开始播放音乐。";
    }
    return "";
}

std::string query_weather_summary(const std::string& target) {
    std::string location = clean_weather_location(target);

    int city_idx = find_city(location);
    if (city_idx < 0) {
        return "没找到" + location + "的天气。";
    }

    double latitude = CHINA_CITIES[city_idx].lat;
    double longitude = CHINA_CITIES[city_idx].lng;
    std::string resolved_name = CHINA_CITIES[city_idx].name;
    if (resolved_name.empty()) resolved_name = location;

    char forecast_url[1024];
    snprintf(forecast_url, sizeof(forecast_url),
             "https://api.open-meteo.com/v1/forecast?latitude=%.6f&longitude=%.6f"
             "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,precipitation"
             "&timezone=auto&forecast_days=1",
             latitude, longitude);

    std::string forecast_body = http_get(forecast_url);
    size_t current_pos = forecast_body.find("\"current\"");
    std::string current = current_pos == std::string::npos
        ? ""
        : find_json_object(forecast_body.substr(current_pos));
    if (current.empty()) {
        return "天气接口暂时没有返回结果。";
    }

    double temp = json_double_value(current, "temperature_2m", 999.0);
    double humidity = json_double_value(current, "relative_humidity_2m", -1.0);
    double wind = json_double_value(current, "wind_speed_10m", -1.0);
    int code = (int)json_double_value(current, "weather_code", -1.0);

    if (temp > 998.0) {
        return "天气接口暂时没有温度结果。";
    }

    char summary[512];
    snprintf(summary, sizeof(summary),
             "%s现在%.0f度，%s，湿度%.0f%%，风速%.0f公里每小时。",
             resolved_name.c_str(), temp, weather_code_text(code).c_str(),
             humidity < 0.0 ? 0.0 : humidity,
             wind < 0.0 ? 0.0 : wind);
    return summary;
}

std::string baidu_api_key() {
    // 优先从环境变量读取：BAIDU_AI_SEARCH_API_KEY > BAIDU_API_KEY > QIANFAN_API_KEY
    const char* key = std::getenv("BAIDU_AI_SEARCH_API_KEY");
    if (!key || !key[0]) key = std::getenv("BAIDU_API_KEY");
    if (!key || !key[0]) key = std::getenv("QIANFAN_API_KEY");
    std::string value = key ? trim_copy(key) : "";
    if (!value.empty()) {
        return value;
    }

    // 从本地配置文件读取（不得提交到 Git）
    auto read_key_file = [](const std::string& path) -> std::string {
        std::ifstream in(path);
        if (!in.is_open()) return "";

        std::string line;
        while (std::getline(in, line)) {
            line = trim_copy(line);
            if (line.empty() || line[0] == '#') continue;

            size_t eq = line.find('=');
            if (eq != std::string::npos) {
                std::string name = trim_copy(line.substr(0, eq));
                if (name != "BAIDU_AI_SEARCH_API_KEY" &&
                    name != "BAIDU_API_KEY" &&
                    name != "QIANFAN_API_KEY") {
                    continue;
                }
                line = trim_copy(line.substr(eq + 1));
            }

            if (line.size() >= 2 &&
                ((line.front() == '"' && line.back() == '"') ||
                 (line.front() == '\'' && line.back() == '\''))) {
                line = line.substr(1, line.size() - 2);
            }
            line = trim_copy(line);
            if (line == "your_baidu_api_key_here" || line == "<AppBuilder API Key>" ||
                line == "<API Key>") {
                continue;
            }
            return line;
        }
        return "";
    };

    const char* home = std::getenv("HOME");
    const std::vector<std::string> paths = {
        ".env",
        "../.env",
        "config/baidu_api_key.txt",
        "../config/baidu_api_key.txt",
        home ? std::string(home) + "/.verbot/baidu_api_key.txt" : ""
    };
    for (const auto& path : paths) {
        if (path.empty()) continue;
        value = read_key_file(path);
        if (!value.empty()) {
            fprintf(stdout, "[OpenDomainQA] Loaded Baidu API key from %s\n", path.c_str());
            return value;
        }
    }

    // 注意：不要在此处硬编码 API Key fallback！
    // 密钥应从环境变量 BAIDU_AI_SEARCH_API_KEY 或配置文件 config/baidu_api_key.txt 提供。
    // 硬编码密钥会导致 secret 泄漏（已被 Git 历史记录捕获的风险）。
    return "";
}

std::string extract_open_domain_source(const Action& action) {
    std::string params = action.params;
    std::transform(params.begin(), params.end(), params.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    if (params.find("baike") != std::string::npos || params.find("百科") != std::string::npos) {
        return "baidu_baike";
    }
    if (params.find("news") != std::string::npos || params.find("新闻") != std::string::npos) {
        return "baidu_news";
    }

    const std::string& q = action.target;
    if (q.find("新闻") != std::string::npos || q.find("最近") != std::string::npos ||
        q.find("最新") != std::string::npos || q.find("热点") != std::string::npos ||
        q.find("发生了什么") != std::string::npos) {
        return "baidu_news";
    }
    if (q.find("是什么") != std::string::npos || q.find("介绍") != std::string::npos ||
        q.find("百科") != std::string::npos || q.find("谁是") != std::string::npos ||
        q.find("是谁") != std::string::npos) {
        return "baidu_baike";
    }
    return "baidu_search";
}

std::string clean_baike_query(std::string query) {
    query = trim_copy(query);
    const std::string puncts[] = {"？", "?", "。", ".", "！", "!", "，", ","};
    bool removed = true;
    while (removed) {
        removed = false;
        query = trim_copy(query);
        for (const auto& punct : puncts) {
            if (query.size() >= punct.size() &&
                query.compare(query.size() - punct.size(), punct.size(), punct) == 0) {
                query.erase(query.size() - punct.size());
                removed = true;
                break;
            }
        }
    }
    const std::string prefixes[] = {"介绍一下", "介绍下", "查一下", "查下", "百度百科", "百科"};
    for (const auto& prefix : prefixes) {
        size_t pos = query.find(prefix);
        if (pos == 0) {
            query.erase(0, prefix.size());
            break;
        }
    }
    const std::string suffixes[] = {"是什么", "是谁", "的百科", "百科"};
    for (const auto& suffix : suffixes) {
        size_t pos = query.find(suffix);
        if (pos != std::string::npos) {
            query.erase(pos, suffix.size());
        }
    }
    query = trim_copy(query);
    return query.empty() ? "百度" : query;
}

std::string extract_ai_search_answer(const std::string& json) {
    size_t choices = json.find("\"choices\"");
    if (choices == std::string::npos) return "";
    size_t message = json.find("\"message\"", choices);
    if (message == std::string::npos) return "";
    return trim_copy(json_string_value(json.substr(message), "content"));
}

std::string clean_spoken_answer(std::string text) {
    const std::string marks[] = {"**", "__", "`", "#", "- "};
    for (const auto& mark : marks) {
        size_t pos = 0;
        while ((pos = text.find(mark, pos)) != std::string::npos) {
            text.erase(pos, mark.size());
        }
    }
    return trim_copy(text);
}

std::string baidu_baike_answer(const std::string& query, const std::string& api_key) {
    std::string lemma = clean_baike_query(query);
    std::string url = "https://appbuilder.baidu.com/v2/baike/lemma/get_content"
        "?search_type=lemmaTitle&search_key=" + url_escape(lemma);
    std::vector<std::string> headers = {
        "Authorization: Bearer " + api_key,
        "Content-Type: application/json"
    };

    std::string body = http_get_with_headers(url, headers, 10);

    std::string result = first_json_object_in_array(body, "result");
    if (result.empty()) {
        size_t result_pos = body.find("\"result\"");
        if (result_pos != std::string::npos) {
            result = find_json_object(body.substr(result_pos));
        }
    }

    std::string title = json_string_value(result, "lemma_title");
    std::string desc = json_string_value(result, "lemma_desc");
    std::string summary = json_string_value(result, "summary");
    if (summary.empty()) summary = json_string_value(result, "abstract_plain");

    std::string answer;
    if (!title.empty()) answer += title + "：";
    if (!desc.empty()) answer += desc + "。";
    answer += summary;
    answer = trim_copy(answer);
    if (answer.size() > 420) {
        answer = answer.substr(0, 420) + "。";
    }
    return answer;
}

std::string baidu_ai_search_answer(const std::string& query,
                                   const std::string& source,
                                   const std::string& api_key) {
    std::string effective_query = query;
    std::string recency;
    if (source == "baidu_news") {
        effective_query += " 最新新闻";
        recency = "\"search_recency_filter\":\"week\",";
    }

    std::string model = "ernie-4.5-turbo-32k";
    if (const char* env_model = std::getenv("BAIDU_AI_SEARCH_MODEL")) {
        if (env_model[0]) model = env_model;
    }

    std::string instruction =
        "你是语音助手。请只用中文口语回答，控制在80字以内；"
        "不要使用 Markdown、星号、编号或引用标记；"
        "新闻类最多说三点；不确定时明确说未查到可靠信息。";

    std::string body =
        "{"
        "\"messages\":[{\"role\":\"user\",\"content\":\"" + json_escape(effective_query) + "\"}],"
        "\"search_source\":\"baidu_search_v2\","
        "\"resource_type_filter\":[{\"type\":\"web\",\"top_k\":4}],"
        + recency +
        "\"stream\":false,"
        "\"model\":\"" + json_escape(model) + "\","
        "\"instruction\":\"" + json_escape(instruction) + "\","
        "\"temperature\":0.1,"
        "\"top_p\":0.5,"
        "\"search_mode\":\"required\","
        "\"enable_reasoning\":false,"
        "\"enable_deep_search\":false,"
        "\"enable_followup_queries\":false,"
        "\"enable_corner_markers\":false,"
        "\"response_format\":\"text\","
        "\"max_completion_tokens\":\"160\""
        "}";

    std::vector<std::string> headers = {
        "X-Appbuilder-Authorization: Bearer " + api_key,
        "Content-Type: application/json"
    };
    std::string response = http_post_json(
        "https://qianfan.baidubce.com/v2/ai_search/chat/completions",
        body,
        headers,
        15);
    return extract_ai_search_answer(response);
}

std::string query_open_domain_answer(const Action& action) {
    std::string query = trim_copy(action.target.empty() ? action.response_text : action.target);
    if (query.empty()) return "你想问什么问题？";

    std::string api_key = baidu_api_key();
    if (api_key.empty()) {
        fprintf(stderr, "[OpenDomainQA] Missing API key. Set BAIDU_AI_SEARCH_API_KEY.\n");
        return "还没有配置百度搜索的 API Key。";
    }

    std::string source = extract_open_domain_source(action);
    fprintf(stdout, "[OpenDomainQA] source=%s query=\"%s\"\n", source.c_str(), query.c_str());

    std::string answer;
    if (source == "baidu_baike") {
        answer = baidu_baike_answer(query, api_key);
        if (answer.empty()) {
            answer = baidu_ai_search_answer(query + " 百度百科", "baidu_search", api_key);
        }
    } else {
        answer = baidu_ai_search_answer(query, source, api_key);
    }

    if (answer.empty()) {
        return "我没有查到可靠结果，你可以换个说法再问一次。";
    }
    return clean_spoken_answer(answer);
}

}

SemanticEngine::SemanticEngine()
    : m_conversation(8) {
    // 初始化 Action 处理器为默认
    for (int i = 0; i < 13; ++i) {
        m_action_handlers[i] = nullptr;
    }
}

SemanticEngine::~SemanticEngine() {
    shutdown();
}

bool SemanticEngine::init(const std::string& model_path,
                          int n_ctx, int n_threads, bool use_gpu) {
    fprintf(stdout, "[SemanticEngine] Initializing, model=%s, ctx=%d, threads=%d, gpu=%d\n",
            model_path.c_str(), n_ctx, n_threads, (int)use_gpu);

    // 1. 加载 LLM 模型
    if (!m_llm.load_model(model_path, n_ctx, n_threads, use_gpu)) {
        fprintf(stderr, "[SemanticEngine] Failed to load LLM model\n");
        return false;
    }

    fprintf(stdout, "[SemanticEngine] LLM loaded: %s\n", m_llm.model_info().c_str());

    // 2. 设置系统提示词
    std::string system_prompt =
        "你是一个智能语音助手，运行在 macOS 系统上。\n"
        "你的任务是根据用户的语音输入理解语义，拆解任务，给出简短回复，并在安全时给出动作。\n"
        "\n"
        "你必须只输出一个 JSON 对象，不要使用 Markdown，不要输出代码块，不要额外解释。\n"
        "使用结构化 ReAct：先规划 steps，再由程序执行每个 action，最后用 reply 做语音播报。\n"
        "JSON 字段固定为：reply, steps, confidence。\n"
        "steps 是数组，每个元素字段固定为：action, target, params, confidence。\n"
        "action 只能是 open_app、search_web、get_weather、get_time、play_music、open_domain_qa、custom。无动作时 steps 为空数组。\n"
        "不允许输出 system_cmd。\n"
        "reply 使用中文，必须适合最终语音播报；工具类动作尽量控制在 20 个汉字以内，开放域问答先回复“我查一下。”。\n"
        "params 如果没有额外参数就填空字符串。\n"
        "confidence 是 0 到 1 的数字。\n"
        "如果用户一句话里有多个明确任务，必须拆成多个 steps，并保持用户说话顺序。\n"
        "\n"
        "例如：\n"
        "用户说\"打开计算器\"，输出 {\"reply\":\"已打开计算器。\",\"steps\":[{\"action\":\"open_app\",\"target\":\"Calculator\",\"params\":\"\",\"confidence\":0.95}],\"confidence\":0.95}\n"
        "用户说\"帮我搜索 Python 教程\"，输出 {\"reply\":\"已搜索 Python 教程。\",\"steps\":[{\"action\":\"search_web\",\"target\":\"Python 教程\",\"params\":\"\",\"confidence\":0.95}],\"confidence\":0.95}\n"
        "用户说\"北京天气怎么样\"，输出 {\"reply\":\"查询北京天气。\",\"steps\":[{\"action\":\"get_weather\",\"target\":\"北京\",\"params\":\"\",\"confidence\":0.95}],\"confidence\":0.95}\n"
        "用户说\"打开计算器查上海天气\"，输出 {\"reply\":\"已打开计算器，并查询上海天气。\",\"steps\":[{\"action\":\"open_app\",\"target\":\"Calculator\",\"params\":\"\",\"confidence\":0.95},{\"action\":\"get_weather\",\"target\":\"上海\",\"params\":\"\",\"confidence\":0.95}],\"confidence\":0.95}\n"
        "用户说\"量子计算是什么\"，输出 {\"reply\":\"我查一下。\",\"steps\":[{\"action\":\"open_domain_qa\",\"target\":\"量子计算是什么\",\"params\":\"source=baidu_baike\",\"confidence\":0.9}],\"confidence\":0.9}\n"
        "用户说\"最近人工智能有什么新闻\"，输出 {\"reply\":\"我查一下最新消息。\",\"steps\":[{\"action\":\"open_domain_qa\",\"target\":\"最近人工智能有什么新闻\",\"params\":\"source=baidu_news\",\"confidence\":0.9}],\"confidence\":0.9}\n"
        "用户说\"杭州有哪些好玩的地方\"，输出 {\"reply\":\"我查一下。\",\"steps\":[{\"action\":\"open_domain_qa\",\"target\":\"杭州有哪些好玩的地方\",\"params\":\"source=baidu_search\",\"confidence\":0.9}],\"confidence\":0.9}\n"
        "用户说\"现在几点了\"，输出 {\"reply\":\"我看看时间。\",\"steps\":[{\"action\":\"get_time\",\"target\":\"\",\"params\":\"\",\"confidence\":0.95}],\"confidence\":0.95}\n"
        "用户说\"打开网易云音乐\"，输出 {\"reply\":\"已打开网易云音乐。\",\"steps\":[{\"action\":\"play_music\",\"target\":\"网易云音乐\",\"params\":\"command=open;provider=netease_app\",\"confidence\":0.95}],\"confidence\":0.95}\n"
        "用户说\"播放音乐\"，输出 {\"reply\":\"你想听什么歌？\",\"steps\":[],\"confidence\":0.9}\n"
        "用户说\"播放稻香\"，输出 {\"reply\":\"好的，播放稻香。\",\"steps\":[{\"action\":\"play_music\",\"target\":\"稻香\",\"params\":\"command=play;provider=netease_app\",\"confidence\":0.9}],\"confidence\":0.9}\n"
        "用户说\"我要听周杰伦的歌\"，输出 {\"reply\":\"好的，播放周杰伦的歌。\",\"steps\":[{\"action\":\"play_music\",\"target\":\"周杰伦\",\"params\":\"command=artist_play;provider=netease_app\",\"confidence\":0.95}],\"confidence\":0.95}\n"
        "用户说\"暂停音乐\"，输出 {\"reply\":\"已暂停。\",\"steps\":[{\"action\":\"play_music\",\"target\":\"\",\"params\":\"command=pause;provider=netease_app\",\"confidence\":0.95}],\"confidence\":0.95}\n"
        "用户说\"继续播放\"，输出 {\"reply\":\"继续播放。\",\"steps\":[{\"action\":\"play_music\",\"target\":\"\",\"params\":\"command=resume;provider=netease_app\",\"confidence\":0.95}],\"confidence\":0.95}\n"
        "用户说\"下一首\"，输出 {\"reply\":\"下一首。\",\"steps\":[{\"action\":\"play_music\",\"target\":\"\",\"params\":\"command=next;provider=netease_app\",\"confidence\":0.95}],\"confidence\":0.95}\n"
        "用户说\"你好\"，输出 {\"reply\":\"你好，我在。\",\"steps\":[],\"confidence\":0.9}\n"
        "\n"
        "注意：\n"
        "1. 必须准确理解用户意图，不要过度猜测\n"
        "2. 如果不确定，回复用户询问确认\n"
        "3. 只有用户明确要求打开应用、搜索网页、查询天气、询问时间、播放或控制音乐、查询事实知识或实时信息时才加入 step\n"
        "4. 天气必须用 get_weather，不要用 open_domain_qa\n"
        "5. 开放域问答包括百科解释、事实问答、旅游建议、新闻热点、最近/最新信息；百科解释 params 用 source=baidu_baike，新闻热点用 source=baidu_news，其他用 source=baidu_search\n"
        "6. 音乐相关指令必须用 play_music；target 填歌曲名、歌手名或网易云音乐，控制命令写入 params 的 command 字段。用户说\"想听/我要听/播放/放一下 + 歌手 + 的歌/歌曲/音乐\"时，必须使用 command=artist_play，target 只填歌手名；用户说具体歌名时才使用 command=play\n"
        "7. 如果用户说\"结束\"或\"退出\"，只回复告别，steps 为空数组";

    m_conversation.set_system_prompt(system_prompt);

    // 3. 启动任务队列
    if (!m_task_queue.start(&m_llm)) {
        fprintf(stderr, "[SemanticEngine] Failed to start task queue\n");
        m_llm.unload_model();
        return false;
    }

    // 4. 初始化 TTS 引擎（嵌入式 Qwen3-TTS / MLX-Audio）
    {
        const char* env_model_dir = std::getenv("QWEN_TTS_MODEL");
        const char* env_python_home = std::getenv("QWEN_TTS_PYTHON_HOME");
        const char* env_bridge_dir = std::getenv("QWEN_TTS_BRIDGE_DIR");

        std::string tts_model_dir = env_model_dir
            ? env_model_dir
            : "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16";
        std::string tts_python_home = env_python_home
            ? env_python_home
            : "/opt/homebrew/Caskroom/miniforge/base/envs/cosyvoice";
        std::string tts_bridge_dir = env_bridge_dir
            ? env_bridge_dir
            : "/Users/boby/work/study2/python";

        if (init_tts(tts_model_dir, tts_python_home, tts_bridge_dir)) {
            const char* backend = std::getenv("TTS_BACKEND");
            fprintf(stdout, "[SemanticEngine] TTS engine initialized (%s)\n",
                    backend ? backend : "macos");
        } else {
            fprintf(stderr, "[SemanticEngine] TTS engine init failed (non-fatal). "
                    "Set QWEN_TTS_MODEL, QWEN_TTS_PYTHON_HOME, QWEN_TTS_BRIDGE_DIR if your paths differ.\n");
        }
    }

    fprintf(stdout, "[SemanticEngine] Initialization complete\n");
    return true;
}

void SemanticEngine::shutdown() {
    if (!m_task_queue.is_running() && !m_llm.is_loaded()) {
        return;
    }

    // 先停止任务队列，不再提交新的 LLM 任务
    m_task_queue.stop();

    // 等待所有 TTS 播放完成，再销毁引擎
    // 避免 detached 线程仍在访问 Python/TTS 时被析构
    m_tts_initialized = false;
    wait_for_tts();

    // 释放 TTS Python 资源（必须在 Python atexit 处理程序执行之前调用）
    m_tts_engine.shutdown();

    m_llm.unload_model();

    fprintf(stdout, "[SemanticEngine] Shutdown\n");
}

void SemanticEngine::process_asr_result(const std::string& asr_text,
                                         ActionCallback callback) {
    if (asr_text.empty()) return;

    fprintf(stdout, "[SemanticEngine] Processing ASR: \"%s\"\n", asr_text.c_str());

    TaskPlan fast_plan;
    if (build_fast_music_plan(asr_text, fast_plan)) {
        fprintf(stdout, "[SemanticEngine] Fast music path: target=\"%s\" params=\"%s\"\n",
                fast_plan.actions.empty() ? "" : fast_plan.actions.front().target.c_str(),
                fast_plan.actions.empty() ? "" : fast_plan.actions.front().params.c_str());

        m_conversation.add_turn("user", asr_text);
        if (!fast_plan.reply.empty()) {
            m_conversation.add_turn("assistant", fast_plan.reply);
        }

        for (const auto& action : fast_plan.actions) {
            dispatch_action(action);
            if (callback) {
                callback(action);
            }
        }

        bool has_music_action = false;
        for (const auto& action : fast_plan.actions) {
            if (action.type == ActionType::PLAY_MUSIC) {
                has_music_action = true;
                break;
            }
        }

        if (!has_music_action && m_auto_speak && m_tts_initialized && !fast_plan.reply.empty()) {
            speak(fast_plan.reply, m_default_spk_id);
        }

        if (callback) {
            Action final_action;
            final_action.type = ActionType::NONE;
            final_action.action_name = "none";
            final_action.response_text = fast_plan.reply;
            final_action.confidence = fast_plan.confidence;
            callback(final_action);
        }
        return;
    }

    // 1. 将用户输入加入对话历史
    m_conversation.add_turn("user", asr_text);

    // 2. 构建 prompt
    std::string prompt = build_prompt(asr_text);

    // 3. 提交 LLM 推理任务
    m_task_queue.push_task(prompt, [this, callback](const LlmResult& result) {
        if (!result.success) {
            fprintf(stderr, "[SemanticEngine] LLM inference failed: %s\n",
                    result.error_msg.c_str());
            if (callback) {
                Action action;
                action.response_text = "语义模型这次没有生成有效回复。";
                callback(action);
            }
            return;
        }

        fprintf(stdout, "[SemanticEngine] LLM response (%d tokens, %.0fms):\n%s\n",
                result.token_count, result.elapsed_ms,
                result.text.c_str());

        // 4. 解析 LLM 输出
        TaskPlan plan = parse_task_plan(result.text);

        // 5. 将助手回复加入对话历史
        if (!plan.reply.empty()) {
            m_conversation.add_turn("assistant", plan.reply);
        } else if (!result.text.empty()) {
            m_conversation.add_turn("assistant", result.text);
        }

        // 6. 分发 Actions。实际系统调用由外层 callback 执行，SemanticEngine
        // 这里只记录结构化日志并保持自定义 handler 兼容。
        std::vector<std::string> action_summaries;
        std::vector<std::string> weather_summaries;
        std::vector<std::string> qa_summaries;
        for (auto action : plan.actions) {
            if (action.type == ActionType::NONE) continue;
            if (action.type == ActionType::GET_WEATHER) {
                std::string weather_summary = query_weather_summary(action.target);
                weather_summaries.push_back(weather_summary);
                action.params = weather_summary;
                action.response_text.clear();
                fprintf(stdout, "[Weather] %s\n", weather_summary.c_str());
            } else if (action.type == ActionType::OPEN_DOMAIN_QA) {
                std::string qa_answer = query_open_domain_answer(action);
                qa_summaries.push_back(qa_answer);
                action.params = qa_answer;
                action.response_text.clear();
                fprintf(stdout, "[OpenDomainQA] %s\n", qa_answer.c_str());
            } else {
                std::string done = completed_text_for_action(action);
                if (!done.empty()) {
                    action_summaries.push_back(done);
                }
            }
            dispatch_action(action);
            if (callback) {
                callback(action);
            }
        }

        if (!action_summaries.empty() || !weather_summaries.empty() || !qa_summaries.empty()) {
            std::string combined;
            for (const auto& summary : action_summaries) {
                if (!combined.empty()) combined += " ";
                combined += summary;
            }
            for (const auto& summary : weather_summaries) {
                if (!combined.empty()) combined += " ";
                combined += summary;
            }
            for (const auto& summary : qa_summaries) {
                if (!combined.empty()) combined += " ";
                combined += summary;
            }
            plan.reply = combined;
        }

        // 7. TTS 只播最终总结，避免多任务中每个 action 都触发一段声音。
        std::string spoken_text = plan.reply;
        if (spoken_text.empty() && plan.actions.size() == 1) {
            spoken_text = spoken_text_for_action(plan.actions.front());
        }
        bool has_music_action = false;
        for (const auto& action : plan.actions) {
            if (action.type == ActionType::PLAY_MUSIC) {
                has_music_action = true;
                break;
            }
        }
        if (!has_music_action && m_auto_speak && m_tts_initialized && !spoken_text.empty()) {
            speak(spoken_text, m_default_spk_id);
        }

        // 8. 最终回复回调
        if (callback) {
            Action final_action;
            final_action.type = ActionType::NONE;
            final_action.action_name = "none";
            final_action.response_text = plan.reply;
            final_action.confidence = plan.confidence;
            callback(final_action);
        }
    });

}

void SemanticEngine::set_action_handler(ActionType type, ActionCallback handler) {
    std::lock_guard<std::mutex> lock(m_handler_mutex);
    int idx = static_cast<int>(type);
    if (idx >= 0 && idx < 13) {
        m_action_handlers[idx] = handler;
    }
}

void SemanticEngine::dispatch_action(const Action& action) {
    // 默认处理
    default_action_handler(action);

    // 自定义处理器
    std::lock_guard<std::mutex> lock(m_handler_mutex);
    int idx = static_cast<int>(action.type);
    if (idx >= 0 && idx < 13 && m_action_handlers[idx]) {
        m_action_handlers[idx](action);
    }
}

Action SemanticEngine::parse_llm_response(const std::string& llm_output) {
    Action action;

    std::string json_str = find_json_object(llm_output);
    if (json_str.empty()) {
        action.response_text = trim_copy(llm_output);
        return action;
    }

    fprintf(stdout, "[SemanticEngine] Parsing JSON: %s\n", json_str.c_str());

    return action_from_json_object(json_str);
}

TaskPlan SemanticEngine::parse_task_plan(const std::string& llm_output) {
    TaskPlan plan;

    std::string json_str = find_json_object(llm_output);
    if (json_str.empty()) {
        plan.reply = trim_copy(llm_output);
        return plan;
    }

    fprintf(stdout, "[SemanticEngine] Parsing JSON: %s\n", json_str.c_str());

    plan.reply = trim_copy(json_string_value(json_str, "reply"));
    plan.confidence = json_float_value(json_str, "confidence", 0.0f);

    std::vector<std::string> step_objects = json_object_array_value(json_str, "steps");
    if (step_objects.empty()) {
        step_objects = json_object_array_value(json_str, "tasks");
    }

    for (const auto& step_json : step_objects) {
        Action action = action_from_json_object(step_json);
        action.response_text.clear();
        if (action.type == ActionType::SEARCH_WEB && action.target.find("天气") != std::string::npos) {
            action.type = ActionType::GET_WEATHER;
            action.action_name = "get_weather";
        }
        if (action.type != ActionType::NONE) {
            plan.actions.push_back(action);
        }
    }

    // 兼容旧版单 Action 输出：{"reply","action","target","params","confidence"}
    if (plan.actions.empty()) {
        Action single = action_from_json_object(json_str, plan.reply);
        if (single.type != ActionType::NONE) {
            std::string single_reply = single.response_text;
            single.response_text.clear();
            if (single.type == ActionType::SEARCH_WEB && single.target.find("天气") != std::string::npos) {
                single.type = ActionType::GET_WEATHER;
                single.action_name = "get_weather";
            }
            plan.actions.push_back(single);
            if (plan.reply.empty()) {
                plan.reply = single_reply;
            }
        }
    }

    return plan;
}

std::string SemanticEngine::build_prompt(const std::string& user_input) {
    (void)user_input;

    std::vector<std::pair<std::string, std::string>> messages;
    std::string system_prompt = m_conversation.get_system_prompt();
    if (!system_prompt.empty()) {
        messages.push_back({"system", system_prompt});
    }

    for (const auto& turn : m_conversation.get_turns()) {
        if (turn.role == "user" || turn.role == "assistant") {
            messages.push_back({turn.role, turn.content});
        }
    }

    std::string prompt = m_llm.format_chat_prompt(messages);
    if (!prompt.empty()) {
        return prompt;
    }

    return m_conversation.get_context();
}

void SemanticEngine::default_action_handler(const Action& action) {
    const char* type_names[] = {
        "NONE", "OPEN_APP", "SEARCH_WEB", "SEND_MESSAGE",
        "GET_TIME", "SET_REMINDER", "PLAY_MUSIC", "GET_WEATHER",
        "OPEN_DOMAIN_QA", "SYSTEM_CMD", "CUSTOM"
    };

    int idx = static_cast<int>(action.type);
    const char* type_name = (idx >= 0 && idx < 11) ? type_names[idx] : "UNKNOWN";

    fprintf(stdout, "[Action] type=%s action=\"%s\" target=\"%s\" params=\"%s\" response=\"%s\"\n",
            type_name,
            action.action_name.c_str(),
            action.target.c_str(),
            action.params.c_str(),
            action.response_text.c_str());
}

// ============================================================================
// TTS — Embedded Python + Qwen3-TTS
// ============================================================================

bool SemanticEngine::init_tts(const std::string& model_dir,
                               const std::string& python_home,
                               const std::string& bridge_script_dir) {
    if (!std::getenv("QWEN_TTS_STREAMING_INTERVAL")) {
        setenv("QWEN_TTS_STREAMING_INTERVAL", "0.16", 0);
    }
    if (!std::getenv("TTS_STREAM_START_BUFFER_SEC")) {
        setenv("TTS_STREAM_START_BUFFER_SEC", "0.0", 0);
    }

    if (!m_tts_engine.init(model_dir, python_home, bridge_script_dir)) {
        fprintf(stderr, "[TTS] m_tts_engine.init() failed\n");
        return false;
    }

    m_tts_sample_rate = 24000;
    if (const char* env_sample_rate = std::getenv("TTS_SAMPLE_RATE")) {
        int configured_sample_rate = std::atoi(env_sample_rate);
        if (configured_sample_rate > 0) {
            m_tts_sample_rate = configured_sample_rate;
        }
    }

    // 初始化音频播放器（macOS AudioQueue）
    if (!m_audio_player.init(m_tts_sample_rate)) {
        fprintf(stderr, "[TTS] AudioPlayer init failed\n");
        return false;
    }

    m_tts_initialized = true;
    return true;
}

bool SemanticEngine::synthesize_text(const std::string& text,
                                      std::vector<int16_t>& out_pcm,
                                      const std::string& spk_id) {
    if (!m_tts_initialized) {
        fprintf(stderr, "[TTS] Engine not initialized\n");
        return false;
    }

    // 调用 TTS 引擎合成（同步接口 synthesize_sync）
    if (!m_tts_engine.synthesize_sync(text, out_pcm, spk_id)) {
        fprintf(stderr, "[TTS] Synthesis failed for: \"%s\"\n", text.c_str());
        return false;
    }

    return true;
}

void SemanticEngine::speak(const std::string& text, const std::string& spk_id) {
    const uint64_t generation = ++m_tts_generation;
    m_audio_player.interrupt();

    {
        std::lock_guard<std::mutex> lock(m_tts_mutex);
        m_tts_active_jobs++;
    }

    // 在独立线程中播放，不阻塞 LLM 推理；串行化 Python TTS 调用和 AudioQueue 播放
    std::thread([this, text, spk_id, generation]() {
        {
            std::lock_guard<std::mutex> serial_lock(m_tts_serial_mutex);
            if (generation != m_tts_generation.load()) {
                std::lock_guard<std::mutex> lock(m_tts_mutex);
                m_tts_active_jobs--;
                m_tts_cv.notify_all();
                return;
            }

            fprintf(stdout, "[TTS] Streaming reply: \"%s\"\n", text.c_str());

            size_t total_samples = 0;
            size_t buffered_samples = 0;
            int chunk_count = 0;
            bool playback_started = false;
            std::vector<int16_t> pending_pcm;

            double start_buffer_sec = 0.16;
            if (const char* env_buffer = std::getenv("TTS_STREAM_START_BUFFER_SEC")) {
                try {
                    start_buffer_sec = std::max(0.0, std::atof(env_buffer));
                } catch (...) {
                    start_buffer_sec = 0.16;
                }
            }
            const size_t start_buffer_samples = (size_t)(start_buffer_sec * (double)m_tts_sample_rate);

            auto start_playback = [&]() {
                if (playback_started || pending_pcm.empty()) return;
                m_audio_player.start_stream();
                m_audio_player.play(pending_pcm);
                playback_started = true;
                fprintf(stdout, "[TTS] Playback started after buffering %.2f sec (%zu samples)\n",
                        (double)pending_pcm.size() / (double)m_tts_sample_rate, pending_pcm.size());
                pending_pcm.clear();
            };

            bool ok = m_tts_engine.synthesize_stream(text, spk_id,
                [this, &total_samples, &buffered_samples, &chunk_count,
                 &pending_pcm, &playback_started, start_buffer_samples, &start_playback,
                 generation]
                (const std::vector<int16_t>& pcm) {
                    if (generation != m_tts_generation.load()) {
                        return false;
                    }
                    total_samples += pcm.size();
                    chunk_count++;
                    fprintf(stdout, "[TTS] Streaming chunk #%d: %zu samples (total %.2f sec)\n",
                            chunk_count, pcm.size(), (double)total_samples / (double)m_tts_sample_rate);

                    if (!playback_started) {
                        pending_pcm.insert(pending_pcm.end(), pcm.begin(), pcm.end());
                        buffered_samples = pending_pcm.size();
                        if (buffered_samples >= start_buffer_samples) {
                            start_playback();
                        }
                    } else {
                        m_audio_player.play(pcm);
                    }
                    return true;
                });

            if (!playback_started) {
                start_playback();
            }
            m_audio_player.finish_stream();
            if (ok) {
                fprintf(stdout, "[TTS] Stream playback queued: %zu samples in %d chunks (%.2f sec)\n",
                        total_samples, chunk_count, (double)total_samples / (double)m_tts_sample_rate);
                int wait_timeout_ms = 2000;
                if (m_tts_sample_rate > 0) {
                    wait_timeout_ms += (int)((double)total_samples * 1000.0 / (double)m_tts_sample_rate);
                }
                m_audio_player.wait_for_finish(wait_timeout_ms);
            } else {
                if (generation != m_tts_generation.load()) {
                    fprintf(stdout, "[TTS] Stream interrupted by newer reply: \"%s\"\n", text.c_str());
                } else {
                    fprintf(stderr, "[TTS] Stream synthesis failed for: \"%s\"\n", text.c_str());
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(m_tts_mutex);
            m_tts_active_jobs--;
        }
        m_tts_cv.notify_all();
    }).detach();
}

void SemanticEngine::wait_for_tts() {
    std::unique_lock<std::mutex> lock(m_tts_mutex);
    m_tts_cv.wait(lock, [this]() {
        return m_tts_active_jobs == 0;
    });
}

bool SemanticEngine::wait_for_tts_for(int timeout_ms) {
    std::unique_lock<std::mutex> lock(m_tts_mutex);
    return m_tts_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
        return m_tts_active_jobs == 0;
    });
}

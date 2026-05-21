#include "tts_client.h"

#include <cstdio>
#include <cstring>
#include <curl/curl.h>
#include <sstream>

// ============================================================
// cURL 辅助：写回调（将 HTTP response body 写入内存 buffer）
// ============================================================
struct CurlBuffer {
    std::vector<int16_t> pcm_data;
};

static size_t write_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<CurlBuffer*>(userdata);
    size_t total = size * nmemb;
    const char* data = static_cast<const char*>(ptr);

    // CosyVoice 流式返回的是 int16 PCM chunk
    // 每个 chunk 是完整的 int16 序列
    size_t n_samples = total / sizeof(int16_t);
    if (n_samples > 0) {
        const int16_t* samples = reinterpret_cast<const int16_t*>(data);
        buf->pcm_data.insert(buf->pcm_data.end(), samples, samples + n_samples);
    }
    return total;
}

// 带流式回调的写回调
struct CurlStreamBuffer {
    TtsAudioCallback callback;
    std::vector<int16_t> pcm_data; // 累积数据（用于同步模式）
};

static size_t write_callback_stream(void* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<CurlStreamBuffer*>(userdata);
    size_t total = size * nmemb;
    const char* data = static_cast<const char*>(ptr);

    size_t n_samples = total / sizeof(int16_t);
    if (n_samples > 0) {
        const int16_t* samples = reinterpret_cast<const int16_t*>(data);
        // 累积
        buf->pcm_data.insert(buf->pcm_data.end(), samples, samples + n_samples);
        // 回调
        if (buf->callback) {
            buf->callback(samples, n_samples);
        }
    }
    return total;
}

// ============================================================
// TtsClient 实现
// ============================================================

TtsClient::TtsClient() {
    curl_global_init(CURL_GLOBAL_ALL);
}

TtsClient::~TtsClient() {
    cancel();
    if (m_worker.joinable()) {
        m_worker.join();
    }
    curl_global_cleanup();
}

void TtsClient::set_server(const std::string& host, int port) {
    m_host = host;
    m_port = port;
}

bool TtsClient::synthesize_sync(const std::string& text,
                                 std::vector<int16_t>& out_pcm,
                                 const std::string& spk_id) {
    if (text.empty()) return false;

    CURL* curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "[TTS] Failed to init curl\n");
        return false;
    }

    char url[2048];
    snprintf(url, sizeof(url), "http://%s:%d/inference_sft",
             m_host.c_str(), m_port);

    // POST form data: FastAPI Form() 参数需要 POST + form-urlencoded
    char* encoded_text = curl_easy_escape(curl, text.c_str(), (int)text.size());
    char* encoded_spk  = curl_easy_escape(curl, spk_id.c_str(), (int)spk_id.size());

    char postdata[4096];
    snprintf(postdata, sizeof(postdata), "tts_text=%s&spk_id=%s",
             encoded_text, encoded_spk);

    curl_free(encoded_text);
    curl_free(encoded_spk);

    CurlBuffer buf;
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_COPYPOSTFIELDS, postdata);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);

    CURLcode res = curl_easy_perform(curl);

    // 检查 HTTP 状态码
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        fprintf(stderr, "[TTS] HTTP request failed: %s\n", curl_easy_strerror(res));
        return false;
    }

    if (http_code != 200) {
        fprintf(stderr, "[TTS] HTTP %ld — expected 200. Response likely not audio.\n", http_code);
        return false;
    }

    // 最小样本数保护：中文语句至少 ~2000 samples (≈90ms @ 22050Hz)
    if (buf.pcm_data.size() < 1000) {
        fprintf(stderr, "[TTS] Too few samples (%zu) — server likely returned non-audio data\n",
                buf.pcm_data.size());
        return false;
    }

    out_pcm = std::move(buf.pcm_data);
    fprintf(stdout, "[TTS] Synthesized %zu samples (%.1fs) for '%s'\n",
            out_pcm.size(), out_pcm.size() / 22050.0, text.c_str());
    return true;
}

void TtsClient::synthesize_async(const TtsRequest& request) {
    if (request.text.empty()) {
        if (request.on_done) request.on_done(false, "Empty text");
        return;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        // 如果正在运行则等待（简单处理：覆盖）
        m_pending_request = request;
        m_has_pending = true;
        m_cancel = false;
    }

    if (!m_running) {
        m_running = true;
        m_worker = std::thread(&TtsClient::worker_loop, this);
    }

    m_cv.notify_one();
}

void TtsClient::worker_loop() {
    while (m_running) {
        TtsRequest request;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this]() {
                return m_has_pending || !m_running;
            });

            if (!m_running) break;
            if (!m_has_pending) continue;

            request = m_pending_request;
            m_has_pending = false;
            m_busy = true;
        }

        fprintf(stdout, "[TTS] Synthesizing async: \"%s\"\n", request.text.c_str());

        CURL* curl = curl_easy_init();
        if (!curl) {
            fprintf(stderr, "[TTS] Failed to init curl\n");
            m_busy = false;
            if (request.on_done) request.on_done(false, "CURL init failed");
            continue;
        }

        char url[2048];
        snprintf(url, sizeof(url), "http://%s:%d/inference_sft",
                 m_host.c_str(), m_port);

        // POST form data
        char* encoded_text = curl_easy_escape(curl, request.text.c_str(), (int)request.text.size());
        char* encoded_spk  = curl_easy_escape(curl, request.spk_id.c_str(), (int)request.spk_id.size());

        char postdata[4096];
        snprintf(postdata, sizeof(postdata), "tts_text=%s&spk_id=%s",
                 encoded_text, encoded_spk);

        curl_free(encoded_text);
        curl_free(encoded_spk);

        CurlStreamBuffer buf;
        buf.callback = request.on_audio;

        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_COPYPOSTFIELDS, postdata);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback_stream);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
        curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);

        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        bool success = (res == CURLE_OK) && !buf.pcm_data.empty();
        std::string msg;
        if (res != CURLE_OK) {
            msg = curl_easy_strerror(res);
            fprintf(stderr, "[TTS] Async request failed: %s\n", msg.c_str());
        } else if (buf.pcm_data.empty()) {
            msg = "No audio data";
            fprintf(stderr, "[TTS] No audio data received\n");
        } else {
            fprintf(stdout, "[TTS] Async done: %zu samples (%.1fs)\n",
                    buf.pcm_data.size(), buf.pcm_data.size() / 22050.0);
        }

        m_busy = false;
        if (request.on_done) {
            request.on_done(success, msg);
        }
    }
}

void TtsClient::wait_for_done() {
    while (m_busy) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void TtsClient::cancel() {
    m_cancel = true;
}

bool TtsClient::is_busy() const {
    return m_busy;
}
#ifndef TTS_CLIENT_H
#define TTS_CLIENT_H

#include <string>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>

// ============================================================
// CosyVoice TTS HTTP Client
// 调用本地 CosyVoice FastAPI 服务，将文本合成为语音
// 输出: 22050Hz 16-bit PCM 音频
// ============================================================

// TTS 回调：传入 PCM chunk 数据 (int16_t)
using TtsAudioCallback = std::function<void(const int16_t* data, size_t n_samples)>;

// TTS 完成回调
using TtsDoneCallback = std::function<void(bool success, const std::string& msg)>;

struct TtsRequest {
    std::string text;             // 要合成的文本
    std::string spk_id = "中文女"; // 说话人 ID
    std::string mode = "sft";     // 模式: sft / zero_shot / instruct
    std::string instruct_text;    // instruct 模式时的指令文本
    std::string prompt_wav_path;  // zero_shot 模式的参考音频路径

    TtsAudioCallback on_audio;    // 音频流回调（可选）
    TtsDoneCallback   on_done;    // 完成回调（可选）
};

class TtsClient {
public:
    TtsClient();
    ~TtsClient();

    // 设置 CosyVoice 服务器地址
    void set_server(const std::string& host, int port);

    // 同步方式：调用 TTS 并返回完整的 PCM 数据 (22050Hz, int16)
    // 返回 true 表示成功，data 中为完整的音频数据
    bool synthesize_sync(const std::string& text,
                         std::vector<int16_t>& out_pcm,
                         const std::string& spk_id = "中文女");

    // 异步方式：调用 TTS，音频数据通过回调实时传递
    // 调用后立即返回，在后台线程中执行 HTTP 请求
    void synthesize_async(const TtsRequest& request);

    // 等待异步请求完成（阻塞）
    void wait_for_done();

    // 取消当前请求
    void cancel();

    // 是否正在处理
    bool is_busy() const;

private:
    // 后台工作线程
    void worker_loop();

    std::string m_host = "127.0.0.1";
    int m_port = 50000;

    std::atomic<bool> m_running{false};
    std::atomic<bool> m_busy{false};
    std::atomic<bool> m_cancel{false};

    std::thread m_worker;
    std::mutex m_mutex;
    std::condition_variable m_cv;

    TtsRequest m_pending_request;
    bool m_has_pending{false};
};

#endif // TTS_CLIENT_H
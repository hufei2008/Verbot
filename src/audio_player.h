#ifndef AUDIO_PLAYER_H
#define AUDIO_PLAYER_H

#include <vector>
#include <cstdint>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

// ============================================================
// macOS 音频播放器
// 使用 AudioToolbox AudioQueue 实时播放 PCM 音频
// 支持: 22050Hz / 44100Hz, int16, mono
// ============================================================

class AudioPlayer {
public:
    AudioPlayer();
    ~AudioPlayer();

    // 初始化播放器（设置采样率，默认 22050Hz）
    bool init(int sample_rate = 22050);

    // 播放 PCM 数据（int16, mono）
    // 非阻塞：数据被送入播放队列后立即返回
    // 如果正在播放，数据会追加到队列末尾
    void play(const int16_t* data, size_t n_samples);

    // 重载：播放 vector
    void play(const std::vector<int16_t>& data);

    // 流式播放：start 后可多次 play 追加音频，finish 后等队列耗尽再停止
    void start_stream();
    void finish_stream();

    // 等待播放完成（阻塞），最多等待 timeout_ms 毫秒
    // 返回 true 表示播放已完成，false 表示超时
    bool wait_for_finish(int timeout_ms = 30000);

    // 停止播放并清空队列
    void stop();

    // 暂停
    void pause();

    // 恢复
    void resume();

    // 是否正在播放
    bool is_playing() const;

    // 设置播放完成回调
    void set_finished_callback(std::function<void()> cb);

    // 播放音量 (0.0 ~ 1.0)
    void set_volume(float vol);

public:
    // 内部：填充 buffer（从队列取数据）
    // public 因为被 C 回调函数调用
    void fill_buffer(void* buffer);

private:
    // AudioQueue 回调（使用 void* 避免暴露 AudioToolbox 类型）
    static void audio_queue_output_callback(void* inUserData,
                                            void* inAQ,
                                            void* inBuffer);


    int m_sample_rate = 22050;

    void* m_queue{nullptr};   // AudioQueueRef
    void* m_format{nullptr};  // AudioStreamBasicDescription

    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_playing{false};
    std::atomic<bool> m_streaming{false};

    // 播放数据队列
    std::vector<int16_t> m_playback_buffer;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;

    std::function<void()> m_finished_callback;

    // 预分配的 AudioQueue buffers（在 play() 时一起入队）
    std::vector<void*> m_buffers;

    // 统计
    std::atomic<size_t> m_total_played{0};
};

#endif // AUDIO_PLAYER_H

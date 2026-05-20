#pragma once

#include <vector>
#include <cstdint>
#include <atomic>
#include <functional>

class AudioRecorder {
public:
    using AudioDataCallback = std::function<void(const float* data, size_t n)>;

    AudioRecorder();
    ~AudioRecorder();

    // 开始录音
    // continuous=true 时通过回调实时传递数据，不累积到内部 buffer
    bool start(bool continuous = false);

    // 设置实时数据回调（仅 continuous 模式生效）
    void setDataCallback(AudioDataCallback cb);

    // 停止录音
    void stop();

    // 获取录音数据（16kHz 32-bit float PCM, 归一化到 [-1, 1]）
    const std::vector<float>& getAudioData() const;

    // 是否正在录音
    bool isRecording() const;

    // C 回调需要访问（公开给内部使用）
    bool m_continuous{false};
    AudioDataCallback m_callback;

private:
    std::vector<float> m_audioData;
    std::atomic<bool> m_recording{false};

    void* m_queue{nullptr};        // AudioQueueRef
    void* m_format{nullptr};       // AudioStreamBasicDescription
};
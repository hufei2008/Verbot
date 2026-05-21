#include "audio_player.h"

#include <cstdio>
#include <cstring>
#include <AudioToolbox/AudioToolbox.h>

// ============================================================
// AudioPlayer 使用 macOS AudioQueue 播放 PCM
//
// 关键设计：
// - init() 只创建 AudioQueue，不预填 buffer
// - play() 时填第一个 buffer 并 AudioQueueStart
// - fill_buffer 回调中持续填充数据，数据耗尽时 AudioQueueStop
// - wait_for_finish 通过 condition_variable 等待 stop 完成
// ============================================================

// 每个 AudioQueue buffer 的大小（帧数）
static constexpr int kBufferFrames = 1024;

// 回调转发（从 C 函数到 C++ 方法）
struct AudioPlayerContext {
    AudioPlayer* player;
};

// C 风格回调（AudioQueue 要求的格式）
static void audio_queue_callback(void* inUserData,
                                  AudioQueueRef inAQ,
                                  AudioQueueBufferRef inBuffer) {
    auto* ctx = static_cast<AudioPlayerContext*>(inUserData);
    ctx->player->fill_buffer(static_cast<void*>(inBuffer));
}

// ============================================================
// 构造 / 析构
// ============================================================

AudioPlayer::AudioPlayer() {}

AudioPlayer::~AudioPlayer() {
    stop();
}

bool AudioPlayer::init(int sample_rate) {
    if (m_initialized) return true;

    m_sample_rate = sample_rate;

    // 配置 AudioStreamBasicDescription
    AudioStreamBasicDescription asbd;
    memset(&asbd, 0, sizeof(asbd));
    asbd.mSampleRate       = sample_rate;
    asbd.mFormatID         = kAudioFormatLinearPCM;
    asbd.mFormatFlags      = kAudioFormatFlagIsSignedInteger |
                             kAudioFormatFlagIsPacked;
    asbd.mBitsPerChannel   = 16;
    asbd.mChannelsPerFrame = 1;
    asbd.mFramesPerPacket  = 1;
    asbd.mBytesPerFrame    = (asbd.mBitsPerChannel / 8) * asbd.mChannelsPerFrame;
    asbd.mBytesPerPacket   = asbd.mBytesPerFrame * asbd.mFramesPerPacket;
    asbd.mReserved         = 0;

    // 保存 ASBD
    m_format = malloc(sizeof(asbd));
    if (!m_format) return false;
    memcpy(m_format, &asbd, sizeof(asbd));

    // 创建 AudioQueue
    AudioQueueRef queue = nullptr;
    AudioPlayerContext* ctx = new AudioPlayerContext{this};

    OSStatus status = AudioQueueNewOutput(
        &asbd,
        audio_queue_callback,
        ctx,
        NULL,        // NULL = 使用 AudioQueue 内部线程调度回调
        NULL,        // 内部 run loop
        0,           // 保留
        &queue
    );

    if (status != noErr) {
        fprintf(stderr, "[AudioPlayer] AudioQueueNewOutput failed: %d\n", (int)status);
        free(m_format);
        m_format = nullptr;
        delete ctx;
        return false;
    }

    m_queue = static_cast<void*>(queue);

    // 仅分配 3 个 buffer，但不入队（避免静音 buffer 触发停止）
    m_buffers.reserve(3);
    for (int i = 0; i < 3; i++) {
        AudioQueueBufferRef buffer = nullptr;
        status = AudioQueueAllocateBuffer(queue, kBufferFrames * asbd.mBytesPerFrame, &buffer);
        if (status != noErr) {
            fprintf(stderr, "[AudioPlayer] AudioQueueAllocateBuffer failed: %d\n", (int)status);
            continue;
        }
        m_buffers.push_back(static_cast<void*>(buffer));
    }

    // 设置默认音量
    AudioQueueSetParameter(queue, kAudioQueueParam_Volume, 1.0);

    m_initialized = true;
    fprintf(stdout, "[AudioPlayer] Initialized: %dHz, 16-bit mono PCM\n", sample_rate);
    return true;
}

void AudioPlayer::play(const int16_t* data, size_t n_samples) {
    if (!m_initialized || !data || n_samples == 0) return;

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_playback_buffer.insert(m_playback_buffer.end(), data, data + n_samples);
    }

    if (!m_playing) {
        AudioQueueRef queue = static_cast<AudioQueueRef>(m_queue);

        // 填充并 enqueue 所有预分配的 buffer，让 AudioQueue 开始播放
        // 这样 fill_buffer 回调会被触发，消耗缓冲区中的数据
        AudioStreamBasicDescription* asbd = static_cast<AudioStreamBasicDescription*>(m_format);
        for (void* buf_ptr : m_buffers) {
            auto* buffer = static_cast<AudioQueueBufferRef>(buf_ptr);
            // 从 m_playback_buffer 拷贝数据到这个 buffer
            std::lock_guard<std::mutex> lock2(m_mutex);
            if (!m_playback_buffer.empty()) {
                size_t max_frames = buffer->mAudioDataBytesCapacity / asbd->mBytesPerFrame;
                size_t frames = std::min(m_playback_buffer.size(), max_frames);
                memcpy(buffer->mAudioData, m_playback_buffer.data(), frames * sizeof(int16_t));
                if (frames * asbd->mBytesPerFrame < buffer->mAudioDataBytesCapacity) {
                    memset(static_cast<char*>(buffer->mAudioData) + frames * asbd->mBytesPerFrame,
                           0,
                           buffer->mAudioDataBytesCapacity - frames * asbd->mBytesPerFrame);
                }
                buffer->mAudioDataByteSize = buffer->mAudioDataBytesCapacity;
                m_playback_buffer.erase(m_playback_buffer.begin(),
                                        m_playback_buffer.begin() + frames);
                m_total_played += frames;
            } else {
                // 数据太少（< 3 buffer），填静音然后入队
                memset(buffer->mAudioData, 0, buffer->mAudioDataBytesCapacity);
                buffer->mAudioDataByteSize = buffer->mAudioDataBytesCapacity;
            }
            AudioQueueEnqueueBuffer(queue, buffer, 0, nullptr);
        }

        AudioQueueStart(queue, nullptr);
        m_playing = true;
        fprintf(stdout, "[AudioPlayer] Playback started (%zu samples, %dHz)\n",
                n_samples, m_sample_rate);
    }
}

void AudioPlayer::play(const std::vector<int16_t>& data) {
    play(data.data(), data.size());
}

void AudioPlayer::start_stream() {
    m_streaming = true;
}

void AudioPlayer::finish_stream() {
    m_streaming = false;
    m_cv.notify_all();
}

void AudioPlayer::fill_buffer(void* buffer_ptr) {
    auto* buffer = static_cast<AudioQueueBufferRef>(buffer_ptr);
    bool should_enqueue = true;
    bool should_stop = false;

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        AudioStreamBasicDescription* asbd = static_cast<AudioStreamBasicDescription*>(m_format);

        if (m_playback_buffer.empty()) {
            if (m_streaming) {
                memset(buffer->mAudioData, 0, buffer->mAudioDataBytesCapacity);
                buffer->mAudioDataByteSize = buffer->mAudioDataBytesCapacity;
            } else {
                should_enqueue = false;
                should_stop = true;
            }
        } else {
            size_t max_frames = buffer->mAudioDataBytesCapacity / asbd->mBytesPerFrame;
            size_t frames_to_copy = std::min(m_playback_buffer.size(), max_frames);

            // 拷贝实际数据
            memcpy(buffer->mAudioData, m_playback_buffer.data(), frames_to_copy * sizeof(int16_t));
            m_total_played += frames_to_copy;

            size_t bytes_filled = frames_to_copy * asbd->mBytesPerFrame;
            if (bytes_filled < buffer->mAudioDataBytesCapacity) {
                // 不足一帧时剩余部分填充静音
                memset(static_cast<char*>(buffer->mAudioData) + bytes_filled,
                       0,
                       buffer->mAudioDataBytesCapacity - bytes_filled);
            }
            buffer->mAudioDataByteSize = buffer->mAudioDataBytesCapacity;

            // 移除已拷贝数据
            m_playback_buffer.erase(m_playback_buffer.begin(),
                                    m_playback_buffer.begin() + frames_to_copy);
        }
    }

    if (should_stop) {
        AudioQueueRef queue = static_cast<AudioQueueRef>(m_queue);
        OSStatus st = AudioQueueStop(queue, false);
        if (st == noErr) {
            m_playing = false;
        }
        m_cv.notify_all();
        return;
    }

    if (should_enqueue) {
        AudioQueueEnqueueBuffer(static_cast<AudioQueueRef>(m_queue), buffer, 0, nullptr);
    }
}

void AudioPlayer::wait_for_finish() {
    if (!m_playing) return;

    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait_for(lock, std::chrono::seconds(30), [this]() {
        return !m_playing;
    });

    // 等待 AudioQueue 真正播完最后一个 buffer
    lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void AudioPlayer::stop() {
    if (!m_initialized) return;

    AudioQueueRef queue = static_cast<AudioQueueRef>(m_queue);
    if (m_playing) {
        AudioQueueStop(queue, true);  // immediate
        m_playing = false;
    }
    m_streaming = false;

    AudioQueueDispose(queue, true);
    m_queue = nullptr;

    if (m_format) {
        free(m_format);
        m_format = nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_playback_buffer.clear();
        m_buffers.clear();
    }

    m_initialized = false;
    m_cv.notify_all();
    fprintf(stdout, "[AudioPlayer] Stopped and disposed\n");
}

void AudioPlayer::pause() {
    if (!m_initialized || !m_playing) return;
    AudioQueueRef queue = static_cast<AudioQueueRef>(m_queue);
    AudioQueuePause(queue);
}

void AudioPlayer::resume() {
    if (!m_initialized || m_playing) return;
    AudioQueueRef queue = static_cast<AudioQueueRef>(m_queue);
    AudioQueueStart(queue, nullptr);
    m_playing = true;
}

bool AudioPlayer::is_playing() const {
    return m_playing;
}

void AudioPlayer::set_finished_callback(std::function<void()> cb) {
    m_finished_callback = std::move(cb);
}

void AudioPlayer::set_volume(float vol) {
    AudioQueueRef queue = static_cast<AudioQueueRef>(m_queue);
    AudioQueueSetParameter(queue, kAudioQueueParam_Volume, vol);
}

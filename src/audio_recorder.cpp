#include "audio_recorder.h"

#include <AudioToolbox/AudioToolbox.h>
#include <cstring>
#include <iostream>

// 临时 buffer 用于 continuous 模式下的转换
static thread_local std::vector<float> t_floatBuffer;

// 音频队列回调：录制到 buffer
static void audioQueueInputCallback(
    void * __nullable       inUserData,
    AudioQueueRef           inAQ,
    AudioQueueBufferRef     inBuffer,
    const AudioTimeStamp *  inStartTime,
    UInt32                  inNumberPacketDescriptions,
    const AudioStreamPacketDescription * __nullable inPacketDescs) {

    auto * recorder = static_cast<AudioRecorder *>(inUserData);
    if (!recorder) return;

    // inBuffer->mAudioData 是 SInt16 数据 (因为我们设定了 kAudioFormatFlagIsSignedInteger)
    // 需要转换为 float
    const int16_t * samples = static_cast<const int16_t *>(inBuffer->mAudioData);
    size_t nSamples = inBuffer->mAudioDataByteSize / sizeof(int16_t);

    if (recorder->m_continuous) {
        // continuous 模式：通过回调实时传递数据
        if (recorder->m_callback) {
            t_floatBuffer.resize(nSamples);
            for (size_t i = 0; i < nSamples; ++i) {
                t_floatBuffer[i] = samples[i] / 32768.0f;
            }
            recorder->m_callback(t_floatBuffer.data(), nSamples);
        }
    } else {
        // 一次性录制模式：累积到内部 buffer
        auto & data = const_cast<std::vector<float>&>(recorder->getAudioData());
        size_t oldSize = data.size();
        data.resize(oldSize + nSamples);
        for (size_t i = 0; i < nSamples; ++i) {
            data[oldSize + i] = samples[i] / 32768.0f;  // 归一化到 [-1, 1]
        }
    }

    // 重新入队 buffer 以继续录制
    AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, nullptr);
}

AudioRecorder::AudioRecorder() {}

AudioRecorder::~AudioRecorder() {
    stop();
}

void AudioRecorder::setDataCallback(AudioDataCallback cb) {
    m_callback = std::move(cb);
}

bool AudioRecorder::start(bool continuous) {
    if (m_recording) {
        std::cerr << "Already recording." << std::endl;
        return false;
    }

    m_continuous = continuous;

    // 设置音频格式: 16kHz, 16-bit mono PCM
    AudioStreamBasicDescription desc = {};
    desc.mSampleRate = 16000.0;
    desc.mFormatID = kAudioFormatLinearPCM;
    desc.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked;
    desc.mBytesPerPacket = 2;
    desc.mFramesPerPacket = 1;
    desc.mBytesPerFrame = 2;
    desc.mChannelsPerFrame = 1;
    desc.mBitsPerChannel = 16;

    m_format = new AudioStreamBasicDescription(desc);

    AudioQueueRef queue = nullptr;
    OSStatus status = AudioQueueNewInput(
        static_cast<const AudioStreamBasicDescription *>(m_format),
        audioQueueInputCallback,
        this,           // user data
        nullptr,        // run loop (null = current)
        nullptr,        // run loop mode
        0,              // flags
        &queue
    );

    if (status != noErr) {
        std::cerr << "Failed to create AudioQueue: " << status << std::endl;
        delete static_cast<AudioStreamBasicDescription *>(m_format);
        m_format = nullptr;
        return false;
    }

    m_queue = queue;

    // 分配 3 个 buffer，每个 0.5 秒
    UInt32 bufferSize = 16000 * sizeof(int16_t) / 2; // 0.5s = 8000 samples
    for (int i = 0; i < 3; ++i) {
        AudioQueueBufferRef buffer = nullptr;
        AudioQueueAllocateBuffer(queue, bufferSize, &buffer);
        AudioQueueEnqueueBuffer(queue, buffer, 0, nullptr);
    }

    m_audioData.clear();
    m_recording = true;

    status = AudioQueueStart(queue, nullptr);
    if (status != noErr) {
        std::cerr << "Failed to start AudioQueue: " << status << std::endl;
        AudioQueueDispose(queue, true);
        m_queue = nullptr;
        delete static_cast<AudioStreamBasicDescription *>(m_format);
        m_format = nullptr;
        m_recording = false;
        return false;
    }

    return true;
}

void AudioRecorder::stop() {
    if (!m_recording) return;

    m_recording = false;

    AudioQueueRef queue = static_cast<AudioQueueRef>(m_queue);
    if (queue) {
        AudioQueueStop(queue, true);
        AudioQueueDispose(queue, true);
        m_queue = nullptr;
    }

    delete static_cast<AudioStreamBasicDescription *>(m_format);
    m_format = nullptr;
}

const std::vector<float>& AudioRecorder::getAudioData() const {
    return m_audioData;
}

bool AudioRecorder::isRecording() const {
    return m_recording;
}

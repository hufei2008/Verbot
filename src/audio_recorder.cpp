// ============================================================
// AudioRecorder 实现 — 使用 macOS VoiceProcessingIO AudioUnit 录音
//
// VoiceProcessingIO 提供：
// - 回声消除 (AEC)
// - 噪声抑制 (NS)
// - 自动增益控制 (AGC)
//
// 数据流：
// - 16kHz, 32-bit float, 单声道输入
// - 音频回调中通过 AudioUnitRender 获取 PCM 数据
// - 支持连续回调模式和累积模式
// ============================================================

#include "audio_recorder.h"

#include <AudioToolbox/AudioToolbox.h>
#include <AudioUnit/AudioUnit.h>
#include <cstring>
#include <iostream>

// 线程局部缓冲区，复用避免每帧分配
static thread_local std::vector<float> t_floatBuffer;

// 将音频采样数据分发给回调或累积到内部缓冲区
// recorder: 录音器实例
// samples:  float 音频采样数据
// nSamples: 采样数（帧数）
static void deliver_samples(AudioRecorder* recorder, const float* samples, size_t nSamples) {
    if (!recorder || !samples || nSamples == 0) return;

    if (recorder->m_continuous) {
        // 连续模式：通过回调实时传递
        if (recorder->m_callback) {
            recorder->m_callback(samples, nSamples);
        }
        return;
    }

    // 累积模式：追加到内部缓冲区，后续通过 getAudioData() 获取
    auto& data = const_cast<std::vector<float>&>(recorder->getAudioData());
    data.insert(data.end(), samples, samples + nSamples);
}

// VoiceProcessingIO 输入回调 — AudioUnit 渲染新音频数据时触发
// 通过 AudioUnitRender 从 bus 1 获取麦克风输入，
// 然后调用 deliver_samples 分发给录音器
static OSStatus voiceProcessingInputCallback(void* inRefCon,
                                             AudioUnitRenderActionFlags* ioActionFlags,
                                             const AudioTimeStamp* inTimeStamp,
                                             UInt32 inBusNumber,
                                             UInt32 inNumberFrames,
                                             AudioBufferList* ioData) {
    (void)inBusNumber;
    (void)ioData;

    // 获取录音器实例
    auto* recorder = static_cast<AudioRecorder*>(inRefCon);
    if (!recorder || !recorder->isRecording()) {
        return noErr;
    }

    AudioUnit audioUnit = static_cast<AudioUnit>(recorder->m_audioUnit);
    if (!audioUnit) {
        return noErr;
    }

    // 调整线程局部缓冲区大小
    t_floatBuffer.resize(inNumberFrames);

    // 准备 AudioBufferList 接收浮点音频数据
    AudioBufferList bufferList;
    bufferList.mNumberBuffers = 1;
    bufferList.mBuffers[0].mNumberChannels = 1;
    bufferList.mBuffers[0].mDataByteSize = inNumberFrames * sizeof(float);
    bufferList.mBuffers[0].mData = t_floatBuffer.data();

    // 从 bus 1 渲染音频数据到缓冲区
    OSStatus status = AudioUnitRender(audioUnit,
                                      ioActionFlags,
                                      inTimeStamp,
                                      1,
                                      inNumberFrames,
                                      &bufferList);
    if (status != noErr) {
        return status;
    }

    // 将渲染后的音频数据传递给录音器
    deliver_samples(recorder, t_floatBuffer.data(), inNumberFrames);
    return noErr;
}

// VoiceProcessingIO 输出回调 — 输出总线需要静音数据
// 用零填充输出缓冲区，防止反馈循环
static OSStatus voiceProcessingOutputCallback(void* inRefCon,
                                              AudioUnitRenderActionFlags* ioActionFlags,
                                              const AudioTimeStamp* inTimeStamp,
                                              UInt32 inBusNumber,
                                              UInt32 inNumberFrames,
                                              AudioBufferList* ioData) {
    (void)inRefCon;
    (void)ioActionFlags;
    (void)inTimeStamp;
    (void)inBusNumber;
    (void)inNumberFrames;

    if (!ioData) return noErr;

    // 所有输出通道填充静音（零值），避免回声
    for (UInt32 i = 0; i < ioData->mNumberBuffers; ++i) {
        if (ioData->mBuffers[i].mData && ioData->mBuffers[i].mDataByteSize > 0) {
            std::memset(ioData->mBuffers[i].mData, 0, ioData->mBuffers[i].mDataByteSize);
        }
    }
    return noErr;
}

// ============================================================
// 构造 / 析构
// ============================================================

AudioRecorder::AudioRecorder() {}

// 析构时自动停止录音并释放 AudioUnit 资源
AudioRecorder::~AudioRecorder() {
    stop();
}

// 设置音频数据回调（用于连续模式）
void AudioRecorder::setDataCallback(AudioDataCallback cb) {
    m_callback = std::move(cb);
}

// 启动录音
// continuous=true:  数据通过回调实时传递
// continuous=false: 数据累积在 m_audioData 中，通过 getAudioData() 获取
bool AudioRecorder::start(bool continuous) {
    if (m_recording) {
        std::cerr << "Already recording." << std::endl;
        return false;
    }

    m_continuous = continuous;
    m_audioData.clear();

    // 配置音频格式：16kHz, 32-bit float, 单声道 PCM
    AudioStreamBasicDescription desc = {};
    desc.mSampleRate = 16000.0;
    desc.mFormatID = kAudioFormatLinearPCM;
    desc.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    desc.mBytesPerPacket = sizeof(float);
    desc.mFramesPerPacket = 1;
    desc.mBytesPerFrame = sizeof(float);
    desc.mChannelsPerFrame = 1;
    desc.mBitsPerChannel = 32;

    // 保存音频格式描述
    m_format = new AudioStreamBasicDescription(desc);

    // 查找 VoiceProcessingIO AudioUnit 组件（Apple 硬件加速）
    AudioComponentDescription componentDesc = {};
    componentDesc.componentType = kAudioUnitType_Output;
    componentDesc.componentSubType = kAudioUnitSubType_VoiceProcessingIO;
    componentDesc.componentManufacturer = kAudioUnitManufacturer_Apple;

    // 查找 AudioUnit 组件
    AudioComponent component = AudioComponentFindNext(nullptr, &componentDesc);
    if (!component) {
        std::cerr << "Failed to find VoiceProcessingIO AudioUnit." << std::endl;
        delete static_cast<AudioStreamBasicDescription*>(m_format);
        m_format = nullptr;
        return false;
    }

    // 创建 AudioUnit 实例
    AudioUnit audioUnit = nullptr;
    OSStatus status = AudioComponentInstanceNew(component, &audioUnit);
    if (status != noErr || !audioUnit) {
        std::cerr << "Failed to create VoiceProcessingIO AudioUnit: " << status << std::endl;
        delete static_cast<AudioStreamBasicDescription*>(m_format);
        m_format = nullptr;
        return false;
    }

    // 保存 AudioUnit 实例
    m_audioUnit = audioUnit;

    // 启用总线 0（输出），VoiceProcessingIO 要求输出总线开启
    UInt32 enable = 1;
    status = AudioUnitSetProperty(audioUnit,
                                  kAudioOutputUnitProperty_EnableIO,
                                  kAudioUnitScope_Output,
                                  0,
                                  &enable,
                                  sizeof(enable));
    if (status != noErr) {
        std::cerr << "Warning: failed to enable VoiceProcessingIO output: " << status << std::endl;
    }

    // 启用总线 1（输入），用于麦克风录音
    status = AudioUnitSetProperty(audioUnit,
                                  kAudioOutputUnitProperty_EnableIO,
                                  kAudioUnitScope_Input,
                                  1,
                                  &enable,
                                  sizeof(enable));
    if (status != noErr) {
        std::cerr << "Failed to enable VoiceProcessingIO input: " << status << std::endl;
        stop();
        return false;
    }

    // 启用声音处理（AEC + NS），设为 0 表示开启（bypass=0）
    UInt32 bypass = 0;
    status = AudioUnitSetProperty(audioUnit,
                                  kAUVoiceIOProperty_BypassVoiceProcessing,
                                  kAudioUnitScope_Global,
                                  1,
                                  &bypass,
                                  sizeof(bypass));
    if (status != noErr) {
        std::cerr << "Warning: failed to enable voice processing, status=" << status << std::endl;
    }

    // 启用自动增益控制（AGC），稳定音量
    UInt32 agc = 1;
    status = AudioUnitSetProperty(audioUnit,
                                  kAUVoiceIOProperty_VoiceProcessingEnableAGC,
                                  kAudioUnitScope_Global,
                                  1,
                                  &agc,
                                  sizeof(agc));
    if (status != noErr) {
        std::cerr << "Warning: failed to enable VoiceProcessingIO AGC, status=" << status << std::endl;
    }

    // 设置输入总线 1 的流格式（麦克风数据格式）
    status = AudioUnitSetProperty(audioUnit,
                                  kAudioUnitProperty_StreamFormat,
                                  kAudioUnitScope_Output,
                                  1,
                                  &desc,
                                  sizeof(desc));
    if (status != noErr) {
        std::cerr << "Failed to set VoiceProcessingIO input format: " << status << std::endl;
        stop();
        return false;
    }

    // 设置输出总线 0 的流格式
    status = AudioUnitSetProperty(audioUnit,
                                  kAudioUnitProperty_StreamFormat,
                                  kAudioUnitScope_Input,
                                  0,
                                  &desc,
                                  sizeof(desc));
    if (status != noErr) {
        std::cerr << "Warning: failed to set VoiceProcessingIO output format: " << status << std::endl;
    }

    // 注册输入回调：从麦克风获取音频数据并填充静音输出
    AURenderCallbackStruct callback = {};
    callback.inputProc = voiceProcessingInputCallback;
    callback.inputProcRefCon = this;
    status = AudioUnitSetProperty(audioUnit,
                                  kAudioOutputUnitProperty_SetInputCallback,
                                  kAudioUnitScope_Global,
                                  1,
                                  &callback,
                                  sizeof(callback));
    if (status != noErr) {
        std::cerr << "Failed to set VoiceProcessingIO input callback: " << status << std::endl;
        stop();
        return false;
    }

    // 注册输出回调：输出总线填充静音，防止回声
    AURenderCallbackStruct outputCallback = {};
    outputCallback.inputProc = voiceProcessingOutputCallback;
    outputCallback.inputProcRefCon = this;
    status = AudioUnitSetProperty(audioUnit,
                                  kAudioUnitProperty_SetRenderCallback,
                                  kAudioUnitScope_Input,
                                  0,
                                  &outputCallback,
                                  sizeof(outputCallback));
    if (status != noErr) {
        std::cerr << "Warning: failed to set VoiceProcessingIO output callback: " << status << std::endl;
    }

    // 初始化 AudioUnit
    status = AudioUnitInitialize(audioUnit);
    if (status != noErr) {
        std::cerr << "Failed to initialize VoiceProcessingIO AudioUnit: " << status << std::endl;
        stop();
        return false;
    }

    // 标记录音状态
    m_recording = true;

    // 启动 AudioUnit，开始触发音频回调
    status = AudioOutputUnitStart(audioUnit);
    if (status != noErr) {
        std::cerr << "Failed to start VoiceProcessingIO AudioUnit: " << status << std::endl;
        stop();
        return false;
    }

    std::cout << "[AudioRecorder] Started with VoiceProcessingIO (AEC/NS/AGC path), 16kHz float mono" << std::endl;
    return true;
}

// 停止录音，释放 AudioUnit 资源
void AudioRecorder::stop() {
    AudioUnit audioUnit = static_cast<AudioUnit>(m_audioUnit);

    if (audioUnit) {
        // 停止、反初始化、销毁 AudioUnit
        AudioOutputUnitStop(audioUnit);
        AudioUnitUninitialize(audioUnit);
        AudioComponentInstanceDispose(audioUnit);
        m_audioUnit = nullptr;
    }

    // 释放音频格式描述
    delete static_cast<AudioStreamBasicDescription*>(m_format);
    m_format = nullptr;
    m_recording = false;
}

// 获取累积的音频数据（非连续模式下使用）
const std::vector<float>& AudioRecorder::getAudioData() const {
    return m_audioData;
}

// 检查是否正在录音
bool AudioRecorder::isRecording() const {
    return m_recording;
}
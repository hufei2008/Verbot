#include "audio_recorder.h"

#include <AudioToolbox/AudioToolbox.h>
#include <AudioUnit/AudioUnit.h>
#include <cstring>
#include <iostream>

static thread_local std::vector<float> t_floatBuffer;

static void deliver_samples(AudioRecorder* recorder, const float* samples, size_t nSamples) {
    if (!recorder || !samples || nSamples == 0) return;

    if (recorder->m_continuous) {
        if (recorder->m_callback) {
            recorder->m_callback(samples, nSamples);
        }
        return;
    }

    auto& data = const_cast<std::vector<float>&>(recorder->getAudioData());
    data.insert(data.end(), samples, samples + nSamples);
}

static OSStatus voiceProcessingInputCallback(void* inRefCon,
                                             AudioUnitRenderActionFlags* ioActionFlags,
                                             const AudioTimeStamp* inTimeStamp,
                                             UInt32 inBusNumber,
                                             UInt32 inNumberFrames,
                                             AudioBufferList* ioData) {
    (void)inBusNumber;
    (void)ioData;

    auto* recorder = static_cast<AudioRecorder*>(inRefCon);
    if (!recorder || !recorder->isRecording()) {
        return noErr;
    }

    AudioUnit audioUnit = static_cast<AudioUnit>(recorder->m_audioUnit);
    if (!audioUnit) {
        return noErr;
    }

    t_floatBuffer.resize(inNumberFrames);

    AudioBufferList bufferList;
    bufferList.mNumberBuffers = 1;
    bufferList.mBuffers[0].mNumberChannels = 1;
    bufferList.mBuffers[0].mDataByteSize = inNumberFrames * sizeof(float);
    bufferList.mBuffers[0].mData = t_floatBuffer.data();

    OSStatus status = AudioUnitRender(audioUnit,
                                      ioActionFlags,
                                      inTimeStamp,
                                      1,
                                      inNumberFrames,
                                      &bufferList);
    if (status != noErr) {
        return status;
    }

    deliver_samples(recorder, t_floatBuffer.data(), inNumberFrames);
    return noErr;
}

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
    for (UInt32 i = 0; i < ioData->mNumberBuffers; ++i) {
        if (ioData->mBuffers[i].mData && ioData->mBuffers[i].mDataByteSize > 0) {
            std::memset(ioData->mBuffers[i].mData, 0, ioData->mBuffers[i].mDataByteSize);
        }
    }
    return noErr;
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
    m_audioData.clear();

    AudioStreamBasicDescription desc = {};
    desc.mSampleRate = 16000.0;
    desc.mFormatID = kAudioFormatLinearPCM;
    desc.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    desc.mBytesPerPacket = sizeof(float);
    desc.mFramesPerPacket = 1;
    desc.mBytesPerFrame = sizeof(float);
    desc.mChannelsPerFrame = 1;
    desc.mBitsPerChannel = 32;

    m_format = new AudioStreamBasicDescription(desc);

    AudioComponentDescription componentDesc = {};
    componentDesc.componentType = kAudioUnitType_Output;
    componentDesc.componentSubType = kAudioUnitSubType_VoiceProcessingIO;
    componentDesc.componentManufacturer = kAudioUnitManufacturer_Apple;

    AudioComponent component = AudioComponentFindNext(nullptr, &componentDesc);
    if (!component) {
        std::cerr << "Failed to find VoiceProcessingIO AudioUnit." << std::endl;
        delete static_cast<AudioStreamBasicDescription*>(m_format);
        m_format = nullptr;
        return false;
    }

    AudioUnit audioUnit = nullptr;
    OSStatus status = AudioComponentInstanceNew(component, &audioUnit);
    if (status != noErr || !audioUnit) {
        std::cerr << "Failed to create VoiceProcessingIO AudioUnit: " << status << std::endl;
        delete static_cast<AudioStreamBasicDescription*>(m_format);
        m_format = nullptr;
        return false;
    }

    m_audioUnit = audioUnit;

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

    status = AudioUnitSetProperty(audioUnit,
                                  kAudioUnitProperty_StreamFormat,
                                  kAudioUnitScope_Input,
                                  0,
                                  &desc,
                                  sizeof(desc));
    if (status != noErr) {
        std::cerr << "Warning: failed to set VoiceProcessingIO output format: " << status << std::endl;
    }

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

    status = AudioUnitInitialize(audioUnit);
    if (status != noErr) {
        std::cerr << "Failed to initialize VoiceProcessingIO AudioUnit: " << status << std::endl;
        stop();
        return false;
    }

    m_recording = true;

    status = AudioOutputUnitStart(audioUnit);
    if (status != noErr) {
        std::cerr << "Failed to start VoiceProcessingIO AudioUnit: " << status << std::endl;
        stop();
        return false;
    }

    std::cout << "[AudioRecorder] Started with VoiceProcessingIO (AEC/NS/AGC path), 16kHz float mono" << std::endl;
    return true;
}

void AudioRecorder::stop() {
    AudioUnit audioUnit = static_cast<AudioUnit>(m_audioUnit);

    if (audioUnit) {
        AudioOutputUnitStop(audioUnit);
        AudioUnitUninitialize(audioUnit);
        AudioComponentInstanceDispose(audioUnit);
        m_audioUnit = nullptr;
    }

    delete static_cast<AudioStreamBasicDescription*>(m_format);
    m_format = nullptr;
    m_recording = false;
}

const std::vector<float>& AudioRecorder::getAudioData() const {
    return m_audioData;
}

bool AudioRecorder::isRecording() const {
    return m_recording;
}

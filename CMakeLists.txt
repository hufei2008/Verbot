cmake_minimum_required(VERSION 3.14)
project(Verbot C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认 Release 模式
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# macOS 需要 AudioToolbox 框架（AudioUnit 音频播放）
find_library(AUDIO_TOOLBOX AudioToolbox REQUIRED)
find_library(AUDIO_UNIT AudioUnit REQUIRED)
find_package(CURL REQUIRED)

# ════════════════════════════════════════
# 1. llama.cpp — 大语言模型推理引擎
# ════════════════════════════════════════
set(LLAMA_BUILD_COMMON  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP     OFF CACHE BOOL "" FORCE)

# macOS Metal 加速
set(GGML_METAL ON CACHE BOOL "" FORCE)

add_subdirectory(vendor/llama.cpp)

# ════════════════════════════════════════
# 2. whisper.cpp — 语音识别引擎
# ════════════════════════════════════════
# 注意：whisper.cpp 内部 if (NOT TARGET ggml) 检查，
# 所以当第1步已经添加了 ggml 后，这里不会重复添加。
add_subdirectory(vendor/whisper.cpp)

# ════════════════════════════════════════
# 3. TTS Embedded Python 路径
# ════════════════════════════════════════
# conda env: cosyvoice (Python 3.10, numpy, mlx-audio)
set(TTS_PYTHON_PREFIX "/opt/homebrew/Caskroom/miniforge/base/envs/cosyvoice")

# ════════════════════════════════════════
# 4. 主程序 (含语义理解 + TTS)
# ════════════════════════════════════════
add_executable(Verbot
    src/main.cpp
    src/audio_recorder.cpp
    src/audio_recorder.h
    src/llm_client.cpp
    src/llm_client.h
    src/llm_task.cpp
    src/llm_task.h
    src/conversation.cpp
    src/conversation.h
    src/semantic_engine.cpp
    src/semantic_engine.h
    src/tts_engine.cpp
    src/tts_engine.h
    src/audio_player.cpp
    src/audio_player.h
)

set_target_properties(Verbot PROPERTIES OUTPUT_NAME asr_demo)

target_include_directories(Verbot PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
    ${TTS_PYTHON_PREFIX}/include/python3.10
    ${TTS_PYTHON_PREFIX}/lib/python3.10/site-packages/numpy/core/include
)

target_link_libraries(Verbot PRIVATE
    whisper
    llama
    ${AUDIO_TOOLBOX}
    ${AUDIO_UNIT}
    CURL::libcurl
    ${TTS_PYTHON_PREFIX}/lib/libpython3.10.dylib
)

# 复制 Python bridge 脚本到构建目录
add_custom_command(TARGET Verbot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/python
        $<TARGET_FILE_DIR:Verbot>/python
    COMMENT "Copying Python bridge scripts to build directory"
)

# ════════════════════════════════════════
# 4. 测试程序：验证 Whisper 识别准确性
# ════════════════════════════════════════
add_executable(test_whisper
    src/test_whisper.cpp
    vendor/whisper.cpp/examples/common-whisper.cpp
    vendor/whisper.cpp/examples/common.cpp
)

target_include_directories(test_whisper PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/vendor/whisper.cpp/examples
)

target_link_libraries(test_whisper PRIVATE
    whisper
)

# ════════════════════════════════════════
# 5. TTS 独立测试程序
# ════════════════════════════════════════
add_executable(test_tts
    src/test_tts.cpp
    src/tts_engine.cpp
    src/tts_engine.h
    src/audio_player.cpp
    src/audio_player.h
)

target_include_directories(test_tts PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
    ${TTS_PYTHON_PREFIX}/include/python3.10
    ${TTS_PYTHON_PREFIX}/lib/python3.10/site-packages/numpy/core/include
)

target_link_libraries(test_tts PRIVATE
    ${AUDIO_TOOLBOX}
    ${TTS_PYTHON_PREFIX}/lib/libpython3.10.dylib
)

# 复制 Python bridge 脚本到构建目录
add_custom_command(TARGET test_tts POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/python
        $<TARGET_FILE_DIR:test_tts>/python
    COMMENT "Copying Python bridge scripts to build directory"
)

# ════════════════════════════════════════
# 6. 复制模型文件到构建目录
# ════════════════════════════════════════
add_custom_command(TARGET Verbot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:Verbot>/models
    COMMENT "Copying models to build directory"
)

add_custom_command(TARGET test_whisper POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:test_whisper>/models
    COMMENT "Copying models to build directory"
)

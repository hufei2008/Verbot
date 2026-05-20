cmake_minimum_required(VERSION 3.14)
project(asr_demo C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认 Release 模式
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# macOS 需要 AudioToolbox 框架
find_library(AUDIO_TOOLBOX AudioToolbox REQUIRED)

# 添加 whisper.cpp 子目录
add_subdirectory(vendor/whisper.cpp)

# 我们的 ASR 程序
add_executable(asr_demo
    src/main.cpp
    src/audio_recorder.cpp
    src/audio_recorder.h
)

target_include_directories(asr_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor/whisper.cpp/include
)

target_link_libraries(asr_demo PRIVATE
    whisper
    ${AUDIO_TOOLBOX}
)

# 测试程序：验证 Whisper 识别准确性
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

# 复制模型文件到构建目录
add_custom_command(TARGET asr_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:asr_demo>/models
    COMMENT "Copying models to build directory"
)

add_custom_command(TARGET test_whisper POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:test_whisper>/models
    COMMENT "Copying models to build directory"
)

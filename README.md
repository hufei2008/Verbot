# ASR Demo - 实时语音识别 (Real-time Speech Recognition)

基于 [whisper.cpp](https://github.com/ggerganov/whisper.cpp) 的 macOS 实时语音识别（ASR）程序，支持 VAD（语音活动检测）。

## 🎯 功能

- 实时麦克风录音 + VAD 语音检测
- 自动检测说话起点和终点
- 实时转写为中文文字
- 终端彩色界面

## 📋 系统要求

- macOS 10.15+ (Catalina 及以上)
- 麦克风权限
- **无需 GPU**（纯 CPU 运行，Apple Silicon 性能更佳）

---

## 🚀 方案一：预编译发布包（推荐，最简单）

我直接给你编译好的可执行文件，你解压就能用。

### 获取方式

1. 在我这边运行打包脚本：
   ```bash
   chmod +x scripts/build_release.sh
   ./scripts/build_release.sh
   ```
2. 打包完成后得到 `build/asr_demo-xxxx-macos.zip`
3. 把这个 zip 发给对方

### 对方（其他 Mac 用户）使用方式

1. 解压 `asr_demo-xxxx-macos.zip`
2. 双击 `run.sh`（或在终端中运行 `./run.sh`）
3. 首次运行时会请求麦克风权限，点击"允许"
4. 对着麦克风说话，程序自动识别并显示文字

### 注意事项

- 如果双击 `run.sh` 打不开，可以在终端中运行：
  ```bash
  cd 解压后的目录
  chmod +x run.sh
  ./run.sh
  ```
- 如果提示"无法验证开发者"，请去 **系统设置 → 隐私与安全性 → 安全性** 中点击"仍要打开"
- 或者终端用 `./run.sh` 运行就不会有提示

---

## 🔧 方案二：源码编译（需要对方安装 Xcode/CMake）

如果对方是开发者，或者想自己修改代码，可以直接从源码编译。

### 给对方发什么？

直接把整个 `study2` 文件夹发过去（不包含 `build/` 目录），或者在 Git 上共享。

### 对方编译运行

方式一：一键脚本

```bash
chmod +x scripts/setup_mac.sh
./scripts/setup_mac.sh
```

方式二：手动编译

```bash
# 1. 安装 Xcode Command Line Tools（如果没装）
xcode-select --install

# 2. 安装 CMake（如果没装）
brew install cmake

# 3. 编译
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target asr_demo -j$(sysctl -n hw.ncpu)

# 4. 运行
cd build && ./asr_demo
```

### 编译依赖

| 依赖 | 说明 | 安装方式 |
|------|------|----------|
| Xcode CLT | C++ 编译器 | `xcode-select --install` |
| CMake | 构建系统 | `brew install cmake` |
| AudioToolbox | 系统音频框架 | macOS 自带 |
| whisper.cpp | 语音识别引擎 | 已包含在 `vendor/` 中 |

---

## 📦 模型文件

程序需要以下两个模型文件（已包含在 `models/` 目录中）：

- `models/ggml-tiny.bin` - Whisper tiny 模型（~75MB）
- `models/ggml-silero-v6.2.0.bin` - Silero VAD 模型（~4MB）

如需重新下载：
```bash
# 从 Hugging Face 下载 Whisper 模型
curl -L -o models/ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

# 下载 VAD 模型
curl -L -o models/ggml-silero-v6.2.0.bin https://github.com/ggerganov/whisper.cpp/raw/master/models/ggml-silero-v6.2.0.bin
```

## ⚙️ 自定义参数

```bash
# 使用其他 Whisper 模型
./asr_demo path/to/ggml-base.bin

# 指定 VAD 模型
./asr_demo models/ggml-tiny.bin path/to/vad-model.bin
```

## 📄 项目结构

```
study2/
├── CMakeLists.txt            # CMake 构建配置
├── scripts/
│   ├── build_release.sh      # 打包发布脚本（给我自己用）
│   └── setup_mac.sh          # 一键构建脚本（给别人用）
├── src/
│   ├── main.cpp              # 主程序（VAD + ASR 逻辑）
│   ├── audio_recorder.cpp    # 麦克风录音实现
│   └── audio_recorder.h      # 录音头文件
├── models/
│   ├── ggml-tiny.bin         # Whisper 模型
│   └── ggml-silero-v6.2.0.bin # VAD 模型
├── vendor/
│   └── whisper.cpp/          # whisper.cpp 源码
└── test_speech.wav           # 测试音频文件
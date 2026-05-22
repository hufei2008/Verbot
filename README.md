# Verbot — 语音交互 + 语义理解 + TTS 语音助手

基于 **whisper.cpp** (语音识别) + **llama.cpp** (大语言模型) + **Qwen3-TTS / CosyVoice** (语音合成) 的 macOS 语音对话助手。

> ⚠️ 本项目短期内经历了大量迭代。如果你是从旧版（仅 ASR）升级，请务必重新编译，配置文件有重大变更。

---

## ✨ 功能

- **实时语音识别** — 麦克风录音 + Silero VAD 语音端点检测 + Whisper 转写（GPU 加速）
- **语义理解** — 语音识别结果送本地大语言模型（LLM）理解用户意图
- **动作执行** — 打开应用、搜索网页、查询天气、获取时间等
- **流式 TTS 语音合成** — 嵌入 Python 解释器，直接调用 Qwen3-TTS / CosyVoice 模型，零网络延迟
- **文本模式** — 直接 `--text "打开计算器"` 命令行交互，无需麦克风
- **安全退出** — 优雅处理 Ctrl+C，避免 Python 析构崩溃

---

## 📋 系统要求

- **macOS 10.15+**（Catalina 及以上）
- **麦克风权限**（语音模式需要）
- **Apple Silicon** 推荐（Metal GPU 加速）
- **conda 环境**（TTS 功能需要，见下方说明）

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# Xcode Command Line Tools
xcode-select --install

# CMake
brew install cmake

# conda 环境（TTS 功能需要）
# 详见下方 "TTS 配置" 章节
```

### 2. 克隆并初始化子模块

```bash
git clone --recursive <your-repo-url>
cd Verbot
git submodule update --init --recursive
```

### 3. 下载模型文件

将以下模型放入 `models/` 目录：

| 模型 | 说明 | 下载 |
|------|------|------|
| `ggml-medium.bin` | Whisper medium 模型（~1.5GB） | `curl -L -o models/ggml-medium.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin` |
| `ggml-silero-v6.2.0.bin` | Silero VAD 模型（~4MB） | `curl -L -o models/ggml-silero-v6.2.0.bin https://github.com/ggerganov/whisper.cpp/raw/master/models/ggml-silero-v6.2.0.bin` |
| GGUF 格式 LLM 模型 | 如 Gemma 4、Qwen 等 | 自行下载，默认路径 `models/gemma4-26b-a4b-it-q4_k_m.gguf` |

默认配置的 LLM 模型路径可通过命令行参数 `--llm` 覆盖：

```bash
./Verbot --llm models/qwen2.5-7b-instruct-q4_k_m.gguf
```

### 4. 编译

```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

### 5. 运行

**语音模式**（默认）：

```bash
cd build && ./Verbot
```

**文本模式**（直接传一句话，跳过语音识别）：

```bash
./Verbot --text "帮我搜索 Python 教程"
./Verbot --text "打开计算器"
./Verbot --text "北京天气怎么样"
```

**指定 LLM 模型**：

```bash
./Verbot --llm /path/to/model.gguf
```

---

## 🎯 动作系统

语音助手支持以下结构化动作，由 LLM 自动识别并执行：

| 动作 | 触发词例 | 执行方式 |
|------|---------|---------|
| `open_app` | 打开计算器 / 打开浏览器 | `open -a "AppName"` |
| `search_web` | 搜索 Python 教程 | 打开 Google 搜索页 |
| `get_time` | 现在几点了 | 显示当前时间 |
| `get_weather` | 北京天气怎么样 | 调用 Open-Meteo API 查询实时天气 |
| 日常聊天 | 你好 / 谢谢 | 仅语音回复，无动作 |

---

## 🔊 TTS 配置（语音合成）

TTS 功能依赖嵌入式 Python 解释器，需要配置 conda 环境。

### 方案一：macOS 系统 TTS（稳定，音质一般）

```bash
export TTS_BACKEND=macos
export MACOS_TTS_VOICE=Tingting  # 可选，中文语音 Tingting
```

### 方案二：Qwen3-TTS / CosyVoice（高质量，需 conda 环境）

1. 创建 conda 环境：

```bash
conda create -n cosyvoice python=3.10
conda activate cosyvoice
pip install mlx-audio numpy
```

2. 设置环境变量：

```bash
export QWEN_TTS_PYTHON_HOME=/opt/homebrew/Caskroom/miniforge/base/envs/cosyvoice
export QWEN_TTS_BRIDGE_DIR=/path/to/Verbot/python
export QWEN_TTS_MODEL=mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16
```

3. 运行程序，TTS 会自动初始化。

> 注意：CMake 编译时会链接 `libpython3.10.dylib`，如果 conda 环境路径不同，需要修改 `CMakeLists.txt` 中的 `TTS_PYTHON_PREFIX`。

---

## 📦 项目结构

```
Verbot/
├── CMakeLists.txt               # CMake 构建配置
├── scripts/
│   ├── setup_mac.sh             # 一键构建脚本
│   ├── build_release.sh         # 打包发布脚本
│   ├── batch_test.py            # 批量测试脚本
│   ├── strip_vision.py          # 模型裁剪工具
│   └── generate_test_audio.py   # 测试音频生成
├── src/
│   ├── main.cpp                 # 主程序入口（VAD + ASR + LLM 主循环）
│   ├── audio_recorder.cpp/h     # 麦克风录音（AudioUnit）
│   ├── audio_player.cpp/h       # 音频播放（AudioQueue）
│   ├── llm_client.cpp/h         # LLM 推理客户端（llama.cpp）
│   ├── llm_task.cpp/h           # 异步 LLM 任务队列
│   ├── conversation.cpp/h       # 对话历史管理
│   ├── semantic_engine.cpp/h    # 语义引擎（LLM + Action 分发 + TTS）
│   ├── tts_engine.cpp/h         # TTS 引擎（嵌入式 Python 桥接）
│   ├── test_whisper.cpp         # Whisper 识别测试程序
│   └── test_tts.cpp             # TTS 独立测试程序
├── python/
│   ├── cosyvoice_bridge.py      # Python TTS 桥接脚本
│   ├── test_speaker.py          # TTS 音色测试
│   ├── test_stability.py        # TTS 稳定性测试
│   └── assets/                  # TTS 参考音频
├── models/                      # 模型文件目录（需自行下载）
├── vendor/
│   ├── whisper.cpp/             # whisper.cpp 子模块
│   └── llama.cpp/               # llama.cpp 子模块
├── docs/
│   └── superpowers/             # 技能文档
└── output/                      # 测试输出目录
```

---

## ⚙️ 高级配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TTS_BACKEND` | TTS 后端（macos / qwen） | macos |
| `MACOS_TTS_VOICE` | macOS TTS 语音名 | Tingting |
| `QWEN_TTS_PYTHON_HOME` | conda Python 路径 | /opt/.../envs/cosyvoice |
| `QWEN_TTS_BRIDGE_DIR` | bridge 脚本目录 | python/ |
| `QWEN_TTS_MODEL` | TTS 模型名/路径 | mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 |
| `TTS_SAMPLE_RATE` | TTS 采样率 | 24000 |
| `TTS_STREAM_START_BUFFER_SEC` | 流式播放预缓冲秒数 | 0.0 |

### 命令行参数

```bash
./Verbot                          # 语音模式
./Verbot --text "你好"            # 文本模式
./Verbot --text "北京天气" --llm models/qwen.gguf
./Verbot models/ggml-base.bin                          # 指定 Whisper 模型
./Verbot models/ggml-base.bin models/vad-model.bin     # 指定 VAD 模型
```

---

## 🧪 测试程序

```bash
# TTS 测试
cmake --build build --target test_tts -j$(sysctl -n hw.ncpu)
cd build && ./test_tts

# Whisper 准确度测试
cmake --build build --target test_whisper -j$(sysctl -n hw.ncpu)
cd build && ./test_whisper

# Python 端 TTS 稳定性测试
python python/test_stability.py
python python/test_speaker.py
```

---

## 🔧 常见问题

**Q: 语音模式下没有检测到语音？**
调节 VAD 参数（在 `main.cpp` 中）：
- `vad_params.threshold`（默认 0.7，降低以提高灵敏度）
- `SPEECH_RMS_THRESHOLD`（默认 0.025）

**Q: LLM 回复太慢？**
确保使用 Metal 加速的 GGUF 模型，并降低模型大小（如使用 Q4_K_M 量化）。

**Q: TTS 没有声音？**
1. 确认 conda 环境已正确配置
2. 确认环境变量已设置
3. 查看运行日志中是否有 `[TTS]` 相关错误
4. 尝试切换 `TTS_BACKEND=macos` 使用系统 TTS

**Q: 退出时崩溃？**
这是已知的 Python atexit finalization 问题，已通过 `safe_exit()`（调用 `_exit()` 跳过析构）规避。如仍有崩溃，请报告。
#!/bin/bash
#
# setup_mac.sh - 在其他 Mac 电脑上一键构建运行 Verbot
#
# 使用方法:
#   将整个工程文件夹拷贝到新的 Mac 上，然后:
#   cd study2
#   chmod +x scripts/setup_mac.sh
#   ./scripts/setup_mac.sh
#
# 或者直接 git clone 后运行：
#   git clone <your-repo-url> study2
#   cd study2
#   chmod +x scripts/setup_mac.sh
#   ./scripts/setup_mac.sh
#

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

echo "=========================================="
echo " Verbot - macOS 一键构建脚本"
echo "=========================================="
echo ""

# ---- 1. 检查 Xcode Command Line Tools ----
if ! xcode-select -p &>/dev/null; then
    echo "[1/5] 安装 Xcode Command Line Tools..."
    xcode-select --install
    echo "   安装完成后请重新运行此脚本。"
    exit 1
else
    echo "[1/5] ✅ Xcode Command Line Tools 已安装"
fi

# ---- 2. 检查 CMake ----
if ! command -v cmake &>/dev/null; then
    echo "[2/5] ❌ CMake 未安装，正在通过 Homebrew 安装..."

    if ! command -v brew &>/dev/null; then
        echo "    正在安装 Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    brew install cmake
else
    echo "[2/5] ✅ CMake $(cmake --version | head -1 | awk '{print $3}')"
fi

# ---- 3. 检查 whisper.cpp 依赖 ----
echo "[3/5] 检查依赖..."
if [ ! -f "vendor/whisper.cpp/CMakeLists.txt" ]; then
    echo "    whisper.cpp 缺失，尝试初始化子模块..."
    git submodule update --init --recursive 2>/dev/null || {
        echo "    ❌ 子模块初始化失败，请确保 vendor/whisper.cpp 目录存在"
        exit 1
    }
fi
echo "    ✅ whisper.cpp 已就绪"

# ---- 4. 下载模型文件（如果缺失）----
echo "[4/5] 检查模型文件..."

mkdir -p models

download_model() {
    local model="$1"
    local url="$2"
    if [ ! -f "models/${model}" ]; then
        echo "    下载 ${model} ($3)..."
        curl -L --progress-bar -o "models/${model}.tmp" "$url"
        mv "models/${model}.tmp" "models/${model}"
        echo "    ✅ ${model} 下载完成"
    else
        echo "    ✅ ${model} 已存在，跳过"
    fi
}

# Tiny 模型 (74MB) - 速度快，精度够用
download_model "ggml-tiny.bin" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin" \
    "74MB"

# VAD 模型 (885KB) - 语音活动检测
download_model "ggml-silero-v6.2.0.bin" \
    "https://github.com/ggerganov/whisper.cpp/raw/master/models/ggml-silero-vad-v6.2.0.bin" \
    "885KB"

# ---- 5. 编译 ----
echo ""
echo "[5/5] 开始编译..."

mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target Verbot -j "$(sysctl -n hw.ncpu)"

echo ""
echo "=========================================="
echo " ✅ 构建成功!"
echo "=========================================="
echo ""
echo "运行方式:"
echo "  cd build && ./Verbot"
echo ""
echo "注意:"
echo "  - 第一次运行时，macOS 可能会请求麦克风权限"
echo "      请在 系统偏好设置 → 隐私与安全性 → 麦克风 中允许"
echo "  - 如果麦克风没声音，检查系统音量设置"
echo "=========================================="
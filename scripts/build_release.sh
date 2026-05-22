#!/bin/bash
#
# build_release.sh - 构建可分发的 release 包
#
# 使用方法:
#   ./scripts/build_release.sh
#
# 这将在 build/ 目录下编译 Verbot，然后打包成 zip 发布包
#

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

echo "=========================================="
echo " Verbot - Release Package Builder"
echo "=========================================="

# 检查 whisper.cpp 依赖
if [ ! -f "vendor/whisper.cpp/CMakeLists.txt" ]; then
    echo ""
    echo "[!] whisper.cpp 缺失！"
    echo "    请确保 vendor/whisper.cpp 目录存在"
    echo "    可以执行: git submodule update --init --recursive"
    exit 1
fi

# 检查模型文件
if [ ! -f "models/ggml-tiny.bin" ] || [ ! -f "models/ggml-silero-v6.2.0.bin" ]; then
    echo ""
    echo "[!] 模型文件缺失！"
    echo "    请确保 models/ 目录下包含:"
    echo "      - ggml-tiny.bin"
    echo "      - ggml-silero-v6.2.0.bin"
    exit 1
fi

# 创建 build 目录
mkdir -p build

echo ""
echo "[1/3] 配置 CMake (Release 模式)..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

echo ""
echo "[2/3] 编译 Verbot..."
cmake --build build --target Verbot -j "$(sysctl -n hw.ncpu)"

echo ""
echo "[3/3] 打包发布包..."

# 确定版本号（使用 git tag 或日期）
VERSION="$(git describe --tags --always 2>/dev/null || date '+%Y%m%d')"
PKG_NAME="Verbot-${VERSION}-macos"

# 创建发布目录
RELEASE_DIR="build/${PKG_NAME}"
mkdir -p "${RELEASE_DIR}"

# 复制二进制文件
cp build/Verbot "${RELEASE_DIR}/"

# 复制模型文件
cp -R models "${RELEASE_DIR}/models"

# 复制 README
if [ -f "README.md" ]; then
    cp README.md "${RELEASE_DIR}/"
fi

# 创建启动脚本
cat > "${RELEASE_DIR}/run.sh" << 'SCRIPT'
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
./Verbot
SCRIPT
chmod +x "${RELEASE_DIR}/run.sh"

# 打包成 zip
cd build
zip -r "${PKG_NAME}.zip" "${PKG_NAME}"
rm -rf "${PKG_NAME}"

echo ""
echo "=========================================="
echo " ✅ 打包完成!"
echo "=========================================="
echo ""
echo "发布包: build/${PKG_NAME}.zip"
echo ""
echo "其他 Mac 用户只需要:"
echo "  1. 解压 zip"
echo "  2. 双击 run.sh（或在终端运行 ./run.sh）"
echo ""
echo "注意: 需要 macOS 10.15+，无需额外安装依赖"
echo "=========================================="
#!/bin/bash
# ============================================
#  Breeze 电影推荐系统 - Linux/macOS 环境安装脚本
#  用法: bash setup.sh
# ============================================
set -e

echo "=========================================="
echo "  Breeze - 安装环境 (Linux/macOS)"
echo "=========================================="
echo ""

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "[错误] 未找到 conda，请先安装 Anaconda 或 Miniconda"
    echo "下载: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "[OK] conda 已就绪"

# 初始化 conda
eval "$(conda shell.bash hook)"

# 创建环境
echo ""
echo "[1/4] 创建 recbole 环境 (Python 3.9) ..."
conda create -n recbole python=3.9 -y 2>/dev/null || echo "环境可能已存在，跳过创建"
conda activate recbole

# 安装 PyTorch
echo ""
echo "[2/4] 安装 PyTorch (CPU版)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
# GPU版请用: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

# 安装 RecBole
echo ""
echo "[3/4] 安装 RecBole 1.2.1 ..."
pip install recbole==1.2.1 -q

# 安装其他依赖
echo ""
echo "[4/4] 安装其他依赖 ..."
pip install numpy pandas scipy flask pyyaml -q

# 验证
echo ""
echo "=========================================="
echo "  验证安装 ..."
echo "=========================================="
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import recbole; print('RecBole:', recbole.__version__)"
python -c "import flask; print('Flask: OK')"

echo ""
echo "=========================================="
echo "  安装完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate recbole"
echo "  python train.py --device cpu --epochs 200"
echo "  python app/main.py"
echo ""
echo "浏览器打开 http://localhost:5000"
echo "登录: user_1 / 123456"

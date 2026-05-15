@echo off
REM ============================================
REM  Breeze 电影推荐系统 - Windows 环境安装脚本
REM  用法: 双击运行 或 在命令行执行 setup.bat
REM ============================================
setlocal enabledelayedexpansion

echo ==========================================
echo   Breeze - 安装环境
echo ==========================================
echo.

REM 检查 conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 conda，请先安装 Anaconda 或 Miniconda
    echo 下载: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)
echo [OK] conda 已就绪

REM 获取 conda 安装路径
for /f "tokens=*" %%i in ('conda info --base') do set CONDA_BASE=%%i
call "%CONDA_BASE%\Scripts\activate.bat"

REM 创建环境
echo.
echo [1/4] 创建 recbole 环境 (Python 3.9) ...
call conda create -n recbole python=3.9 -y 2>nul
if %errorlevel% neq 0 (
    echo 环境可能已存在，跳过创建
)
call conda activate recbole

REM 安装 PyTorch (CPU版，如需GPU将cu118改为对应cuda版本)
echo.
echo [2/4] 安装 PyTorch ...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
REM GPU版请用下面这行替代:
REM pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

REM 安装 RecBole
echo.
echo [3/4] 安装 RecBole 1.2.1 ...
pip install recbole==1.2.1 -q

REM 安装其他依赖
echo.
echo [4/4] 安装其他依赖 ...
pip install numpy pandas scipy flask pyyaml -q

REM 验证安装
echo.
echo ==========================================
echo   验证安装 ...
echo ==========================================
call conda activate recbole
python -c "import torch; print('PyTorch:', torch.__version__)" 2>nul || echo [警告] PyTorch 未安装成功
python -c "import recbole; print('RecBole:', recbole.__version__)" 2>nul || echo [警告] RecBole 未安装成功
python -c "import flask; print('Flask: OK')" 2>nul || echo [警告] Flask 未安装成功

echo.
echo ==========================================
echo   安装完成!
echo ==========================================
echo.
echo 使用方法:
echo   conda activate recbole
echo   python train.py --device cpu --epochs 200
echo   python app/main.py
echo.
echo 浏览器打开 http://localhost:5000
echo 登录: user_1 / 123456
echo.
pause

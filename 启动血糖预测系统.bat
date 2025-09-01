@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo 🚀 启动血糖预测与智能饮食推荐系统
echo ============================================================
echo.

cd /d "%~dp0"

echo 📋 检查Python环境...
python --version
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    pause
    exit /b 1
)

echo.
echo 📦 检查依赖包...
python -c "import torch, numpy, pandas, matplotlib, seaborn, sklearn" 2>nul
if errorlevel 1 (
    echo ⚠️  部分依赖包缺失，尝试安装...
    pip install torch numpy pandas matplotlib seaborn scikit-learn
)

echo.
echo 🎯 启动血糖预测系统主程序...
echo.

python main.py

echo.
echo ============================================================
echo ✅ 系统运行完成！
echo ============================================================
echo.
echo 📋 系统功能:
echo    - 智能血糖预测服务
echo    - 学术级演示模式
echo    - 多模态特征融合测试
echo    - 系统状态检查
echo    - 完整测试套件
echo.
echo 🎯 使用建议:
echo    1. 选择"2"运行学术级演示模式
echo    2. 选择"1"启动智能预测服务
echo    3. 选择"7"运行完整测试套件
echo    4. 选择"8"查看操作指南
echo.
pause 
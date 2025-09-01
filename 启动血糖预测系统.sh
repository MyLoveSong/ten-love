#!/bin/bash

echo ""
echo "============================================================"
echo "🚀 启动血糖预测与智能饮食推荐系统"
echo "============================================================"
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "📋 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    echo "请先安装Python 3.8+"
    exit 1
fi

python3 --version

echo ""
echo "📦 检查依赖包..."
python3 -c "import torch, numpy, pandas, matplotlib, seaborn, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  部分依赖包缺失，尝试安装..."
    pip3 install torch numpy pandas matplotlib seaborn scikit-learn
fi

echo ""
echo "🎯 启动血糖预测系统主程序..."
echo ""

python3 main.py

echo ""
echo "============================================================"
echo "✅ 系统运行完成！"
echo "============================================================"
echo ""
echo "📋 系统功能:"
echo "   - 智能血糖预测服务"
echo "   - 学术级演示模式"
echo "   - 多模态特征融合测试"
echo "   - 系统状态检查"
echo "   - 完整测试套件"
echo ""
echo "🎯 使用建议:"
echo "   1. 选择'2'运行学术级演示模式"
echo "   2. 选择'1'启动智能预测服务"
echo "   3. 选择'7'运行完整测试套件"
echo "   4. 选择'8'查看操作指南"
echo "" 
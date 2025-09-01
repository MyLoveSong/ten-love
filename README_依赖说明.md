# 血糖预测与智能饮食推荐系统

## 项目简介

本项目是一个基于深度学习的血糖预测与智能饮食推荐系统，利用多模态数据（数值特征、图像、文本）进行血糖水平预测，并提供个性化的饮食建议。

## 系统特点

- 🎯 **智能预测**：基于GluFormer模型的高精度血糖预测
- 🧠 **多模态融合**：结合数值特征、图像和文本数据的多模态分析
- ⚡ **实时服务**：基于FastAPI的高性能API服务
- 📊 **学术级演示**：提供学术级别的模型演示和分析

## 依赖安装

### 基础依赖

```bash
# 安装所有依赖
pip install -r requirements.txt
```

或者手动安装主要依赖：

```bash
# 基础依赖
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib seaborn

# API服务
pip install fastapi uvicorn
pip install python-multipart

# 图像处理
pip install pillow opencv-python

# 文本处理
pip install transformers
pip install sentencepiece

# 其他工具
pip install tqdm
pip install joblib
```

## 快速开始

### 使用main.py启动系统

```bash
python main.py
```

通过main.py的菜单界面，您可以选择以下功能：
1. 启动智能血糖预测服务
2. 运行学术级演示模式
3. 多模态特征融合测试
4. 时空注意力机制测试
5. 多尺度特征提取测试
6. 查看系统状态和模型信息
7. 运行完整测试套件
8. 查看操作指南

### 直接运行特定功能（可选）

您也可以直接运行特定的功能模块：

```bash
# 模型训练
python gluformer_model.py

# 学术级演示
python academic_demo.py

# 智能服务
python smart_integrated_service.py

# 测试API
python test_api.py
```

## 项目结构

- `main.py`: 系统主入口，提供菜单界面
- `gluformer_model.py`: GluFormer模型定义与训练
- `api_service.py`: FastAPI服务
- `multimodal_fusion.py`: 多模态特征融合
- `academic_demo.py`: 学术级演示
- `smart_integrated_service.py`: 智能集成服务
- `test_*.py`: 各模块测试脚本

## 模型架构

GluFormer模型基于LSTM和GRU的融合，结合交叉注意力机制，能够有效捕获血糖变化的长短期依赖关系。

## API文档

API服务提供以下主要端点：

- `/predict/glucose`: 基础血糖预测
- `/predict/multimodal`: 多模态血糖预测
- `/recommend/food`: 食物推荐
- `/analyze/meal`: 膳食分析

详细API文档可在API服务启动后通过 http://localhost:8000/docs 访问。

## 注意事项

- 本系统仅用于研究和学习目的，不应替代专业医疗建议
- 预测结果的准确性受数据质量和模型限制
- 使用真实数据时请确保数据安全和隐私保护 
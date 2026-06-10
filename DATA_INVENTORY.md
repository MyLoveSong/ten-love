# 数据清单

## 结论

`system/` 内数据体量远大于源码体量，包含原始公开数据、重复镜像、派生训练集、模型权重、环境目录和历史输出。第一轮只建立清单，不删除、不移动大型数据。

## 顶层体量

| 路径 | 约大小 | 初步分类 | 处理策略 |
|---|---:|---|---|
| `data/` | 182G | 混合数据，含原始、转换、清洗、扩增、派生训练集 | 保留原位，Git 忽略 |
| `dataset/` | 49G | 公开数据集和推荐系统 benchmark | 保留原位，Git 忽略 |
| `projects/` | 74G | 三个研究子项目，含 glucose 大数据 | 只跟踪源码和轻量文档 |
| `projects/glucose/data/` | 69G | 血糖和营养相关数据镜像 | 保留原位，后续去重 |
| `data_nutrition/` | 2.4G | Nutrition5k 和营养数据库 | 保留原位，Git 忽略 |
| `models/` | 1.4G | 预训练模型和缓存 | Git 忽略，改用下载说明 |
| `XZY/` | 7.5G | 本地环境目录 | Git 忽略 |
| `XZY2/` | 3.3G | 本地环境目录 | Git 忽略 |
| `recbole310/` | 3.0G | 本地环境目录 | Git 忽略 |

## 主要大文件

| 文件 | 约大小 | 观察 |
|---|---:|---|
| `data/external_datasets_stage2.jsonl` | 39.5G | Stage2 外部数据汇总，过大，不适合 Git |
| `data/ml-10m.json` | 20.4G | MovieLens 转换数据 |
| `data/ml-1m.json` | 16.0G | MovieLens 转换数据 |
| `data/merged_all_data.json` | 15.9G | 合并派生数据 |
| `data/train_augmented.json` | 15.6G | 扩增训练数据 |
| `data/train_balanced.json` | 15.4G | 平衡训练数据 |
| `dataset/Yelp JSON/yelp_dataset.tar` | 4.3G | Yelp 原始压缩包 |
| `dataset/amazon-books/Books.csv` | 2.1G | Amazon Books 数据 |
| `backend/world-food-facts_data.tsv` | 1.0G | 食物营养数据 |
| `backend/fooddata-central_data.csv` | 0.9G | FoodData Central 数据 |

## 数据重复风险

观察到 BigIdeas 原始数据在以下位置同时存在，部分文件大小一致：
- `dataset/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.0.0/...`
- `projects/glucose/data/physionet_big_ideas/raw_data/...`

初步判断：`dataset/` 更像原始公开数据镜像，`projects/glucose/data/` 更像子项目本地工作副本。后续建议确定 canonical source，只保留一个真实副本，其它位置用 manifest 或本地 symlink 说明。

## 子项目数据

### Nutrition

| 路径 | 观察 |
|---|---|
| `projects/nutrition/data/` | 约 670M，含预处理数据和扩增数据 |
| `projects/nutrition/data/preprocessed/` | 有预处理报告 JSON |
| `projects/nutrition/outputs/` | 有评估、消融、多 seed 结果 |

### Recommendation

| 路径 | 观察 |
|---|---|
| `projects/recommendation/models/pretrained/vit-base-food101/` | 约 328M，含预训练模型权重 |
| `projects/recommendation/results/constraint_gate_diagnosis.json` | 轻量模块诊断结果 |

### Glucose

| 路径 | 观察 |
|---|---|
| `projects/glucose/data/` | 约 69G，主要体量来源 |
| `projects/glucose/outputs/` | 约 31M，含训练和评估报告 |
| `projects/glucose/suya_model/` | 约 462M，历史模型 |

## Git 管理规则

应纳入 Git：
- 源码。
- 配置模板。
- 轻量 README、审计文档和结果摘要。
- 小型示例数据，前提是不含隐私或密钥。

不应纳入 Git：
- 原始数据集。
- 派生大训练集。
- 模型权重和检查点。
- 本地虚拟环境。
- 大型日志、缓存、逐样本预测和权重 dump。
- 本地 `.env` 或 API key 配置。

## 后续去重 gate

任何删除或移动大型数据前，必须先完成：
1. 生成文件清单、大小、hash 或采样 hash。
2. 标注原始、派生、镜像、结果、缓存五类。
3. 确认至少一个 canonical source。
4. 更新脚本路径引用。
5. 运行最小数据加载验证。

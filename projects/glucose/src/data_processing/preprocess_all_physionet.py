#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整合脚本：预处理所有PhysioNet数据集

基于顶刊论文最佳实践，确保数据高质量且可用于Stage1和Stage2模型
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_processing.physionet_preprocessor import PhysioNetPreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """预处理所有PhysioNet数据集"""

    output_dir = Path("TRAIN/data/preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建预处理器（使用Gluformer推荐的patient_zscore标准化）
    preprocessor = PhysioNetPreprocessor(
        output_dir=output_dir,
        enable_feature_engineering=True,  # TimelyGPT方法：周期性特征
        enable_multimodal=True,  # GlucoLens方法：多模态融合
        normalization_method='patient_zscore'  # Gluformer方法
    )

    datasets = [
        {
            'name': 'physionet_big_ideas',
            'input_path': Path('TRAIN/data/physionet_big_ideas/processed_data.json'),
            'output_filename': 'big_ideas_preprocessed.json'
        },
        {
            'name': 'physionet_cgmacros',
            # CGMacros的processed_data.json只包含元数据，直接从raw_data读取
            'input_path': Path('TRAIN/data/physionet_cgmacros/raw_data'),
            'output_filename': 'cgmacros_preprocessed.json'
        }
    ]

    all_reports = []

    for dataset in datasets:
        input_path = dataset['input_path']

        if not input_path.exists():
            logger.warning(f"数据文件不存在: {input_path}，跳过")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"处理数据集: {dataset['name']}")
        logger.info(f"{'='*70}")

        try:
            report = preprocessor.process_dataset(
                data_path=input_path,
                source_name=dataset['name'],
                output_filename=dataset['output_filename']
            )
            all_reports.append(report)
        except Exception as e:
            logger.error(f"处理 {dataset['name']} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成汇总报告
    if all_reports:
        summary = {
            'total_datasets': len(all_reports),
            'total_patients': sum(r['total_patients'] for r in all_reports),
            'total_samples': sum(r['total_samples'] for r in all_reports),
            'datasets': all_reports
        }

        summary_path = output_dir / 'preprocessing_summary.json'
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*70}")
        logger.info("所有数据集预处理完成！")
        logger.info(f"{'='*70}")
        logger.info(f"数据集数: {summary['total_datasets']}")
        logger.info(f"总患者数: {summary['total_patients']}")
        logger.info(f"总样本数: {summary['total_samples']:,}")
        logger.info(f"汇总报告: {summary_path}")
        logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()

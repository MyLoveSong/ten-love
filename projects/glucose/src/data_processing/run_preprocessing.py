#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行PhysioNet数据预处理的智能脚本

自动检测数据状态并运行预处理
"""

import sys
from pathlib import Path
import json
import subprocess
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_has_glucose(data_path: Path) -> bool:
    """检查数据文件是否包含血糖值"""
    try:
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list) and len(data) > 0:
                first_record = data[0]
            elif isinstance(data, dict) and 'records' in data and len(data['records']) > 0:
                first_record = data['records'][0]
            else:
                return False

            # 检查是否包含血糖值字段
            glucose_fields = ['glucose_mg_dl', 'glucose', 'bg', 'blood_glucose', 'cgm']
            return any(field in first_record for field in glucose_fields)
        return False
    except Exception as e:
        logger.warning(f"检查数据文件失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("PhysioNet数据预处理")
    logger.info("=" * 70)

    # Big Ideas数据集
    big_ideas_processed = Path('TRAIN/data/physionet_big_ideas/processed_data.json')
    big_ideas_raw = Path('TRAIN/data/physionet_big_ideas')

    if big_ideas_processed.exists():
        has_glucose = check_data_has_glucose(big_ideas_processed)
        if has_glucose:
            logger.info("✅ Big Ideas: processed_data.json存在且包含血糖数据")
            input_path = big_ideas_processed
        else:
            logger.warning("⚠️ Big Ideas: processed_data.json存在但不包含血糖数据，尝试从原始文件读取")
            input_path = big_ideas_raw
    else:
        logger.info("ℹ️ Big Ideas: processed_data.json不存在，从原始文件读取")
        input_path = big_ideas_raw

    # 运行Big Ideas预处理（后台运行，因为数据量大）
    logger.info("\n" + "=" * 70)
    logger.info("开始预处理 Big Ideas 数据集 (37GB)")
    logger.info("=" * 70)

    cmd = [
        'python', 'TRAIN/data_processing/physionet_preprocessor.py',
        '--input', str(input_path),
        '--output-dir', 'TRAIN/data/preprocessed',
        '--source-name', 'physionet_big_ideas',
        '--output-filename', 'big_ideas_preprocessed.json'
    ]

    logger.info(f"运行命令: {' '.join(cmd)}")
    logger.info("注意: Big Ideas数据量大，预处理可能需要较长时间...")

    # 在后台运行
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    logger.info(f"Big Ideas预处理进程已启动 (PID: {process.pid})")
    logger.info("预处理将在后台运行，您可以继续使用终端")
    logger.info("查看进度: Get-Process -Id " + str(process.pid))

    # CGMacros数据集（较小，可以同步运行）
    logger.info("\n" + "=" * 70)
    logger.info("检查 CGMacros 数据集")
    logger.info("=" * 70)

    cgmacros_processed = Path('TRAIN/data/physionet_cgmacros/processed_data.json')
    cgmacros_raw = Path('TRAIN/data/physionet_cgmacros/raw_data')

    if cgmacros_processed.exists():
        has_glucose = check_data_has_glucose(cgmacros_processed)
        if has_glucose:
            logger.info("✅ CGMacros: processed_data.json存在且包含血糖数据")
            cgmacros_input = cgmacros_processed
        else:
            logger.warning("⚠️ CGMacros: processed_data.json存在但不包含血糖数据")
            logger.info("CGMacros数据集可能不包含血糖时间序列数据，跳过预处理")
            cgmacros_input = None
    else:
        logger.info("ℹ️ CGMacros: processed_data.json不存在，尝试从原始文件读取")
        cgmacros_input = cgmacros_raw if cgmacros_raw.exists() else None

    if cgmacros_input:
        logger.info("\n开始预处理 CGMacros 数据集")
        cmd_cgm = [
            'python', 'TRAIN/data_processing/physionet_preprocessor.py',
            '--input', str(cgmacros_input),
            '--output-dir', 'TRAIN/data/preprocessed',
            '--source-name', 'physionet_cgmacros',
            '--output-filename', 'cgmacros_preprocessed.json'
        ]

        try:
            result = subprocess.run(cmd_cgm, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info("✅ CGMacros预处理完成")
            else:
                logger.warning(f"⚠️ CGMacros预处理失败: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ CGMacros预处理超时")
        except Exception as e:
            logger.error(f"❌ CGMacros预处理出错: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("预处理任务已启动")
    logger.info("=" * 70)
    logger.info(f"Big Ideas: 后台运行 (PID: {process.pid})")
    logger.info("输出目录: TRAIN/data/preprocessed")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

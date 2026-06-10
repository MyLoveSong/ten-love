#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复CGMacros数据加载问题

如果processed_data.json只包含元数据，重新下载并处理数据
"""

import sys
from pathlib import Path
import json
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent.parent))

from .physionet_cgmacros_downloader import PhysioNetCGMacrosDownloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_processed_data_has_glucose(processed_data_path: Path) -> bool:
    """检查processed_data.json是否包含血糖值"""
    if not processed_data_path.exists():
        return False

    try:
        with open(processed_data_path, 'r', encoding='utf-8') as f:
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
    except Exception as e:
        logger.warning(f"检查processed_data.json失败: {e}")
        return False


def check_zip_file_valid(zip_path: Path) -> bool:
    """检查ZIP文件是否有效"""
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            bad_file = z.testzip()
            if bad_file:
                logger.error(f"ZIP文件损坏: {bad_file}")
                return False
            return True
    except zipfile.BadZipFile:
        logger.error("ZIP文件格式错误")
        return False
    except Exception as e:
        logger.error(f"检查ZIP文件失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("修复CGMacros数据加载问题")
    logger.info("=" * 70)

    output_dir = Path('TRAIN/data/physionet_cgmacros')
    processed_data_path = output_dir / 'processed_data.json'
    zip_path = output_dir / 'raw_data' / 'physionet.org' / 'CGMacros_dateshifted365.zip'

    # 检查processed_data.json
    has_glucose = check_processed_data_has_glucose(processed_data_path)

    if has_glucose:
        logger.info("✅ processed_data.json已包含血糖值，无需修复")
        return

    logger.warning("⚠️ processed_data.json不包含血糖值，需要重新处理")

    # 检查ZIP文件
    if zip_path.exists():
        logger.info(f"检查ZIP文件: {zip_path}")
        zip_valid = check_zip_file_valid(zip_path)

        if not zip_valid:
            logger.warning("⚠️ ZIP文件损坏，需要重新下载")
            logger.info("开始重新下载ZIP文件...")

            downloader = PhysioNetCGMacrosDownloader(output_dir=str(output_dir))
            success = downloader.download(method='auto')

            if not success:
                logger.error("❌ 重新下载失败")
                return

            logger.info("✅ ZIP文件重新下载完成")
        else:
            logger.info("✅ ZIP文件有效")
    else:
        logger.warning("⚠️ ZIP文件不存在，需要下载")
        logger.info("开始下载ZIP文件...")

        downloader = PhysioNetCGMacrosDownloader(output_dir=str(output_dir))
        success = downloader.download(method='auto')

        if not success:
            logger.error("❌ 下载失败")
            return

        logger.info("✅ ZIP文件下载完成")

    # 重新处理数据
    logger.info("=" * 70)
    logger.info("重新处理数据")
    logger.info("=" * 70)

    downloader = PhysioNetCGMacrosDownloader(output_dir=str(output_dir))
    df = downloader.process_downloaded_data()

    if df.empty:
        logger.error("❌ 数据处理失败")
        return

    # 验证结果
    has_glucose = check_processed_data_has_glucose(processed_data_path)

    if has_glucose:
        logger.info("=" * 70)
        logger.info("✅ CGMacros数据修复完成！")
        logger.info("=" * 70)
        logger.info(f"总记录数: {len(df)}")
        logger.info(f"包含血糖值: {df['glucose_mg_dl'].notna().sum() if 'glucose_mg_dl' in df.columns else 0}")
        logger.info(f"唯一患者数: {df['patient_id'].nunique() if 'patient_id' in df.columns else 0}")
    else:
        logger.error("❌ 修复失败：processed_data.json仍不包含血糖值")
        logger.error("请检查原始数据文件是否包含血糖数据")


if __name__ == '__main__':
    main()

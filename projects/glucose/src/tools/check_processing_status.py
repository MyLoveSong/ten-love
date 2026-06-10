#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 PhysioNet 数据集处理状态
"""
import json
from pathlib import Path
from datetime import datetime

def check_status():
    """检查两个数据集的处理状态"""
    base_dir = Path("TRAIN/data")

    datasets = {
        "Big Ideas": base_dir / "physionet_big_ideas",
        "CGMacros": base_dir / "physionet_cgmacros"
    }

    print("=" * 60)
    print("PhysioNet 数据集处理状态检查")
    print("=" * 60)
    print()

    for name, data_dir in datasets.items():
        print(f"【{name}】")
        print(f"  数据目录: {data_dir}")

        # 检查原始数据
        raw_dir = data_dir / "raw_data"
        if raw_dir.exists():
            csv_files = list(raw_dir.rglob("*.csv"))
            part_files = [f for f in csv_files if "_part_" in f.name]
            original_files = [f for f in csv_files if "_part_" not in f.name and not f.name.endswith(".bak")]

            total_size = sum(f.stat().st_size for f in csv_files)
            part_size = sum(f.stat().st_size for f in part_files)

            print(f"  原始CSV文件: {len(original_files)} 个")
            print(f"  拆分后文件: {len(part_files)} 个")
            print(f"  总数据大小: {total_size / (1024**3):.2f} GB")
            print(f"  拆分数据大小: {part_size / (1024**3):.2f} GB")
        else:
            print(f"  [WARN] 原始数据目录不存在")

        # 检查处理结果
        processed_file = data_dir / "processed_data.json"
        metadata_file = data_dir / "metadata.json"

        if processed_file.exists():
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        record_count = len(data)
                    else:
                        record_count = "未知格式"

                file_size = processed_file.stat().st_size / (1024**2)  # MB
                mtime = datetime.fromtimestamp(processed_file.stat().st_mtime)

                print(f"  [OK] 处理完成!")
                print(f"     记录数: {record_count:,}")
                print(f"     文件大小: {file_size:.2f} MB")
                print(f"     处理时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                print(f"  [WARN] 处理文件存在但无法读取: {e}")
        else:
            print(f"  [PROCESSING] 处理中... (processed_data.json 尚未生成)")

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"  [METADATA] 元数据:")
                print(f"     数据集: {metadata.get('dataset_name', 'N/A')}")
                print(f"     版本: {metadata.get('dataset_version', 'N/A')}")
                if 'total_records' in metadata:
                    print(f"     总记录: {metadata['total_records']:,}")
                if 'unique_patients' in metadata:
                    print(f"     唯一患者: {metadata['unique_patients']:,}")
            except Exception as e:
                print(f"  [WARN] 元数据文件无法读取: {e}")

        print()

    print("=" * 60)

if __name__ == "__main__":
    check_status()

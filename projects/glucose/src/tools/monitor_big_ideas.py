#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
持续监控 Big Ideas 数据处理进度
"""
import time
import json
from pathlib import Path
from datetime import datetime

def monitor_processing():
    """监控处理进度"""
    data_dir = Path("TRAIN/data/physionet_big_ideas")
    raw_dir = data_dir / "raw_data"
    processed_file = data_dir / "processed_data.json"
    metadata_file = data_dir / "metadata.json"

    print("=" * 70)
    print("Big Ideas 数据处理监控")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    check_count = 0
    last_file_count = 0

    while True:
        check_count += 1
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 检查处理结果文件
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

                print(f"\n{'='*70}")
                print(f"[{current_time}] 处理完成!")
                print(f"{'='*70}")
                print(f"记录数: {record_count:,}")
                print(f"文件大小: {file_size:.2f} MB")
                print(f"完成时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    print(f"\n元数据:")
                    print(f"  数据集: {metadata.get('dataset_name', 'N/A')}")
                    print(f"  版本: {metadata.get('dataset_version', 'N/A')}")
                    if 'total_records' in metadata:
                        print(f"  总记录: {metadata['total_records']:,}")
                    if 'unique_patients' in metadata:
                        print(f"  唯一患者: {metadata['unique_patients']:,}")

                break
            except Exception as e:
                print(f"[{current_time}] [ERROR] 无法读取处理文件: {e}")

        # 检查拆分文件数量
        if raw_dir.exists():
            part_files = list(raw_dir.rglob("*_part_*.csv"))
            current_file_count = len(part_files)

            if current_file_count != last_file_count:
                total_size = sum(f.stat().st_size for f in part_files)
                total_gb = total_size / (1024**3)
                print(f"[{current_time}] 检查 #{check_count}: 拆分文件数={current_file_count}, 总大小={total_gb:.2f} GB")
                last_file_count = current_file_count
            else:
                print(f"[{current_time}] 检查 #{check_count}: 等待处理完成... (拆分文件: {current_file_count})")
        else:
            print(f"[{current_time}] [WARN] 原始数据目录不存在")

        # 等待30秒后再次检查
        time.sleep(30)

if __name__ == "__main__":
    try:
        monitor_processing()
    except KeyboardInterrupt:
        print("\n\n监控已停止")

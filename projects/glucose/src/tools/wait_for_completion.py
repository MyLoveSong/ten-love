#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
等待 Big Ideas 数据处理完成并通知
"""
import time
import json
from pathlib import Path
from datetime import datetime

def wait_for_completion():
    """等待处理完成"""
    data_dir = Path("TRAIN/data/physionet_big_ideas")
    processed_file = data_dir / "processed_data.json"
    metadata_file = data_dir / "metadata.json"

    print("=" * 70)
    print("Big Ideas 数据处理监控")
    print("=" * 70)
    print(f"开始监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("等待 processed_data.json 文件生成...")
    print("=" * 70)
    print()

    check_interval = 60  # 每60秒检查一次
    check_count = 0

    while True:
        check_count += 1
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if processed_file.exists():
            try:
                # 读取处理结果
                with open(processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        record_count = len(data)
                    else:
                        record_count = "未知格式"

                file_size = processed_file.stat().st_size / (1024**2)  # MB
                mtime = datetime.fromtimestamp(processed_file.stat().st_mtime)

                # 读取元数据
                metadata_info = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata_info = json.load(f)

                # 显示完成信息
                print("\n" + "=" * 70)
                print("🎉 处理完成！")
                print("=" * 70)
                print(f"完成时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"检查次数: {check_count}")
                print()
                print("处理结果:")
                print(f"  记录数: {record_count:,}")
                print(f"  文件大小: {file_size:.2f} MB")

                if metadata_info:
                    print()
                    print("元数据:")
                    print(f"  数据集: {metadata_info.get('dataset_name', 'N/A')}")
                    print(f"  版本: {metadata_info.get('dataset_version', 'N/A')}")
                    if 'total_records' in metadata_info:
                        print(f"  总记录: {metadata_info['total_records']:,}")
                    if 'unique_patients' in metadata_info:
                        print(f"  唯一患者: {metadata_info['unique_patients']:,}")
                    if 'glucose_stats' in metadata_info:
                        stats = metadata_info['glucose_stats']
                        print()
                        print("血糖统计:")
                        if stats.get('mean_mg_dl'):
                            print(f"  平均值: {stats['mean_mg_dl']:.2f} mg/dL")
                        if stats.get('std_mg_dl'):
                            print(f"  标准差: {stats['std_mg_dl']:.2f} mg/dL")
                        if stats.get('min_mg_dl'):
                            print(f"  最小值: {stats['min_mg_dl']:.2f} mg/dL")
                        if stats.get('max_mg_dl'):
                            print(f"  最大值: {stats['max_mg_dl']:.2f} mg/dL")

                print()
                print("=" * 70)
                print("✅ Big Ideas 数据处理已完成！")
                print("=" * 70)

                # 保存完成通知到文件
                completion_file = data_dir / "processing_completed.txt"
                with open(completion_file, 'w', encoding='utf-8') as f:
                    f.write(f"处理完成时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"记录数: {record_count:,}\n")
                    f.write(f"文件大小: {file_size:.2f} MB\n")

                break

            except Exception as e:
                print(f"[{current_time}] [ERROR] 无法读取处理文件: {e}")
                time.sleep(check_interval)
        else:
            # 显示进度提示
            if check_count % 5 == 0:  # 每5次检查显示一次
                print(f"[{current_time}] 检查 #{check_count}: 仍在处理中... (已等待 {check_count * check_interval // 60} 分钟)")

            time.sleep(check_interval)

    return True

if __name__ == "__main__":
    try:
        wait_for_completion()
    except KeyboardInterrupt:
        print("\n\n监控已手动停止")

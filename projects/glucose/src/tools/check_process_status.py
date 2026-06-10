#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 Big Ideas 处理进程状态
"""
import subprocess
import json
from pathlib import Path
from datetime import datetime

def check_process():
    """检查处理进程"""
    print("=" * 70)
    print("Big Ideas 处理进程状态检查")
    print("=" * 70)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 检查进程
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-Process python -ErrorAction SilentlyContinue | "
             "Where-Object { $_.CommandLine -like '*physionet_big_ideas*' } | "
             "Select-Object Id, @{Name='CPU';Expression={$_.CPU}}, "
             "@{Name='Memory(MB)';Expression={[math]::Round($_.WorkingSet/1MB,2)}}, StartTime"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        if result.stdout.strip():
            print("运行中的进程:")
            print(result.stdout)
        else:
            print("[WARN] 未找到运行中的处理进程")
            print("可能需要重新启动处理脚本")
    except Exception as e:
        print(f"[ERROR] 检查进程失败: {e}")

    print()

    # 检查文件状态
    data_dir = Path("TRAIN/data/physionet_big_ideas")
    processed_file = data_dir / "processed_data.json"

    if processed_file.exists():
        print("[OK] processed_data.json 已生成!")
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"  记录数: {len(data):,}")
        except:
            pass
    else:
        print("[PROCESSING] processed_data.json 尚未生成")

        # 检查拆分文件
        raw_dir = data_dir / "raw_data"
        if raw_dir.exists():
            part_files = list(raw_dir.rglob("*_part_*.csv"))
            if part_files:
                # 检查最近修改的文件
                recent_files = [f for f in part_files
                              if f.stat().st_mtime > (datetime.now().timestamp() - 3600)]
                print(f"  拆分文件总数: {len(part_files):,}")
                print(f"  最近1小时修改: {len(recent_files)} 个")

                if recent_files:
                    latest = max(part_files, key=lambda f: f.stat().st_mtime)
                    latest_time = datetime.fromtimestamp(latest.stat().st_mtime)
                    print(f"  最新文件: {latest.name}")
                    print(f"  最新修改时间: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print()
    print("=" * 70)

if __name__ == "__main__":
    check_process()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动监控脚本 - 每60秒检查一次处理状态
"""
import time
import subprocess
from datetime import datetime

def check_and_display():
    """检查并显示状态"""
    result = subprocess.run(
        ["python", "TRAIN/tools/check_processing_status.py"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    print(result.stdout)
    if result.stderr:
        print("错误:", result.stderr)

def main():
    """主循环"""
    print("=" * 70)
    print("自动监控已启动 - 每60秒检查一次")
    print("按 Ctrl+C 停止监控")
    print("=" * 70)
    print()

    try:
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 检查状态...")
            check_and_display()
            print("\n" + "-" * 70)
            print("等待60秒后再次检查...")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    main()

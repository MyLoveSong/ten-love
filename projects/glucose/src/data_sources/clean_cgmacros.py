#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理CGMacros损坏的ZIP文件和解压目录
"""

import os
import shutil
import sys
from pathlib import Path

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    base_dir = Path('TRAIN/data/physionet_cgmacros/raw_data/physionet.org')

    # 删除ZIP文件
    zip_file = base_dir / 'CGMacros_dateshifted365.zip'
    if zip_file.exists():
        try:
            zip_file.unlink()
            print(f"[OK] Deleted ZIP file: {zip_file}")
        except Exception as e:
            print(f"[ERROR] Failed to delete ZIP file: {e}")
    else:
        print("[INFO] ZIP file does not exist")

    # 删除解压目录
    extract_dir = base_dir / 'CGMacros_dateshifted365'
    if extract_dir.exists():
        try:
            shutil.rmtree(extract_dir)
            print(f"[OK] Deleted extracted directory: {extract_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to delete extracted directory: {e}")
    else:
        print("[INFO] Extracted directory does not exist")

    print("\nCleanup completed!")
    print("Please place the locally downloaded ZIP file at:")
    print(f"  {zip_file}")

if __name__ == '__main__':
    main()

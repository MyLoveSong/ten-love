#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PhysioNet 大文件拆分脚本

示例:
    python tools/split_large_physionet_files.py ^
        --input-dir TRAIN/data/physionet_big_ideas/raw_data ^
        --threshold-mb 500 ^
        --chunk-rows 200000
"""

import argparse
from pathlib import Path
import logging

from ..utils.large_file_splitter import cli_split_large_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拆分PhysioNet下载目录中的超大CSV文件")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="需要扫描的目录, 例如 TRAIN/data/physionet_big_ideas/raw_data",
    )
    parser.add_argument(
        "--threshold-mb",
        type=int,
        default=200,
        help="触发拆分的文件大小阈值(MB)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=200_000,
        help="拆分时每个子文件的行数",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="拆分完成后保留原始大文件(默认会加 .bak 后缀)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要拆分的文件, 不实际执行",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli_split_large_files(
        input_dir=args.input_dir,
        threshold_mb=args.threshold_mb,
        chunk_rows=args.chunk_rows,
        rename_original=not args.keep_original,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大文件拆分工具

功能:
- 遍历指定目录, 找到超过阈值的CSV文件
- 使用pandas按行分块读取, 保存为多个较小的CSV
- 可选地在拆分完成后对原始大文件改名(避免重复处理)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def split_large_csv_files(
    target_dir: Path,
    size_threshold_mb: int = 200,
    chunk_rows: int = 200_000,
    rename_original: bool = True,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    拆分指定目录下的大型CSV文件

    Args:
        target_dir: 要遍历的目录
        size_threshold_mb: 文件大小阈值(MB), 超过才会拆分
        chunk_rows: 每个块包含的行数
        rename_original: 拆分完成后是否给原始文件加 .bak 后缀
        dry_run: 仅打印将要拆分的文件, 不实际执行

    Returns:
        拆分记录列表
    """
    target_dir = Path(target_dir)
    if not target_dir.exists():
        logger.warning("目录不存在, 跳过拆分: %s", target_dir)
        return []

    split_records: List[Dict[str, Any]] = []
    threshold_bytes = size_threshold_mb * 1024 * 1024

    csv_files = sorted(target_dir.rglob("*.csv"))
    for csv_path in csv_files:
        file_size = csv_path.stat().st_size
        if file_size < threshold_bytes:
            continue

        human_size = round(file_size / (1024 ** 2), 2)
        logger.info("检测到大文件: %s (%.2f MB)", csv_path, human_size)

        chunk_template = csv_path.parent / f"{csv_path.stem}_part_*.csv"
        if list(csv_path.parent.glob(chunk_template.name)):
            logger.info("拆分文件已存在, 跳过: %s", csv_path)
            continue

        if dry_run:
            split_records.append(
                {
                    "file": str(csv_path),
                    "size_mb": human_size,
                    "status": "dry_run",
                }
            )
            continue

        try:
            for idx, chunk in enumerate(
                pd.read_csv(csv_path, chunksize=chunk_rows, low_memory=False)
            ):
                out_path = csv_path.parent / f"{csv_path.stem}_part_{idx:04d}.csv"
                chunk.to_csv(out_path, index=False)
                logger.debug("写出拆分文件: %s", out_path)

            if rename_original:
                bak_path = csv_path.with_suffix(csv_path.suffix + ".bak")
                csv_path.rename(bak_path)
                logger.info("原始文件已重命名为: %s", bak_path)

            split_records.append(
                {
                    "file": str(csv_path),
                    "size_mb": human_size,
                    "parts": idx + 1,
                    "status": "success",
                }
            )
        except Exception as exc:
            logger.error("拆分文件失败 %s: %s", csv_path, exc)
            split_records.append(
                {
                    "file": str(csv_path),
                    "size_mb": human_size,
                    "status": f"failed: {exc}",
                }
            )

    return split_records


def cli_split_large_files(
    input_dir: str,
    threshold_mb: int,
    chunk_rows: int,
    rename_original: bool,
    dry_run: bool = False,
) -> None:
    """CLI封装, 方便脚本调用"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    records = split_large_csv_files(
        target_dir=Path(input_dir),
        size_threshold_mb=threshold_mb,
        chunk_rows=chunk_rows,
        rename_original=rename_original,
        dry_run=dry_run,
    )
    logger.info("拆分任务完成, 处理文件数: %d", len(records))

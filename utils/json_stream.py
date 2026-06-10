#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量 JSON 流式读取工具
======================

Stage2 数据文件（如 `stage2/data/train.json`）动辄上百 MB。
直接 `json.load` 往往会触发 MemoryError。该模块提供
基于标准库 `json.JSONDecoder` 的流式解析方法，可同时
兼容 JSON 数组与 NDJSON（每行一个 JSON）的格式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Iterable, Union

JsonLike = Union[dict, list]


def stream_json_records(path: Union[str, Path]) -> Generator[JsonLike, None, None]:
    """
    逐条读取大型 JSON 文件。

    支持两类输入格式：
    1. `[ {...}, {...}, ... ]` —— 常见的 JSON 数组
    2. `{"x": 1}\n{"x": 2}` —— NDJSON

    Args:
        path: 文件路径

    Yields:
        解析后的 JSON 对象（dict 或 list）
    """

    decoder = json.JSONDecoder()
    path = Path(path)

    with path.open("r", encoding="utf-8") as fh:
        buffer = ""
        eof = False
        array_mode_detected = False

        while not eof:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                eof = True
            buffer += chunk

            while True:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break

                consumed = len(buffer) - len(stripped)
                if consumed:
                    buffer = stripped

                # 跳过分隔符
                if buffer[0] in ",\n\r\t":
                    buffer = buffer[1:]
                    continue

                if buffer[0] == "[":
                    array_mode_detected = True
                    buffer = buffer[1:]
                    continue

                if buffer[0] == "]":
                    buffer = buffer[1:]
                    continue

                try:
                    obj, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    # 需要更多数据
                    break

                yield obj
                buffer = buffer[idx:]

        # NDJSON 结尾可能没有换行
        buffer = buffer.strip()
        if buffer:
            if buffer not in {"[", "]"}:
                try:
                    obj, _ = decoder.raw_decode(buffer)
                    yield obj
                except json.JSONDecodeError:
                    raise ValueError(f"无法解析 {path} 结尾的 JSON 片段")


def load_small_json(path: Union[str, Path]) -> JsonLike:
    """用于加载小文件的便捷函数，保持接口统一。"""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)

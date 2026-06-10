"""
ASCII图表工具
提供命令行友好的可视化工具
"""

import numpy as np
from typing import List, Tuple, Optional


def create_ascii_chart(
    data_series: List[List[float]],
    labels: Optional[List[str]] = None,
    title: str = "",
    width: int = 80,
    height: int = 10,
    symbols: str = "▁▂▃▄▅▆▇█"
) -> str:
    """
    创建ASCII图表

    Args:
        data_series: 数据系列列表
        labels: 数据系列标签
        title: 图表标题
        width: 图表宽度
        height: 图表高度
        symbols: 用于绘制的符号

    Returns:
        ASCII图表字符串
    """
    if not data_series or not data_series[0]:
        return "没有数据可显示"

    # 确保所有系列长度相同
    max_len = max(len(series) for series in data_series)
    for i, series in enumerate(data_series):
        if len(series) < max_len:
            data_series[i] = series + [series[-1]] * (max_len - len(series))

    # 如果数据点多于宽度，进行降采样
    if max_len > width:
        resampled_data = []
        for series in data_series:
            # 使用平均值降采样
            indices = np.linspace(0, len(series) - 1, width, dtype=int)
            resampled = [series[i] for i in indices]
            resampled_data.append(resampled)
        data_series = resampled_data
        max_len = width

    # 计算所有数据的最小值和最大值
    all_data = [val for series in data_series for val in series]
    min_val = min(all_data)
    max_val = max(all_data)

    # 避免除零错误
    if max_val == min_val:
        max_val = min_val + 1

    # 创建图表
    chart = []

    # 添加标题
    if title:
        chart.append(title.center(width))
        chart.append("-" * width)

    # 创建图表主体
    symbol_count = len(symbols) - 1
    rows = []

    for y in range(height):
        row = []
        for x in range(max_len):
            # 在此位置绘制的符号
            cell_symbols = []

            for series_idx, series in enumerate(data_series):
                if x < len(series):
                    # 计算该数据点在高度上的位置
                    normalized = (series[x] - min_val) / (max_val - min_val)
                    pos = int(normalized * height)

                    # 如果数据点落在当前行，绘制符号
                    if height - 1 - y == pos:
                        # 使用不同的符号或颜色表示不同系列
                        series_symbol = symbols[-1]
                        cell_symbols.append(series_symbol)

            # 如果多个系列在同一位置，使用特殊符号
            if len(cell_symbols) > 1:
                row.append("*")
            elif len(cell_symbols) == 1:
                row.append(cell_symbols[0])
            else:
                row.append(" ")

        rows.append("".join(row))

    chart.extend(rows)

    # 添加图例
    if labels:
        chart.append("-" * width)
        legend = []
        for i, label in enumerate(labels):
            if i < len(data_series):
                legend.append(f"{symbols[-1]} {label}")
        chart.append("  ".join(legend))

    # 添加坐标轴标签
    chart.append("-" * width)

    # 添加最小值和最大值标签
    min_label = f"{min_val:.4f}"
    max_label = f"{max_val:.4f}"
    axis_label = min_label + " " * (width - len(min_label) - len(max_label)) + max_label
    chart.append(axis_label)

    return "\n".join(chart)


def create_ascii_bar_chart(
    values: List[float],
    labels: Optional[List[str]] = None,
    title: str = "",
    width: int = 40
) -> str:
    """
    创建ASCII柱状图

    Args:
        values: 数值列表
        labels: 标签列表
        title: 图表标题
        width: 图表宽度

    Returns:
        ASCII柱状图字符串
    """
    if not values:
        return "没有数据可显示"

    # 确保标签和值的数量相同
    if not labels:
        labels = [f"项目{i+1}" for i in range(len(values))]
    elif len(labels) < len(values):
        labels.extend([f"项目{i+1}" for i in range(len(labels), len(values))])

    # 计算最大值和最大标签长度
    max_val = max(values)
    max_label_len = max(len(label) for label in labels)

    # 避免除零错误
    if max_val == 0:
        max_val = 1

    # 创建图表
    chart = []

    # 添加标题
    if title:
        chart.append(title.center(max_label_len + width + 10))
        chart.append("-" * (max_label_len + width + 10))

    # 创建柱状图
    for i, (value, label) in enumerate(zip(values, labels)):
        # 计算柱子长度
        bar_len = int((value / max_val) * (width - 10))
        bar = "█" * bar_len

        # 添加标签和值
        chart.append(f"{label.ljust(max_label_len)} │ {bar} {value:.4f}")

    return "\n".join(chart)

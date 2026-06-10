

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级智能健康监测集成系统 - 研究可视化界面
基于Streamlit，支持血糖预测、图像识别、健康评估和结果查看
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from app.PIL import Image
import io
import json

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="学术级健康监测集成系统", layout="wide")
st.title("🩺 学术级智能健康监测集成系统 研究可视化 (Streamlit)")
st.markdown("---")

# 侧边栏导航
menu = st.sidebar.radio("功能导航", [
    "血糖预测", "图像识别", "健康综合评估", "系统状态", "API文档"
])

# 血糖预测
if menu == "血糖预测":
    st.header("血糖预测服务")
    st.write("上传血糖相关数据，获得未来血糖趋势预测和健康建议。")

    # 数据上传
    uploaded_file = st.file_uploader("上传血糖数据 (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())

        # 选择一行进行预测
        st.write("选择一行数据进行预测：")
        row_idx = st.number_input("行号", min_value=0, max_value=len(df)-1, value=0)
        input_data = df.iloc[row_idx].to_dict()
        st.json(input_data)

        if st.button("提交预测"):
            with st.spinner("正在预测..."):
                resp = requests.post(f"{API_BASE}/predict/glucose", json=input_data)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success("预测结果：")
                    st.json(result)
                    if "predictions" in result:
                        st.line_chart(result["predictions"])
                else:
                    st.error(f"预测失败: {resp.text}")

# 图像识别
elif menu == "图像识别":
    st.header("医学图像识别服务")
    st.write("上传医学图像，获得分类结果和健康建议。")

    image_file = st.file_uploader("上传图像 (JPG/PNG)", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="上传的图像", use_column_width=True)

        if st.button("提交识别"):
            with st.spinner("正在识别..."):
                files = {"file": (image_file.name, image_file, image_file.type)}
                resp = requests.post(f"{API_BASE}/predict/image", files=files)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success("识别结果：")
                    st.json(result)
                    if "top_predictions" in result:
                        labels = [x["class_name"] for x in result["top_predictions"]]
                        probs = [x["probability"] for x in result["top_predictions"]]
                        st.bar_chart(pd.DataFrame({"类别": labels, "概率": probs}))
                else:
                    st.error(f"识别失败: {resp.text}")

# 健康综合评估
elif menu == "健康综合评估":
    st.header("健康综合评估服务")
    st.write("上传血糖数据和图像，获得多模态健康综合评估。")

    # 血糖数据
    glucose_file = st.file_uploader("上传血糖数据 (CSV/Excel)", type=["csv", "xlsx"], key="glucose_eval")
    # 图像
    eval_image_file = st.file_uploader("上传医学图像 (JPG/PNG)", type=["jpg", "jpeg", "png", "bmp", "tiff"], key="image_eval")

    if glucose_file and eval_image_file:
        if glucose_file.name.endswith(".csv"):
            df = pd.read_csv(glucose_file)
        else:
            df = pd.read_excel(glucose_file)
        st.dataframe(df.head())
        row_idx = st.number_input("选择血糖数据行号", min_value=0, max_value=len(df)-1, value=0, key="row_eval")
        glucose_data = df.iloc[row_idx].to_dict()
        st.json(glucose_data)

        image = Image.open(eval_image_file)
        st.image(image, caption="上传的图像", use_column_width=True)

        if st.button("提交综合评估"):
            with st.spinner("正在评估..."):
                files = {"image_file": (eval_image_file.name, eval_image_file, eval_image_file.type)}
                data = {"glucose_data": json.dumps(glucose_data)}
                resp = requests.post(f"{API_BASE}/assess/health", data=data, files=files)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success("综合评估结果：")
                    st.json(result)
                else:
                    st.error(f"评估失败: {resp.text}")

# 系统状态
elif menu == "系统状态":
    st.header("系统状态与监控")
    st.write("查看系统运行状态、性能指标和监控信息。")

    if st.button("刷新状态"):
        with st.spinner("正在获取系统状态..."):
            resp = requests.get(f"{API_BASE}/health")
            if resp.status_code == 200:
                status = resp.json()
                st.success("系统状态：")
                st.json(status)
            else:
                st.error(f"获取失败: {resp.text}")

    # 展示监控日志
    try:
        with open("logs/system_metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.subheader("系统性能历史")
        st.line_chart(pd.DataFrame(metrics)[["cpu_usage", "memory_usage", "disk_usage"]])
    except Exception:
        st.info("暂无系统监控数据。")

# API文档
elif menu == "API文档":
    st.header("API文档与交互")
    st.write("点击下方链接访问 FastAPI 自动生成的交互式API文档：")
    st.markdown(f"[API文档 (Swagger UI)]({API_BASE}/docs)")
    st.markdown(f"[健康检查接口]({API_BASE}/health)")
    st.markdown(f"[OpenAPI规范]({API_BASE}/openapi.json)")

__all__ = ["'API_BASE'", "'menu'"]

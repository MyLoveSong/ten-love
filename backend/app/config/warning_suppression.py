#!/usr/bin/env python3
"""
全局警告抑制配置
一次性解决所有INFO/WARNING噪声
"""

import os
import warnings
import logging

def suppress_all_warnings():
    """抑制所有警告和噪声"""

    # 1. 抑制Python警告
    warnings.filterwarnings('ignore')

    # 2. 抑制TensorFlow警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示ERROR
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化警告

    # 3. 抑制PyTorch警告
    try:
        import torch
        torch.set_warn_always(False)
    except ImportError:
        pass

    # 4. 抑制transformers警告
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

    # 5. 抑制PIL警告
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
    except ImportError:
        pass

    # 6. 抑制matplotlib警告
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        matplotlib.rcParams['figure.max_open_warning'] = 0
    except ImportError:
        pass

    # 7. 抑制numpy警告
    try:
        import numpy as np
        np.seterr(all='ignore')
    except ImportError:
        pass

    # 8. 抑制pandas警告
    try:
        import pandas as pd
        pd.options.mode.chained_assignment = None
    except ImportError:
        pass

    # 9. 抑制sklearn警告 - 使用更安全的方式
    try:
        warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        # 设置环境变量避免threadpoolctl问题
        os.environ['SKLEARN_THREADPOOLCTL'] = '0'
    except Exception:
        pass  # 如果sklearn不可用，跳过

    # 10. 抑制huggingface警告
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

    # 11. 抑制torchvision警告
    try:
        import torchvision
        torchvision.disable_beta_transforms_warning()
    except ImportError:
        pass

    # 12. 抑制tqdm进度条
    os.environ['TQDM_DISABLE'] = '1'

    # 13. 抑制urllib3警告
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

    # 14. 抑制requests警告
    try:
        import requests
        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    except ImportError:
        pass

    # 15. 抑制SSL警告
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    except ImportError:
        pass

    # 16. 抑制logging噪声
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('torchvision').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    logging.getLogger('tokenizers').setLevel(logging.ERROR)
    logging.getLogger('datasets').setLevel(logging.ERROR)
    logging.getLogger('accelerate').setLevel(logging.ERROR)
    logging.getLogger('safetensors').setLevel(logging.ERROR)

    # 17. 抑制特定库的INFO级别日志
    logging.getLogger('numexpr').setLevel(logging.ERROR)
    logging.getLogger('numexpr.utils').setLevel(logging.ERROR)

    print("🔇 所有警告和噪声已抑制")

def setup_clean_environment():
    """设置清洁的运行环境"""
    suppress_all_warnings()

    # 设置环境变量
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 禁用CUDA阻塞警告

    # 设置日志级别
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    print("🧹 清洁环境设置完成")

if __name__ == "__main__":
    setup_clean_environment()

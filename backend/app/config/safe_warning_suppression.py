#!/usr/bin/env python3
"""
安全的警告抑制配置
避免导入可能引起问题的模块
"""

import os
import warnings
import logging

def suppress_all_warnings():
    """抑制所有警告和噪声 - 安全版本"""

    # 1. 抑制Python警告
    warnings.filterwarnings('ignore')

    # 2. 抑制TensorFlow警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # 3. 抑制transformers警告
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

    # 4. 抑制huggingface警告
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

    # 5. 抑制tqdm进度条
    os.environ['TQDM_DISABLE'] = '1'

    # 6. 抑制sklearn相关警告
    os.environ['SKLEARN_THREADPOOLCTL'] = '0'
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

    # 7. 抑制logging噪声
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
    logging.getLogger('numexpr').setLevel(logging.ERROR)
    logging.getLogger('numexpr.utils').setLevel(logging.ERROR)

    print("🔇 所有警告和噪声已抑制")

def setup_clean_environment():
    """设置清洁的运行环境"""
    suppress_all_warnings()

    # 设置环境变量
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    # 设置日志级别
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    print("🧹 清洁环境设置完成")

if __name__ == "__main__":
    setup_clean_environment()

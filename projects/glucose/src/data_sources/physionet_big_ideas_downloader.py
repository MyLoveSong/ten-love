#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PhysioNet Big Ideas Glycemic Wearable 数据集下载和处理脚本

数据集信息:
- 名称: big-ideas-glycemic-wearable
- 版本: 1.0.0
- 大小: 34.1 GB (未压缩), 4.7 GB (ZIP)
- URL: https://physionet.org/files/big-ideas-glycemic-wearable/1.0.0/

下载方式:
1. 使用wget: wget -r -N -c -np https://physionet.org/files/big-ideas-glycemic-wearable/1.0.0/
2. 使用AWS CLI: aws s3 sync --no-sign-request s3://physionet-open/big-ideas-glycemic-wearable/1.0.0/ DESTINATION
3. 使用Python脚本 (本文件)
"""

import os
import sys
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import zipfile
import tempfile
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    logger.warning("chardet未安装，编码检测功能受限。安装: pip install chardet")

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from ..utils.large_file_splitter import split_large_csv_files

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhysioNetBigIdeasDownloader:
    """PhysioNet Big Ideas数据集下载器"""

    def __init__(
        self,
        output_dir: str = "TRAIN/data/physionet_big_ideas",
        use_cache: bool = True,
        timeout: Optional[int] = 3600,
        split_large_files: bool = False,
        split_threshold_mb: int = 200,
        split_chunk_rows: int = 200_000,
        rename_original_after_split: bool = True,
    ):
        """
        初始化下载器

        Args:
            output_dir: 输出目录
            use_cache: 是否使用缓存
            timeout: 子进程超时时间(秒), <=0 表示不限
            split_large_files: 是否在处理前拆分超大CSV
            split_threshold_mb: 拆分阈值(MB)
            split_chunk_rows: 单个拆分文件的行数
            rename_original_after_split: 拆分成功后是否重命名原始文件
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        # timeout为None表示不限制时间
        if timeout is not None and timeout <= 0:
            timeout = None
        self.command_timeout = timeout
        self.split_large_files = split_large_files
        self.split_threshold_mb = split_threshold_mb
        self.split_chunk_rows = split_chunk_rows
        self.rename_original_after_split = rename_original_after_split

        # PhysioNet数据集信息
        self.dataset_name = "big-ideas-glycemic-wearable"
        self.dataset_version = "1.0.0"
        self.base_url = f"https://physionet.org/files/{self.dataset_name}/{self.dataset_version}/"
        self.s3_url = f"s3://physionet-open/{self.dataset_name}/{self.dataset_version}/"

        # 缓存文件
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.processed_data_path = self.output_dir / "processed_data.json"
        self.metadata_path = self.output_dir / "metadata.json"

        logger.info(f"初始化PhysioNet Big Ideas下载器")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"数据集: {self.dataset_name} v{self.dataset_version}")

    def check_dependencies(self) -> Dict[str, bool]:
        """检查必要的依赖工具"""
        dependencies = {
            'wget': False,
            'aws_cli': False,
            'python_requests': True  # Python requests总是可用
        }

        # 检查wget
        try:
            result = subprocess.run(
                ['wget', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                dependencies['wget'] = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # 检查AWS CLI
        try:
            result = subprocess.run(
                ['aws', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                dependencies['aws_cli'] = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return dependencies

    def download_with_wget(self, destination: Optional[Path] = None) -> bool:
        """
        使用wget下载数据集

        Args:
            destination: 目标目录，如果为None则使用self.output_dir

        Returns:
            是否成功
        """
        if destination is None:
            destination = self.output_dir / "raw_data"
        destination.mkdir(parents=True, exist_ok=True)

        logger.info("使用wget下载数据集...")
        logger.info(f"目标目录: {destination}")

        try:
            # wget命令: 递归下载，支持断点续传，不下载父目录
            cmd = [
                'wget',
                '-r',           # 递归下载
                '-N',           # 只下载新文件
                '-c',           # 支持断点续传
                '-np',          # 不下载父目录
                '--no-parent',  # 不下载父目录
                '--cut-dirs=3', # 跳过URL中的3级目录
                '-P', str(destination),  # 输出目录
                self.base_url
            ]

            logger.info(f"执行命令: {' '.join(cmd)}")
            # 直接将wget的stdout/stderr输出到终端，方便用户实时查看下载进度
            run_kwargs: Dict[str, Any] = {}
            if self.command_timeout is not None:
                run_kwargs['timeout'] = self.command_timeout
            result = subprocess.run(
                cmd,
                **run_kwargs
            )

            if result.returncode == 0:
                logger.info("✅ wget下载成功")
                return True
            else:
                logger.error(f"wget下载失败，返回码: {result.returncode}")
                return False

        except FileNotFoundError:
            logger.error("wget未安装，请先安装wget")
            return False
        except subprocess.TimeoutExpired:
            logger.error("下载超时")
            return False
        except Exception as e:
            logger.error(f"wget下载出错: {e}")
            return False

    def download_with_aws_cli(self, destination: Optional[Path] = None) -> bool:
        """
        使用AWS CLI下载数据集

        Args:
            destination: 目标目录，如果为None则使用self.output_dir

        Returns:
            是否成功
        """
        if destination is None:
            destination = self.output_dir / "raw_data"
        destination.mkdir(parents=True, exist_ok=True)

        logger.info("使用AWS CLI下载数据集...")
        logger.info(f"目标目录: {destination}")

        try:
            # AWS S3 sync命令
            cmd = [
                'aws',
                's3',
                'sync',
                '--no-sign-request',  # 不需要签名
                self.s3_url,
                str(destination)
            ]

            logger.info(f"执行命令: {' '.join(cmd)}")
            # 直接将AWS CLI的stdout/stderr输出到终端，方便用户实时查看下载进度
            run_kwargs: Dict[str, Any] = {}
            if self.command_timeout is not None:
                run_kwargs['timeout'] = self.command_timeout
            result = subprocess.run(
                cmd,
                **run_kwargs
            )

            if result.returncode == 0:
                logger.info("✅ AWS CLI下载成功")
                return True
            else:
                logger.error(f"AWS CLI下载失败，返回码: {result.returncode}")
                return False

        except FileNotFoundError:
            logger.error("AWS CLI未安装，请先安装AWS CLI")
            return False
        except subprocess.TimeoutExpired:
            logger.error("下载超时")
            return False
        except Exception as e:
            logger.error(f"AWS CLI下载出错: {e}")
            return False

    def download_with_python(self, destination: Optional[Path] = None) -> bool:
        """
        使用Python requests下载数据集（适用于小文件或特定文件）

        注意: 对于大文件，建议使用wget或AWS CLI

        Args:
            destination: 目标目录

        Returns:
            是否成功
        """
        if destination is None:
            destination = self.output_dir / "raw_data"
        destination.mkdir(parents=True, exist_ok=True)

        logger.info("使用Python requests下载数据集...")
        logger.warning("注意: 对于大文件(34GB)，建议使用wget或AWS CLI")

        # 尝试下载文件列表
        try:
            # 获取文件列表（如果可用）
            list_url = self.base_url.rstrip('/') + '/'
            response = requests.get(list_url, timeout=30)

            if response.status_code == 200:
                # 解析HTML获取文件列表
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                except ImportError:
                    logger.warning("BeautifulSoup4未安装，无法解析HTML文件列表")
                    logger.info("请安装: pip install beautifulsoup4")
                    return False
                links = soup.find_all('a', href=True)

                files_to_download = []
                for link in links:
                    href = link['href']
                    if href.endswith(('.csv', '.json', '.txt', '.zip')):
                        files_to_download.append(href)

                logger.info(f"找到 {len(files_to_download)} 个文件")

                # 下载文件（仅下载前几个作为示例）
                for i, filename in enumerate(files_to_download[:5]):  # 限制下载数量
                    file_url = self.base_url + filename
                    file_path = destination / filename

                    if file_path.exists() and self.use_cache:
                        logger.info(f"跳过已存在的文件: {filename}")
                        continue

                    logger.info(f"下载文件 {i+1}/{min(5, len(files_to_download))}: {filename}")
                    try:
                        file_response = requests.get(file_url, stream=True, timeout=60)
                        file_response.raise_for_status()

                        with open(file_path, 'wb') as f:
                            for chunk in file_response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        logger.info(f"✅ 下载完成: {filename}")
                    except Exception as e:
                        logger.warning(f"下载 {filename} 失败: {e}")

                return True
            else:
                logger.warning(f"无法获取文件列表: HTTP {response.status_code}")
                return False

        except ImportError:
            logger.warning("BeautifulSoup未安装，无法解析HTML")
            return False
        except Exception as e:
            logger.error(f"Python下载出错: {e}")
            return False

    def download(self, method: str = 'auto', destination: Optional[Path] = None) -> bool:
        """
        下载数据集（自动选择最佳方法）

        Args:
            method: 下载方法 ('auto', 'wget', 'aws', 'python')
            destination: 目标目录

        Returns:
            是否成功
        """
        if destination is None:
            destination = self.output_dir / "raw_data"

        # 检查依赖
        dependencies = self.check_dependencies()

        # 自动选择方法
        if method == 'auto':
            if dependencies['wget']:
                method = 'wget'
            elif dependencies['aws_cli']:
                method = 'aws'
            else:
                method = 'python'
                logger.warning("未找到wget或AWS CLI，使用Python下载（可能较慢）")

        logger.info(f"使用 {method} 方法下载数据集")

        # 执行下载
        if method == 'wget':
            return self.download_with_wget(destination)
        elif method == 'aws':
            return self.download_with_aws_cli(destination)
        elif method == 'python':
            return self.download_with_python(destination)
        else:
            logger.error(f"未知的下载方法: {method}")
            return False

    def _split_large_files_if_needed(self, data_dir: Path) -> None:
        """根据配置拆分大文件"""
        if not self.split_large_files:
            return
        logger.info("检查并拆分超大CSV文件(阈值: %d MB)", self.split_threshold_mb)
        split_large_csv_files(
            target_dir=data_dir,
            size_threshold_mb=self.split_threshold_mb,
            chunk_rows=self.split_chunk_rows,
            rename_original=self.rename_original_after_split,
        )

    def find_data_files(self, data_dir: Path) -> List[Path]:
        """查找数据文件"""
        data_files = []

        # 查找常见的血糖数据文件
        patterns = ['*.csv', '*.json', '*.txt', '*.tsv']
        for pattern in patterns:
            data_files.extend(data_dir.rglob(pattern))

        # 过滤掉一些明显不是数据文件的文件
        excluded = ['README', 'LICENSE', 'CHANGELOG', '.git']
        data_files = [
            f for f in data_files
            if not any(ex in str(f).upper() for ex in excluded)
        ]

        return sorted(data_files)

    def detect_encoding(self, file_path: Path) -> str:
        """
        检测文件编码

        Args:
            file_path: 文件路径

        Returns:
            检测到的编码
        """
        if not HAS_CHARDET:
            # 如果没有chardet，尝试常见编码
            for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk']:
                try:
                    with open(file_path, 'r', encoding=enc) as test_f:
                        test_f.read(1000)
                    logger.info(f"使用编码: {enc}")
                    return enc
                except:
                    continue
            return 'utf-8'

        # 尝试读取文件的前几KB来检测编码
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # 读取前10KB
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)

                if confidence < 0.7:
                    # 如果置信度低，尝试常见编码
                    for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk']:
                        try:
                            with open(file_path, 'r', encoding=enc) as test_f:
                                test_f.read(1000)
                            logger.info(f"使用编码: {enc} (检测到: {encoding}, 置信度: {confidence:.2f})")
                            return enc
                        except:
                            continue

                logger.info(f"检测到编码: {encoding} (置信度: {confidence:.2f})")
                return encoding
        except Exception as e:
            logger.warning(f"编码检测失败: {e}，使用utf-8")
            return 'utf-8'

    def parse_glucose_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        解析血糖数据文件（支持多种编码）

        Args:
            file_path: 数据文件路径

        Returns:
            解析后的DataFrame或None
        """
        # 尝试多种编码
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk', 'utf-16']

        for encoding in encodings:
            try:
                # 根据文件扩展名选择解析方法
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                elif file_path.suffix == '.json':
                    df = pd.read_json(file_path, encoding=encoding)
                elif file_path.suffix in ['.txt', '.tsv']:
                    df = pd.read_csv(file_path, sep='\t', encoding=encoding, low_memory=False)
                else:
                    logger.warning(f"不支持的文件格式: {file_path.suffix}")
                    return None

                # 标准化列名
                df = self._standardize_columns(df)

                logger.info(f"✅ 成功解析文件 {file_path.name} (编码: {encoding})")
                return df

            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"解析文件 {file_path} 失败 (编码: {encoding}): {e}")
                continue

        # 如果所有编码都失败，尝试自动检测
        try:
            detected_encoding = self.detect_encoding(file_path)
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, encoding=detected_encoding, low_memory=False)
            elif file_path.suffix == '.json':
                df = pd.read_json(file_path, encoding=detected_encoding)
            elif file_path.suffix in ['.txt', '.tsv']:
                df = pd.read_csv(file_path, sep='\t', encoding=detected_encoding, low_memory=False)
            else:
                return None

            df = self._standardize_columns(df)
            logger.info(f"✅ 成功解析文件 {file_path.name} (自动检测编码: {detected_encoding})")
            return df

        except Exception as e:
            logger.error(f"解析文件 {file_path} 完全失败: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        # 常见的列名映射
        column_mapping = {
            # 患者ID
            'patient_id': ['patient_id', 'subject_id', 'id', 'user_id', 'participant_id', 'subject'],
            # 时间戳
            'timestamp': ['timestamp', 'time', 'datetime', 'date_time', 'ts', 'date', 'time_stamp'],
            # 血糖值 (mg/dL)
            'glucose_mg_dl': ['glucose', 'bg', 'blood_glucose', 'glucose_mg_dl', 'glucose_value',
                             'cgm', 'cgm_value', 'glucose_mgdl'],
            # 血糖值 (mmol/L)
            'glucose_mmol_l': ['glucose_mmol_l', 'glucose_mmol', 'bg_mmol', 'glucose_mmolL']
        }

        standardized = pd.DataFrame()

        # 映射列名
        for std_col, possible_cols in column_mapping.items():
            for col in possible_cols:
                if col.lower() in [c.lower() for c in df.columns]:
                    # 找到匹配的列（不区分大小写）
                    matched_col = [c for c in df.columns if c.lower() == col.lower()][0]
                    standardized[std_col] = df[matched_col]
                    break

        # 如果没有找到患者ID，生成一个
        if 'patient_id' not in standardized.columns:
            standardized['patient_id'] = [f'patient_{i:06d}' for i in range(len(df))]

        # 如果没有找到时间戳，尝试从索引或生成
        if 'timestamp' not in standardized.columns:
            if df.index.name and 'time' in df.index.name.lower():
                standardized['timestamp'] = df.index
            else:
                # 生成默认时间戳
                base_time = datetime.now() - pd.Timedelta(days=30)
                standardized['timestamp'] = [
                    base_time + pd.Timedelta(minutes=i*5)
                    for i in range(len(df))
                ]

        # 单位转换
        if 'glucose_mg_dl' in standardized.columns and 'glucose_mmol_l' not in standardized.columns:
            standardized['glucose_mmol_l'] = standardized['glucose_mg_dl'] / 18.0
        elif 'glucose_mmol_l' in standardized.columns and 'glucose_mg_dl' not in standardized.columns:
            standardized['glucose_mg_dl'] = standardized['glucose_mmol_l'] * 18.0

        # 添加数据源标识
        standardized['data_source'] = 'physionet_big_ideas'

        return standardized

    def process_downloaded_data(self, data_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        处理下载的数据

        Args:
            data_dir: 数据目录，如果为None则使用self.output_dir / "raw_data"

        Returns:
            处理后的DataFrame
        """
        if data_dir is None:
            data_dir = self.output_dir / "raw_data"

        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        logger.info(f"处理数据目录: {data_dir}")

        # 处理前尝试拆分超大文件，避免解析失败
        self._split_large_files_if_needed(data_dir)

        # 查找数据文件
        data_files = self.find_data_files(data_dir)
        logger.info(f"找到 {len(data_files)} 个数据文件")

        if not data_files:
            logger.warning("未找到数据文件")
            return pd.DataFrame()

        # 解析所有文件
        all_dataframes = []
        for file_path in tqdm(data_files, desc="处理文件"):
            df = self.parse_glucose_data(file_path)
            if df is not None and not df.empty:
                all_dataframes.append(df)
                logger.debug(f"处理完成: {file_path.name} ({len(df)} 行)")

        if not all_dataframes:
            logger.warning("没有成功解析的数据")
            return pd.DataFrame()

        # 合并所有数据
        logger.info("合并所有数据...")
        merged_df = pd.concat(all_dataframes, ignore_index=True)

        # 数据清洗
        logger.info("清洗数据...")
        merged_df = self._clean_data(merged_df)

        # 保存处理后的数据
        logger.info(f"保存处理后的数据: {self.processed_data_path}")
        merged_df.to_json(self.processed_data_path, orient='records', lines=False, indent=2)

        # 保存元数据
        metadata = {
            'dataset_name': self.dataset_name,
            'dataset_version': self.dataset_version,
            'processing_time': datetime.now().isoformat(),
            'total_records': len(merged_df),
            'unique_patients': merged_df['patient_id'].nunique() if 'patient_id' in merged_df.columns else 0,
            'date_range': {
                'start': str(merged_df['timestamp'].min()) if 'timestamp' in merged_df.columns else None,
                'end': str(merged_df['timestamp'].max()) if 'timestamp' in merged_df.columns else None
            },
            'glucose_stats': {
                'mean_mg_dl': float(merged_df['glucose_mg_dl'].mean()) if 'glucose_mg_dl' in merged_df.columns else None,
                'std_mg_dl': float(merged_df['glucose_mg_dl'].std()) if 'glucose_mg_dl' in merged_df.columns else None,
                'min_mg_dl': float(merged_df['glucose_mg_dl'].min()) if 'glucose_mg_dl' in merged_df.columns else None,
                'max_mg_dl': float(merged_df['glucose_mg_dl'].max()) if 'glucose_mg_dl' in merged_df.columns else None
            }
        }

        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 数据处理完成: {len(merged_df)} 条记录")
        logger.info(f"元数据已保存: {self.metadata_path}")

        return merged_df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        original_len = len(df)

        # 移除重复行
        df = df.drop_duplicates()

        # 移除缺失值过多的行
        if 'glucose_mg_dl' in df.columns:
            df = df.dropna(subset=['glucose_mg_dl'])

        # 过滤异常值
        if 'glucose_mg_dl' in df.columns:
            # 正常血糖范围: 40-400 mg/dL
            df = df[(df['glucose_mg_dl'] >= 40) & (df['glucose_mg_dl'] <= 400)]

        # 转换时间戳
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])

        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"清洗后移除 {removed} 条记录 ({removed/original_len*100:.1f}%)")

        return df

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """获取处理后的数据"""
        if not self.processed_data_path.exists():
            logger.warning("处理后的数据不存在，请先运行process_downloaded_data()")
            return None

        logger.info(f"加载处理后的数据: {self.processed_data_path}")
        df = pd.read_json(self.processed_data_path)

        # 转换时间戳
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="PhysioNet Big Ideas数据集下载和处理工具"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='TRAIN/data/physionet_big_ideas',
        help='输出目录'
    )
    parser.add_argument(
        '--method',
        choices=['auto', 'wget', 'aws', 'python'],
        default='auto',
        help='下载方法'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='仅下载，不处理'
    )
    parser.add_argument(
        '--process-only',
        action='store_true',
        help='仅处理已下载的数据'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='数据目录（用于process-only模式）'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='子进程超时时间(秒)，0表示不限制'
    )
    parser.add_argument(
        '--split-large-files',
        action='store_true',
        help='处理前拆分超大CSV文件'
    )
    parser.add_argument(
        '--split-threshold-mb',
        type=int,
        default=200,
        help='触发拆分的文件大小阈值(MB)'
    )
    parser.add_argument(
        '--split-chunk-rows',
        type=int,
        default=200_000,
        help='拆分时每个子文件的行数'
    )
    parser.add_argument(
        '--keep-original-files',
        action='store_true',
        help='拆分后保留原始大文件(默认加.bak后缀)'
    )

    args = parser.parse_args()

    # 创建下载器
    downloader = PhysioNetBigIdeasDownloader(
        output_dir=args.output_dir,
        timeout=args.timeout,
        split_large_files=args.split_large_files,
        split_threshold_mb=args.split_threshold_mb,
        split_chunk_rows=args.split_chunk_rows,
        rename_original_after_split=not args.keep_original_files
    )

    # 检查依赖
    dependencies = downloader.check_dependencies()
    logger.info("依赖检查:")
    for dep, available in dependencies.items():
        status = "✅" if available else "❌"
        logger.info(f"  {status} {dep}")

    # 下载
    if not args.process_only:
        logger.info("=" * 50)
        logger.info("开始下载数据集")
        logger.info("=" * 50)
        success = downloader.download(method=args.method)
        if not success:
            logger.error("下载失败")
            return

    # 处理
    if not args.download_only:
        logger.info("=" * 50)
        logger.info("开始处理数据")
        logger.info("=" * 50)
        data_dir = Path(args.data_dir) if args.data_dir else None
        try:
            df = downloader.process_downloaded_data(data_dir)
            if df is not None and not df.empty:
                logger.info("=" * 50)
                logger.info("数据处理完成")
                logger.info("=" * 50)
                logger.info(f"总记录数: {len(df)}")
                logger.info(f"唯一患者数: {df['patient_id'].nunique() if 'patient_id' in df.columns else 'N/A'}")
                if 'glucose_mg_dl' in df.columns:
                    logger.info(f"血糖范围: {df['glucose_mg_dl'].min():.1f} - {df['glucose_mg_dl'].max():.1f} mg/dL")
                    logger.info(f"平均血糖: {df['glucose_mg_dl'].mean():.1f} mg/dL")
        except Exception as e:
            logger.error(f"处理数据失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PhysioNet CGM Macros 数据集下载和处理脚本

数据集信息:
- 名称: cgmacros
- 版本: 1.0.0
- 大小: 627.9 MB (未压缩), 627.1 MB (ZIP)
- URL: https://physionet.org/files/cgmacros/1.0.0/

下载方式:
1. 使用wget: wget -r -N -c -np https://physionet.org/files/cgmacros/1.0.0/
2. 使用AWS CLI: aws s3 sync --no-sign-request s3://physionet-open/cgmacros/1.0.0/ DESTINATION
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


class PhysioNetCGMacrosDownloader:
    """PhysioNet CGM Macros数据集下载器"""

    def __init__(
        self,
        output_dir: str = "TRAIN/data/physionet_cgmacros",
        use_cache: bool = True,
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
            split_large_files: 是否拆分超大CSV
            split_threshold_mb: 拆分阈值(MB)
            split_chunk_rows: 拆分行数
            rename_original_after_split: 拆分后是否保留原始文件
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.split_large_files = split_large_files
        self.split_threshold_mb = split_threshold_mb
        self.split_chunk_rows = split_chunk_rows
        self.rename_original_after_split = rename_original_after_split

        # PhysioNet数据集信息
        self.dataset_name = "cgmacros"
        self.dataset_version = "1.0.0"
        self.base_url = f"https://physionet.org/files/{self.dataset_name}/{self.dataset_version}/"
        self.s3_url = f"s3://physionet-open/{self.dataset_name}/{self.dataset_version}/"

        # 缓存文件
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.processed_data_path = self.output_dir / "processed_data.json"
        self.metadata_path = self.output_dir / "metadata.json"

        logger.info(f"初始化PhysioNet CGM Macros下载器")
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

            # 直接将wget的stdout/stderr输出到终端，方便用户查看下载进度
            result = subprocess.run(
                cmd,
                timeout=3600  # 1小时超时
            )

            if result.returncode == 0:
                logger.info("✅ wget下载完成")
                return True
            else:
                logger.error(f"wget下载失败，返回码: {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("wget下载超时")
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
                '--no-sign-request',
                self.s3_url,
                str(destination)
            ]

            # 直接将AWS CLI的stdout/stderr输出到终端，方便用户查看下载进度
            result = subprocess.run(
                cmd,
                timeout=3600  # 1小时超时
            )

            if result.returncode == 0:
                logger.info("✅ AWS CLI下载完成")
                return True
            else:
                logger.error(f"AWS CLI下载失败，返回码: {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("AWS CLI下载超时")
            return False
        except Exception as e:
            logger.error(f"AWS CLI下载出错: {e}")
            return False

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

    def download_with_python(self, destination: Optional[Path] = None) -> bool:
        """
        使用Python requests下载数据集

        Args:
            destination: 目标目录，如果为None则使用self.output_dir

        Returns:
            是否成功
        """
        if destination is None:
            destination = self.output_dir / "raw_data"
        destination.mkdir(parents=True, exist_ok=True)

        logger.info("使用Python requests下载数据集...")
        logger.warning("注意: 对于大文件(627MB)，建议使用wget或AWS CLI")

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

                # 查找所有文件链接
                links = soup.find_all('a', href=True)
                files = []
                for link in links:
                    href = link.get('href')
                    if href and not href.startswith('../') and not href.endswith('/'):
                        files.append(href)

                if not files:
                    logger.warning("未找到文件列表，尝试直接下载ZIP文件")
                    # 尝试下载ZIP文件
                    zip_url = self.base_url.rstrip('/') + '.zip'
                    zip_path = destination / f"{self.dataset_name}.zip"
                    return self._download_file(zip_url, zip_path)

                logger.info(f"找到 {len(files)} 个文件")

                # 下载每个文件
                for i, filename in enumerate(files, 1):
                    file_url = self.base_url + filename
                    file_path = destination / filename

                    logger.info(f"下载文件 {i}/{len(files)}: {filename}")

                    if self._download_file(file_url, file_path):
                        logger.info(f"✅ 下载完成: {filename}")
                    else:
                        logger.warning(f"❌ 下载失败: {filename}")

                return True
            else:
                logger.error(f"无法访问文件列表: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Python下载失败: {e}")
            return False

    def _download_file(self, url: str, file_path: Path) -> bool:
        """下载单个文件"""
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(file_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"下载 {file_path.name}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

            return True
        except Exception as e:
            logger.error(f"下载文件失败 {url}: {e}")
            return False

    def download(self, method: str = 'auto', destination: Optional[Path] = None) -> bool:
        """
        下载数据集

        Args:
            method: 下载方法 ('auto', 'wget', 'aws', 'python')
            destination: 目标目录

        Returns:
            是否成功
        """
        if method == 'auto':
            # 自动选择最佳方法
            dependencies = self.check_dependencies()

            if dependencies['wget']:
                logger.info("使用 wget 方法下载数据集")
                return self.download_with_wget(destination)
            elif dependencies['aws_cli']:
                logger.info("使用 aws 方法下载数据集")
                return self.download_with_aws_cli(destination)
            else:
                logger.warning("未找到wget或AWS CLI，使用Python下载（可能较慢）")
                logger.info("使用 python 方法下载数据集")
                return self.download_with_python(destination)

        elif method == 'wget':
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

        # 首先检查是否有ZIP文件需要解压
        zip_files = list(data_dir.rglob('*.zip'))
        for zip_file in zip_files:
            if 'CGMacros' in zip_file.name or 'cgmacros' in zip_file.name.lower():
                logger.info(f"发现ZIP文件: {zip_file}")
                extracted_dir = zip_file.parent / zip_file.stem

                # 检查ZIP文件是否完整
                try:
                    import zipfile
                    with zipfile.ZipFile(zip_file, 'r') as z:
                        # 测试ZIP文件是否有效
                        bad_file = z.testzip()
                        if bad_file:
                            logger.error(f"ZIP文件损坏: {bad_file}")
                            logger.info("尝试重新下载ZIP文件...")
                            continue
                except zipfile.BadZipFile:
                    logger.error(f"ZIP文件格式错误: {zip_file}")
                    logger.info("ZIP文件可能未完全下载，请重新下载")
                    continue
                except Exception as e:
                    logger.error(f"检查ZIP文件失败: {e}")
                    continue

                if not extracted_dir.exists():
                    logger.info(f"解压ZIP文件到: {extracted_dir}")
                    try:
                        import zipfile
                        with zipfile.ZipFile(zip_file, 'r') as z:
                            # 显示解压进度
                            file_list = z.namelist()
                            logger.info(f"ZIP文件包含 {len(file_list)} 个文件")
                            z.extractall(extracted_dir)
                        logger.info(f"✅ ZIP文件解压完成")
                    except zipfile.BadZipFile:
                        logger.error(f"ZIP文件损坏或未完全下载: {zip_file}")
                        logger.info("请重新下载ZIP文件")
                        continue
                    except Exception as e:
                        logger.error(f"解压ZIP文件失败: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    logger.info(f"ZIP文件已解压到: {extracted_dir}")

                # 在解压后的目录中查找文件
                data_dir = extracted_dir

        # 查找常见的血糖数据文件
        patterns = ['*.csv', '*.json', '*.txt', '*.tsv']
        for pattern in patterns:
            data_files.extend(data_dir.rglob(pattern))

        # 过滤掉一些明显不是数据文件的文件
        excluded = ['README', 'LICENSE', 'CHANGELOG', '.git', 'SHA256', 'Dictionary', 'index', 'robots']
        data_files = [
            f for f in data_files
            if not any(ex in str(f).upper() for ex in excluded)
        ]

        return sorted(data_files)

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
        # 常见的列名映射（包括CGMacros特定的列名）
        column_mapping = {
            # 患者ID
            'patient_id': ['patient_id', 'subject_id', 'id', 'user_id', 'participant_id', 'subject'],
            # 时间戳
            'timestamp': ['timestamp', 'time', 'datetime', 'date_time', 'ts', 'date', 'time_stamp', 'Timestamp'],
            # 血糖值 (mg/dL) - CGMacros使用Libre GL和Dexcom GL
            'glucose_mg_dl': ['glucose', 'bg', 'blood_glucose', 'glucose_mg_dl', 'glucose_value',
                             'cgm', 'cgm_value', 'glucose_mgdl', 'Libre GL', 'Dexcom GL',
                             'libre_gl', 'dexcom_gl', 'Libre_GL', 'Dexcom_GL'],
            # 血糖值 (mmol/L)
            'glucose_mmol_l': ['glucose_mmol_l', 'glucose_mmol', 'bg_mmol', 'glucose_mmolL']
        }

        standardized = df.copy()

        # 映射列名（不区分大小写）
        for std_col, possible_cols in column_mapping.items():
            for col in df.columns:
                if col.lower() in [c.lower() for c in possible_cols]:
                    # 找到匹配的列（不区分大小写）
                    if std_col not in standardized.columns or standardized[std_col].isna().all():
                        standardized[std_col] = df[col]
                    break

        # CGMacros特殊处理：优先使用Libre GL，如果没有则使用Dexcom GL
        if 'glucose_mg_dl' not in standardized.columns:
            if 'Libre GL' in df.columns:
                standardized['glucose_mg_dl'] = pd.to_numeric(df['Libre GL'], errors='coerce')
            elif 'Dexcom GL' in df.columns:
                standardized['glucose_mg_dl'] = pd.to_numeric(df['Dexcom GL'], errors='coerce')
            elif 'libre_gl' in df.columns:
                standardized['glucose_mg_dl'] = pd.to_numeric(df['libre_gl'], errors='coerce')
            elif 'dexcom_gl' in df.columns:
                standardized['glucose_mg_dl'] = pd.to_numeric(df['dexcom_gl'], errors='coerce')

        # 如果没有找到患者ID，尝试从文件名或生成
        if 'patient_id' not in standardized.columns:
            # 尝试从文件名提取患者ID（如果文件名包含患者信息）
            standardized['patient_id'] = [f'cgmacros_patient_{i:06d}' for i in range(len(df))]

        # 如果没有找到时间戳，尝试从Timestamp列
        if 'timestamp' not in standardized.columns:
            if 'Timestamp' in df.columns:
                standardized['timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            elif 'timestamp' in df.columns:
                standardized['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
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
        standardized['data_source'] = 'physionet_cgmacros'

        # 只保留必要的列
        keep_cols = ['patient_id', 'timestamp', 'glucose_mg_dl', 'glucose_mmol_l', 'data_source']
        # 保留所有原始列，但确保标准列存在
        for col in keep_cols:
            if col not in standardized.columns:
                if col == 'glucose_mmol_l' and 'glucose_mg_dl' in standardized.columns:
                    standardized[col] = standardized['glucose_mg_dl'] / 18.0
                elif col == 'glucose_mg_dl' and 'glucose_mmol_l' in standardized.columns:
                    standardized[col] = standardized['glucose_mmol_l'] * 18.0

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

        self._split_large_files_if_needed(data_dir)

        # 查找所有数据文件（会自动解压ZIP文件）
        data_files = self.find_data_files(data_dir)
        logger.info(f"找到 {len(data_files)} 个数据文件")

        if not data_files:
            logger.warning("未找到数据文件")
            return pd.DataFrame()

        # 解析所有文件
        all_data = []
        for file_path in tqdm(data_files, desc="处理文件"):
            df = self.parse_glucose_data(file_path)
            if df is not None and not df.empty:
                # 检查是否包含血糖值
                if 'glucose_mg_dl' in df.columns:
                    # 移除血糖值为空的记录
                    df = df[df['glucose_mg_dl'].notna()]
                    if not df.empty:
                        all_data.append(df)
                        logger.debug(f"成功解析 {file_path.name}: {len(df)} 条记录，包含血糖值")
                    else:
                        logger.warning(f"文件 {file_path.name} 解析后没有有效的血糖值")
                else:
                    logger.warning(f"文件 {file_path.name} 不包含血糖值列")

        if not all_data:
            logger.warning("没有成功解析的数据（包含血糖值）")
            return pd.DataFrame()

        # 合并所有数据
        logger.info("合并所有数据...")
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"合并后总记录数: {len(combined_df)}")

        # 数据清洗
        logger.info("清洗数据...")
        cleaned_df = self._clean_data(combined_df)
        logger.info(f"清洗后记录数: {len(cleaned_df)}")

        # 验证数据包含血糖值
        if 'glucose_mg_dl' not in cleaned_df.columns or cleaned_df['glucose_mg_dl'].isna().all():
            raise ValueError("处理后的数据不包含有效的血糖值！请检查原始数据文件。")

        # 保存处理后的数据（确保包含血糖值）
        logger.info(f"保存处理后的数据: {self.processed_data_path}")
        # 只保存必要的列，减少文件大小
        save_cols = ['patient_id', 'timestamp', 'glucose_mg_dl']
        if 'glucose_mmol_l' in cleaned_df.columns:
            save_cols.append('glucose_mmol_l')
        if 'data_source' in cleaned_df.columns:
            save_cols.append('data_source')

        save_df = cleaned_df[save_cols].copy()

        # 转换时间戳为字符串
        if 'timestamp' in save_df.columns:
            save_df['timestamp'] = save_df['timestamp'].astype(str)

        save_df.to_json(self.processed_data_path, orient='records', date_format='iso', indent=2)

        # 保存元数据
        metadata = self._generate_metadata(cleaned_df)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 数据处理完成: {len(cleaned_df)} 条记录")
        logger.info(f"  - 包含血糖值: {cleaned_df['glucose_mg_dl'].notna().sum()} 条")
        logger.info(f"  - 唯一患者数: {cleaned_df['patient_id'].nunique() if 'patient_id' in cleaned_df.columns else 0}")
        logger.info(f"  - 血糖值范围: {cleaned_df['glucose_mg_dl'].min():.1f} - {cleaned_df['glucose_mg_dl'].max():.1f} mg/dL")
        logger.info(f"元数据已保存: {self.metadata_path}")

        return cleaned_df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        # 移除重复记录
        df = df.drop_duplicates()

        # 移除无效的血糖值
        if 'glucose_mg_dl' in df.columns:
            # 血糖值通常在40-400 mg/dL之间
            df = df[(df['glucose_mg_dl'] >= 40) & (df['glucose_mg_dl'] <= 400)]

        # 确保时间戳是datetime类型
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])

        # 排序
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        return df.reset_index(drop=True)

    def _generate_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成元数据"""
        metadata = {
            'dataset_name': self.dataset_name,
            'dataset_version': self.dataset_version,
            'processing_time': datetime.now().isoformat(),
            'total_records': len(df),
            'unique_patients': df['patient_id'].nunique() if 'patient_id' in df.columns else 0,
        }

        if 'timestamp' in df.columns:
            metadata['date_range'] = {
                'start': df['timestamp'].min().isoformat() if not df.empty else None,
                'end': df['timestamp'].max().isoformat() if not df.empty else None
            }

        if 'glucose_mg_dl' in df.columns:
            metadata['glucose_stats'] = {
                'mean_mg_dl': float(df['glucose_mg_dl'].mean()) if not df.empty else None,
                'std_mg_dl': float(df['glucose_mg_dl'].std()) if not df.empty else None,
                'min_mg_dl': float(df['glucose_mg_dl'].min()) if not df.empty else None,
                'max_mg_dl': float(df['glucose_mg_dl'].max()) if not df.empty else None
            }

        return metadata

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """获取处理后的数据"""
        if not self.processed_data_path.exists():
            return None

        df = pd.read_json(self.processed_data_path)

        # 转换时间戳
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="PhysioNet CGM Macros数据集下载和处理工具"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='TRAIN/data/physionet_cgmacros',
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
    downloader = PhysioNetCGMacrosDownloader(
        output_dir=args.output_dir,
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

        data_dir = None
        if args.data_dir:
            data_dir = Path(args.data_dir)

        df = downloader.process_downloaded_data(data_dir)

        logger.info("=" * 50)
        logger.info("数据处理完成")
        logger.info("=" * 50)
        logger.info(f"总记录数: {len(df)}")
        logger.info(f"唯一患者数: {df['patient_id'].nunique() if 'patient_id' in df.columns else 0}")


if __name__ == '__main__':
    main()

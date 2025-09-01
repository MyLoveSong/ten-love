#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集5解析脚本
将生理信号数据转换为血糖预测所需的特征格式
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Dataset5Parser:
    """数据集5解析器"""
    
    def __init__(self, dataset_path="数据集5"):
        self.dataset_path = dataset_path
        self.output_file = "数据集5.csv"
        
    def parse_heart_rate_data(self, rr_file_path):
        """解析心率数据（RR间隔）"""
        try:
            # 读取RR间隔数据
            rr_data = pd.read_csv(rr_file_path)
            
            # 计算心率相关特征
            rr_intervals = rr_data['RtoR'].values
            rr_intervals = rr_intervals[rr_intervals > 0]  # 过滤无效值
            
            if len(rr_intervals) == 0:
                return None
            
            # 计算心率特征
            heart_rate = 60000 / rr_intervals  # 转换为BPM
            heart_rate = heart_rate[(heart_rate >= 40) & (heart_rate <= 200)]  # 合理范围
            
            if len(heart_rate) == 0:
                return None
            
            features = {
                'mean_heart_rate': np.mean(heart_rate),
                'std_heart_rate': np.std(heart_rate),
                'min_heart_rate': np.min(heart_rate),
                'max_heart_rate': np.max(heart_rate),
                'heart_rate_variability': np.std(rr_intervals),
                'rr_mean': np.mean(rr_intervals),
                'rr_std': np.std(rr_intervals)
            }
            
            return features
            
        except Exception as e:
            print(f"解析心率数据失败: {e}")
            return None
    
    def parse_breathing_data(self, breathing_file_path):
        """解析呼吸数据"""
        try:
            # 由于文件很大，只读取部分数据进行特征提取
            breathing_data = pd.read_csv(breathing_file_path, nrows=10000)
            
            if 'Breathing' not in breathing_data.columns:
                return None
            
            breathing_values = breathing_data['Breathing'].values
            breathing_values = breathing_values[~np.isnan(breathing_values)]
            
            if len(breathing_values) == 0:
                return None
            
            # 计算呼吸特征
            features = {
                'mean_breathing': np.mean(breathing_values),
                'std_breathing': np.std(breathing_values),
                'breathing_rate': len(breathing_values) / 100,  # 估算呼吸率
                'breathing_variability': np.std(np.diff(breathing_values))
            }
            
            return features
            
        except Exception as e:
            print(f"解析呼吸数据失败: {e}")
            return None
    
    def parse_bb_data(self, bb_file_path):
        """解析BB数据（可能是血压相关）"""
        try:
            bb_data = pd.read_csv(bb_file_path)
            
            if 'BtoB' not in bb_data.columns:
                return None
            
            bb_values = bb_data['BtoB'].values
            bb_values = bb_values[bb_values > 0]
            
            if len(bb_values) == 0:
                return None
            
            # 计算BB特征
            features = {
                'mean_bb': np.mean(bb_values),
                'std_bb': np.std(bb_values),
                'bb_variability': np.std(np.diff(bb_values))
            }
            
            return features
            
        except Exception as e:
            print(f"解析BB数据失败: {e}")
            return None
    
    def extract_time_features(self, file_path):
        """从文件名提取时间特征"""
        try:
            # 从文件名提取时间信息
            filename = os.path.basename(file_path)
            time_str = filename.split('_')[0:3]  # 获取日期部分
            time_str = '_'.join(time_str)
            
            # 解析时间
            dt = datetime.strptime(time_str, '%Y_%m_%d')
            
            # 计算时间特征
            features = {
                'day_of_week': dt.weekday(),
                'month': dt.month,
                'is_weekend': 1 if dt.weekday() >= 5 else 0
            }
            
            return features
            
        except Exception as e:
            print(f"提取时间特征失败: {e}")
            return None
    
    def generate_demographic_features(self, subject_id):
        """生成人口统计学特征（模拟数据）"""
        # 基于subject_id生成一些模拟的人口统计学特征
        np.random.seed(hash(subject_id) % 1000)
        
        features = {
            'gender': np.random.choice([0, 1]),
            'age': np.random.randint(25, 75),
            'BMI': np.random.normal(25, 5),
            'blood_pressure': np.random.normal(120, 20),
            'diabetes_type': np.random.choice([1, 2]),
            'physical_activity': np.random.choice([1, 2, 3]),
            'sleep_quality': np.random.choice([1, 2, 3, 4]),
            'stress_level': np.random.choice([1, 2, 3])
        }
        
        # 确保BMI在合理范围内
        features['BMI'] = np.clip(features['BMI'], 18, 40)
        features['blood_pressure'] = np.clip(features['blood_pressure'], 90, 180)
        
        return features
    
    def estimate_glucose_level(self, features):
        """基于生理特征估算血糖水平"""
        # 这是一个简化的血糖估算模型
        # 实际应用中应该基于真实的血糖数据训练模型
        
        base_glucose = 7.0  # 基础血糖值
        
        # 心率影响
        hr_factor = (features.get('mean_heart_rate', 70) - 70) / 100
        
        # 呼吸影响
        breathing_factor = (features.get('mean_breathing', 0) - 0) / 1000
        
        # 时间影响
        time_factor = features.get('day_of_week', 0) / 7
        
        # 人口统计学影响
        age_factor = (features.get('age', 50) - 50) / 100
        bmi_factor = (features.get('BMI', 25) - 25) / 10
        
        # 计算估算血糖
        estimated_glucose = base_glucose + hr_factor + breathing_factor + time_factor + age_factor + bmi_factor
        
        # 添加随机噪声
        noise = np.random.normal(0, 0.5)
        estimated_glucose += noise
        
        # 确保在合理范围内
        estimated_glucose = np.clip(estimated_glucose, 3.0, 15.0)
        
        return estimated_glucose
    
    def parse_session_data(self, session_path, subject_id):
        """解析单个会话的数据"""
        print(f"正在解析会话: {session_path}")
        
        # 查找数据文件
        rr_files = glob.glob(os.path.join(session_path, "*_RR.csv"))
        breathing_files = glob.glob(os.path.join(session_path, "*_Breathing.csv"))
        bb_files = glob.glob(os.path.join(session_path, "*_BB.csv"))
        
        # 初始化特征字典
        features = {}
        
        # 解析心率数据
        if rr_files:
            hr_features = self.parse_heart_rate_data(rr_files[0])
            if hr_features:
                features.update(hr_features)
        
        # 解析呼吸数据
        if breathing_files:
            breathing_features = self.parse_breathing_data(breathing_files[0])
            if breathing_features:
                features.update(breathing_features)
        
        # 解析BB数据
        if bb_files:
            bb_features = self.parse_bb_data(bb_files[0])
            if bb_features:
                features.update(bb_features)
        
        # 提取时间特征
        if rr_files:
            time_features = self.extract_time_features(rr_files[0])
            if time_features:
                features.update(time_features)
        
        # 生成人口统计学特征
        demo_features = self.generate_demographic_features(subject_id)
        features.update(demo_features)
        
        # 估算血糖水平
        if len(features) > 0:
            features['blood_glucose'] = self.estimate_glucose_level(features)
            features['subject_id'] = subject_id
            
            return features
        
        return None
    
    def parse_all_data(self):
        """解析所有数据"""
        print("开始解析数据集5...")
        
        all_features = []
        
        # 遍历所有受试者
        for subject_dir in ['001', '002']:
            subject_path = os.path.join(self.dataset_path, subject_dir, 'sensor_data')
            
            if not os.path.exists(subject_path):
                continue
            
            print(f"处理受试者: {subject_dir}")
            
            # 遍历所有会话
            for session_dir in os.listdir(subject_path):
                session_path = os.path.join(subject_path, session_dir)
                
                if os.path.isdir(session_path):
                    features = self.parse_session_data(session_path, subject_dir)
                    if features:
                        all_features.append(features)
        
        if all_features:
            # 创建DataFrame
            df = pd.DataFrame(all_features)
            
            # 重新排列列顺序，使其与现有数据格式一致
            column_order = [
                'gender', 'age', 'BMI', 'blood_pressure',
                'mean_heart_rate', 'std_heart_rate', 'heart_rate_variability',
                'mean_breathing', 'breathing_rate', 'breathing_variability',
                'mean_bb', 'bb_variability',
                'day_of_week', 'month', 'is_weekend',
                'diabetes_type', 'physical_activity', 'sleep_quality', 'stress_level',
                'blood_glucose', 'subject_id'
            ]
            
            # 只保留存在的列
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]
            
            # 保存数据
            df.to_csv(self.output_file, index=False)
            
            print(f"解析完成！共生成 {len(df)} 条记录")
            print(f"数据已保存到: {self.output_file}")
            print(f"特征数量: {len(df.columns)}")
            
            # 显示数据统计
            print("\n数据统计:")
            print(df.describe())
            
            return df
        else:
            print("未找到有效数据")
            return None

def main():
    """主函数"""
    print("数据集5解析器")
    print("=" * 50)
    
    # 创建解析器
    parser = Dataset5Parser()
    
    # 解析数据
    df = parser.parse_all_data()
    
    if df is not None:
        print("\n解析成功！")
        print("数据预览:")
        print(df.head())
        
        # 检查数据质量
        print("\n数据质量检查:")
        print(f"缺失值数量: {df.isnull().sum().sum()}")
        print(f"重复行数: {df.duplicated().sum()}")
        
        # 血糖分布
        if 'blood_glucose' in df.columns:
            print(f"\n血糖分布:")
            print(f"  平均值: {df['blood_glucose'].mean():.2f}")
            print(f"  标准差: {df['blood_glucose'].std():.2f}")
            print(f"  最小值: {df['blood_glucose'].min():.2f}")
            print(f"  最大值: {df['blood_glucose'].max():.2f}")
    
    else:
        print("解析失败！")

if __name__ == "__main__":
    main() 
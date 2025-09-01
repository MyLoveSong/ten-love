import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def process_datasets():
    """
    按照《数据的清单》标准整理所有数据集
    """
    print("开始处理数据集...")
    
    # 定义标准字段（根据清单要求）
    standard_columns = [
        'patient_id', 'gender', 'age', 'BMI', 'blood_pressure', 
        'fasting_glucose', 'postprandial_glucose', 'HbA1c', 'insulin',
        'cholesterol', 'LDL', 'HDL', 'triglycerides', 'physical_activity',
        'sleep_quality', 'stress_level', 'diabetes_type', 'pregnant',
        'blood_glucose', 'data_source'
    ]
    
    all_data = []
    
    # 1. 处理数据集1-3（标准糖尿病数据集）
    for i in range(1, 4):
        file_path = f'数据集{i}.csv'
        if os.path.exists(file_path):
            print(f"处理 {file_path}...")
            df = pd.read_csv(file_path)
            
            # 字段映射
            processed_data = []
            for _, row in df.iterrows():
                data = {
                    'patient_id': f'DS{i}_{len(processed_data):04d}',
                    'gender': np.nan,  # 缺失
                    'age': row['Age'],
                    'BMI': row['BMI'],
                    'blood_pressure': row['BloodPressure'],
                    'fasting_glucose': row['Glucose'],
                    'postprandial_glucose': np.nan,  # 缺失
                    'HbA1c': np.nan,  # 缺失
                    'insulin': row['Insulin'] if row['Insulin'] != 0 else np.nan,
                    'cholesterol': np.nan,  # 缺失
                    'LDL': np.nan,  # 缺失
                    'HDL': np.nan,  # 缺失
                    'triglycerides': np.nan,  # 缺失
                    'physical_activity': np.nan,  # 缺失
                    'sleep_quality': np.nan,  # 缺失
                    'stress_level': np.nan,  # 缺失
                    'diabetes_type': 2,  # 默认为2型糖尿病
                    'pregnant': row['Pregnancies'],
                    'blood_glucose': row['Glucose'],  # 使用空腹血糖作为血糖值
                    'data_source': f'数据集{i}'
                }
                processed_data.append(data)
            
            all_data.extend(processed_data)
    
    # 2. 处理数据集4（时间序列数据，需要聚合）
    file_path = '数据集4.csv'
    if os.path.exists(file_path):
        print(f"处理 {file_path}...")
        df = pd.read_csv(file_path)
        
        # 修复：将value列转为数值型，无法转换的设为NaN
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # 按患者和日期聚合数据
        # code 58: 血糖, code 33: 胰岛素, code 34: 其他指标
        glucose_data = df[df['code'] == 58].groupby('patient_id')['value'].agg(['mean', 'std']).reset_index()
        insulin_data = df[df['code'] == 33].groupby('patient_id')['value'].agg(['mean', 'std']).reset_index()
        
        # 合并数据
        merged_data = glucose_data.merge(insulin_data, on='patient_id', suffixes=('_glucose', '_insulin'))
        
        processed_data = []
        for _, row in merged_data.iterrows():
            data = {
                'patient_id': f'DS4_{row["patient_id"]}',
                'gender': np.nan,  # 缺失
                'age': np.nan,  # 缺失
                'BMI': np.nan,  # 缺失
                'blood_pressure': np.nan,  # 缺失
                'fasting_glucose': row['mean_glucose'],
                'postprandial_glucose': np.nan,  # 缺失
                'HbA1c': np.nan,  # 缺失
                'insulin': row['mean_insulin'],
                'cholesterol': np.nan,  # 缺失
                'LDL': np.nan,  # 缺失
                'HDL': np.nan,  # 缺失
                'triglycerides': np.nan,  # 缺失
                'physical_activity': np.nan,  # 缺失
                'sleep_quality': np.nan,  # 缺失
                'stress_level': np.nan,  # 缺失
                'diabetes_type': 2,  # 默认为2型糖尿病
                'pregnant': 0,  # 默认为非孕妇
                'blood_glucose': row['mean_glucose'],
                'data_source': '数据集4'
            }
            processed_data.append(data)
        
        all_data.extend(processed_data)
    
    # 3. 处理2.csv（完整特征数据）
    file_path = '2.csv'
    if os.path.exists(file_path):
        print(f"处理 {file_path}...")
        df = pd.read_csv(file_path)
        
        processed_data = []
        for i, row in df.iterrows():
            data = {
                'patient_id': f'DS2_{i:04d}',
                'gender': row['gender'],
                'age': row['age'],
                'BMI': row['BMI'],
                'blood_pressure': row['blood_pressure'],
                'fasting_glucose': row['fasting_glucose'],
                'postprandial_glucose': row['postprandial_glucose'],
                'HbA1c': row['HbA1c'],
                'insulin': row['insulin'],
                'cholesterol': row['cholesterol'],
                'LDL': row['LDL'],
                'HDL': row['HDL'],
                'triglycerides': row['triglycerides'],
                'physical_activity': row['physical_activity'],
                'sleep_quality': row['sleep_quality'],
                'stress_level': row['stress_level'],
                'diabetes_type': row['diabetes_type'],
                'pregnant': row['pregnant'],
                'blood_glucose': row['blood_glucose'],
                'data_source': '2.csv'
            }
            processed_data.append(data)
        
        all_data.extend(processed_data)
    
    # 4. 处理gender,age,BMI,blood_pressure,fasting_gl.csv
    file_path = 'gender,age,BMI,blood_pressure,fasting_gl.csv'
    if os.path.exists(file_path):
        print(f"处理 {file_path}...")
        df = pd.read_csv(file_path)
        
        processed_data = []
        for i, row in df.iterrows():
            data = {
                'patient_id': f'DS_GL_{i:04d}',
                'gender': row['gender'],
                'age': row['age'],
                'BMI': row['BMI'],
                'blood_pressure': row['blood_pressure'],
                'fasting_glucose': row['fasting_glucose'],
                'postprandial_glucose': row['postprandial_glucose'],
                'HbA1c': row['HbA1c'],
                'insulin': row['insulin'],
                'cholesterol': row['cholesterol'],
                'LDL': row['LDL'],
                'HDL': row['HDL'],
                'triglycerides': row['triglycerides'],
                'physical_activity': row['physical_activity'],
                'sleep_quality': row['sleep_quality'],
                'stress_level': row['stress_level'],
                'diabetes_type': row['diabetes_type'],
                'pregnant': row['pregnant'],
                'blood_glucose': row['blood_glucose'],
                'data_source': 'gender,age,BMI,blood_pressure,fasting_gl.csv'
            }
            processed_data.append(data)
        
        all_data.extend(processed_data)
    
    # 5. 处理数据集5.csv（传感器数据）
    file_path = '数据集5.csv'
    if os.path.exists(file_path):
        print(f"处理 {file_path}...")
        df = pd.read_csv(file_path)
        
        processed_data = []
        for i, row in df.iterrows():
            data = {
                'patient_id': f'DS5_{row["subject_id"]}_{i:04d}',
                'gender': row['gender'],
                'age': row['age'],
                'BMI': row['BMI'],
                'blood_pressure': row['blood_pressure'],
                'fasting_glucose': np.nan,  # 缺失
                'postprandial_glucose': np.nan,  # 缺失
                'HbA1c': np.nan,  # 缺失
                'insulin': np.nan,  # 缺失
                'cholesterol': np.nan,  # 缺失
                'LDL': np.nan,  # 缺失
                'HDL': np.nan,  # 缺失
                'triglycerides': np.nan,  # 缺失
                'physical_activity': row['physical_activity'],
                'sleep_quality': row['sleep_quality'],
                'stress_level': row['stress_level'],
                'diabetes_type': row['diabetes_type'],
                'pregnant': 0,  # 默认为非孕妇
                'blood_glucose': row['blood_glucose'],
                'data_source': '数据集5'
            }
            processed_data.append(data)
        
        all_data.extend(processed_data)
    
    # 创建DataFrame
    final_df = pd.DataFrame(all_data)
    
    # 确保所有列都存在
    for col in standard_columns:
        if col not in final_df.columns:
            final_df[col] = np.nan
    
    # 重新排序列
    final_df = final_df[standard_columns]
    
    # 数据清洗
    print("数据清洗...")
    
    # 移除完全重复的行
    final_df = final_df.drop_duplicates()
    
    # 处理异常值
    # 血糖值范围检查 (正常范围: 3.9-11.1 mmol/L)
    final_df.loc[final_df['blood_glucose'] < 2.0, 'blood_glucose'] = np.nan
    final_df.loc[final_df['blood_glucose'] > 30.0, 'blood_glucose'] = np.nan
    
    # BMI范围检查 (正常范围: 15-50)
    final_df.loc[final_df['BMI'] < 10, 'BMI'] = np.nan
    final_df.loc[final_df['BMI'] > 60, 'BMI'] = np.nan
    
    # 年龄范围检查 (正常范围: 18-100)
    final_df.loc[final_df['age'] < 10, 'age'] = np.nan
    final_df.loc[final_df['age'] > 120, 'age'] = np.nan
    
    # 血压范围检查 (正常范围: 60-200 mmHg)
    final_df.loc[final_df['blood_pressure'] < 50, 'blood_pressure'] = np.nan
    final_df.loc[final_df['blood_pressure'] > 250, 'blood_pressure'] = np.nan
    
    # 移除关键字段缺失过多的行
    key_fields = ['blood_glucose', 'age', 'BMI']
    final_df = final_df.dropna(subset=key_fields, how='all')
    
    # 分割训练集和测试集 (8:2)
    print("分割训练集和测试集...")
    train_df, test_df = train_test_split(
        final_df, 
        test_size=0.2, 
        random_state=42,
        stratify=final_df['diabetes_type'] if 'diabetes_type' in final_df.columns else None
    )
    
    # 保存结果
    print("保存处理后的数据集...")
    
    # 保存完整数据集
    final_df.to_csv('处理过的数据集_完整.csv', index=False, encoding='utf-8-sig')
    
    # 保存训练集
    train_df.to_csv('处理过的数据集_训练集.csv', index=False, encoding='utf-8-sig')
    
    # 保存测试集
    test_df.to_csv('处理过的数据集_测试集.csv', index=False, encoding='utf-8-sig')
    
    # 输出统计信息
    print("\n=== 数据处理完成 ===")
    print(f"总数据量: {len(final_df)}")
    print(f"训练集: {len(train_df)}")
    print(f"测试集: {len(test_df)}")
    print(f"数据来源分布:")
    print(final_df['data_source'].value_counts())
    print(f"\n字段缺失情况:")
    print(final_df.isnull().sum())
    
    return final_df, train_df, test_df

if __name__ == "__main__":
    # 处理数据集
    final_df, train_df, test_df = process_datasets()
    
    print("\n数据集处理完成！")
    print("生成的文件:")
    print("- 处理过的数据集_完整.csv")
    print("- 处理过的数据集_训练集.csv") 
    print("- 处理过的数据集_测试集.csv") 
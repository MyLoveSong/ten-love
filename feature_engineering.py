import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

def feature_engineering(input_file, output_file):
    print(f"读取数据: {input_file}")
    df = pd.read_csv(input_file)

    # 1. 缺失值智能填充
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in df.columns if col not in num_cols and col not in ['patient_id', 'data_source']]

    # 数值型：用中位数填充
    for col in num_cols:
        if bool(df[col].isnull().any()):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # 分类型：用众数填充
    for col in cat_cols:
        if bool(df[col].isnull().any()):
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
            else:
                df[col] = df[col].fillna('Unknown')

    # 2. 类别特征自动编码（如gender, physical_activity等）
    # 只对object类型和部分特殊列做one-hot
    encode_cols = []
    for col in ['gender', 'physical_activity', 'sleep_quality', 'stress_level', 'diabetes_type', 'pregnant', 'data_source']:
        if col in df.columns:
            encode_cols.append(col)
    df = pd.get_dummies(df, columns=encode_cols, drop_first=True)

    # 3. 数值特征归一化/标准化
    scale_cols = [col for col in df.columns if col not in ['patient_id'] and df[col].dtype in [np.float64, np.int64]]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # 4. 交互特征、分箱特征
    # BMI分级
    if 'BMI' in df.columns:
        df['BMI_level'] = pd.cut(df['BMI'], bins=[-np.inf, 18.5, 24, 28, np.inf], labels=[0,1,2,3])
    # 年龄分段
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100], labels=[0,1,2,3])
    # 血糖高低风险分箱
    if 'blood_glucose' in df.columns:
        df['glucose_risk'] = pd.cut(df['blood_glucose'], bins=[-np.inf, 6.1, 7.8, 11.1, np.inf], labels=[0,1,2,3])

    # 5. 交互特征示例：BMI*age
    if 'BMI' in df.columns and 'age' in df.columns:
        df['BMI_age'] = df['BMI'] * df['age']

    # 6. 去除无用列
    if 'patient_id' in df.columns:
        df = df.drop(columns=['patient_id'])

    print(f"保存特征工程后数据: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"特征工程完成，输出文件: {output_file}")

if __name__ == "__main__":
    feature_engineering('处理过的数据集_完整_增强版.csv', '处理过的数据集_特征工程版.csv') 
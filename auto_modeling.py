import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 导入XGBoost（已安装）
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import os

def auto_modeling(input_file, target_col='blood_glucose'):
    print(f"读取数据: {input_file}")
    df = pd.read_csv(input_file)

    # 再次填充所有特征的缺失值（均值填充）
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna(0)  # 对非数值型的NaN填0

    # 自动分割训练集和测试集
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }

    results = []
    best_score = -np.inf
    best_model = None
    best_name = ''

    for name, model in models.items():
        print(f"\n训练模型: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({'model': name, 'MSE': mse, 'MAE': mae, 'R2': r2})
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
        # 可选：保存每个模型
        joblib.dump(model, f'model_{name}.pkl')

    # 输出对比报告
    report = pd.DataFrame(results)
    report.to_csv('model_compare_report.csv', index=False)
    print("\n=== 模型对比报告 ===")
    print(report)
    print(f"\n最佳模型: {best_name} (R2={best_score:.4f})，已保存为 model_{best_name}.pkl")

    if best_model is not None:
        # 可选：绘制特征重要性
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feat_names = X.columns
            plt.figure(figsize=(10,5))
            plt.bar(feat_names, importances)
            plt.xticks(rotation=90)
            plt.title(f'Feature Importance ({best_name})')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            print("特征重要性图已保存为 feature_importance.png")

        # 可选：绘制预测效果
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, best_model.predict(X_test), alpha=0.5)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'Prediction Scatter ({best_name})')
        # 确保y_test是pandas Series类型，然后调用min/max方法
        y_test_series = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test
        plt.plot([y_test_series.min(), y_test_series.max()], [y_test_series.min(), y_test_series.max()], 'r--')
        plt.tight_layout()
        plt.savefig('prediction_scatter.png')
        plt.close()
        print("预测效果图已保存为 prediction_scatter.png")
    else:
        print("未能成功训练任何模型，未生成可视化图。")

if __name__ == "__main__":
    auto_modeling('处理过的数据集_特征工程版.csv', target_col='blood_glucose') 
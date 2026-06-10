

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版学术系统启动脚本
Enhanced Academic System Launcher

本脚本提供了完整的学术级系统启动和演示功能：
1. 系统初始化和配置验证
2. 演示科学实验流程
3. 生成示例学术报告
4. 导出发表材料

使用方法:
    python start_enhanced_academic_system.py [选项]

选项:
    --demo: 运行演示实验
    --config: 指定配置文件路径
    --experiment: 运行特定实验
    --validate: 验证系统配置
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入增强系统
from enhanced_academic_system import EnhancedAcademicSystem, quick_scientific_analysis

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_system_setup() -> bool:
    """
    验证系统设置

    Returns:
        验证是否成功
    """
    print("🔍 验证系统设置...")

    try:
        # 1. 检查Python版本
        import sys
        if sys.version_info < (3, 8):
            print("❌ Python版本过低，需要3.8+")
            return False
        print(f"✅ Python版本: {sys.version}")

        # 2. 检查关键依赖
        required_packages = [
            'numpy', 'pandas', 'torch', 'sklearn',
            'matplotlib', 'seaborn', 'plotly', 'fastapi'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}: 已安装")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package}: 未安装")

        if missing_packages:
            print(f"请安装缺失的包: pip install {' '.join(missing_packages)}")
            return False

        # 3. 检查数据文件
        data_paths = [
            "../血糖预测/x血糖/处理过的数据集_完整_增强版.csv",
            "../图像识别/图像识别/图像识别/image_recognition/sample_data"
        ]

        for path in data_paths:
            if Path(path).exists():
                print(f"✅ 数据路径存在: {path}")
            else:
                print(f"⚠️ 数据路径不存在: {path}")

        # 4. 检查模型文件
        model_paths = [
            "../血糖预测/x血糖/trained_gluformer_model.pth",
            "../图像识别/图像识别/图像识别/image_recognition/models/best_model.pth"
        ]

        for path in model_paths:
            if Path(path).exists():
                print(f"✅ 模型文件存在: {path}")
            else:
                print(f"⚠️ 模型文件不存在: {path}")

        print("✅ 系统验证完成")
        return True

    except Exception as e:
        print(f"❌ 系统验证失败: {e}")
        return False

def run_demo_experiment() -> None:
    """运行演示实验"""
    print("\n🧪 开始演示实验...")

    try:
        # 1. 创建增强系统
        system = EnhancedAcademicSystem()
        print("✅ 增强学术系统初始化完成")

        # 2. 启动科学实验
        experiment_id = system.start_scientific_experiment(
            experiment_name="多模态健康监测演示实验",
            description="演示增强版学术系统的完整功能",
            author="研究团队"
        )
        print(f"✅ 科学实验已启动: {experiment_id}")

        # 3. 生成演示数据
        import numpy as np
        import pandas as pd

        # 模拟血糖数据
        np.random.seed(42)
        n_samples = 500

        demo_data = pd.DataFrame({
            'age': np.random.normal(60, 15, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'blood_pressure': np.random.normal(130, 20, n_samples),
            'hba1c': np.random.normal(8, 2, n_samples),
            'glucose_level': np.random.normal(150, 30, n_samples)
        })

        # 确保数据合理性
        demo_data = demo_data.clip(lower=0)
        # 创建更平衡的目标变量
        target_threshold = demo_data['glucose_level'].quantile(0.7)  # 70%分位数
        demo_data['target'] = (demo_data['glucose_level'] > target_threshold).astype(int)

        print(f"✅ 演示数据生成完成: {len(demo_data)} 样本")

        # 4. 实验设计验证
        design_results = system.design_and_validate_experiment(
            demo_data,
            'target',
            ['age', 'bmi']
        )
        print("✅ 实验设计验证完成")

        # 5. 模拟模型训练和预测
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X = demo_data.drop(['target', 'glucose_level'], axis=1)
        y = demo_data['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 训练模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print("✅ 模型训练和预测完成")

        # 6. 综合评估
        metrics = system.run_comprehensive_evaluation(
            y_test.values, y_pred, y_prob,
            task_type='classification'
        )
        print("✅ 综合评估完成")

        # 7. 统计分析
        stats_results = system.perform_statistical_analysis(
            y_test.values.astype(float),
            y_pred.astype(float)
        )
        print("✅ 统计分析完成")

        # 8. 稳健性验证
        robustness_results = system.validate_model_robustness(
            model, X_test.values, y_test.values
        )
        print("✅ 稳健性验证完成")

        # 9. 完成实验
        experiment_results = {
            'basic_metrics': metrics,
            'statistical_analysis': stats_results,
            'experimental_design': design_results,
            'robustness_analysis': robustness_results,
            'dataset': demo_data,
            'model': model,
            'hyperparameters': {
                'n_estimators': 100,
                'random_state': 42
            }
        }

        scientific_result = system.complete_scientific_experiment(experiment_results)
        print("✅ 科学实验完成")

        # 10. 生成报告
        report_path = system.generate_academic_report(scientific_result)
        print(f"✅ 学术报告已生成: {report_path}")

        # 11. 导出发表材料
        publication_files = system.export_results_for_publication(scientific_result)
        print("✅ 发表材料已导出:")
        for material_type, file_path in publication_files.items():
            print(f"   - {material_type}: {file_path}")

        # 12. 显示关键结果
        print("\n📊 关键实验结果:")
        if metrics.accuracy:
            print(f"   - 准确率: {metrics.accuracy.value:.3f}")
        if metrics.auc_roc:
            print(f"   - AUC-ROC: {metrics.auc_roc.value:.3f}")
        if 'cohens_d' in stats_results.get('effect_size_analysis', {}):
            cohen_d = stats_results['effect_size_analysis']['cohens_d']['value']
            print(f"   - Cohen's d: {cohen_d:.3f}")

        print("\n🎉 演示实验成功完成！")

    except Exception as e:
        print(f"❌ 演示实验失败: {e}")
        logger.exception("演示实验异常")

def run_quick_analysis() -> None:
    """运行快速分析"""
    print("\n⚡ 开始快速科学分析...")

    try:
        # 生成示例数据
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        n_samples = 300

        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(1, 1.5, n_samples),
            'feature3': np.random.uniform(-2, 2, n_samples),
            'target': np.random.binomial(1, 0.3, n_samples)
        })

        # 训练模型
        X = data.drop('target', axis=1)
        y = data['target']

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        # 快速分析
        results = quick_scientific_analysis(
            data, 'target', model, "快速分析演示"
        )

        print(f"✅ 快速分析完成")
        print(f"   - 实验ID: {results['experiment_id']}")
        print(f"   - 系统版本: {results['system'].config['version']}")

    except Exception as e:
        print(f"❌ 快速分析失败: {e}")
        logger.exception("快速分析异常")

def show_system_info(config_path: str = None) -> None:
    """显示系统信息"""
    print("\nℹ️ 增强版学术系统信息")
    print("=" * 50)

    try:
        system = EnhancedAcademicSystem(config_path)

        print(f"📋 研究名称: {system.config['study_name']}")
        print(f"🔢 系统版本: {system.config['version']}")
        print(f"📝 描述: {system.config['description']}")
        print(f"🎯 置信水平: {system.config['confidence_level']}")
        print(f"⚡ 显著性水平: {system.config['alpha']}")
        print(f"🔋 统计功效: {system.config['power']}")
        print(f"🎲 随机种子: {system.config['random_seed']}")

        print("\n🔧 启用的功能:")
        features = [
            ("统计分析增强", system.config['statistical_analysis']['enable_multiple_comparison_correction']),
            ("实验设计优化", system.config['experimental_design']['enable_stratified_sampling']),
            ("综合评估指标", system.config['evaluation_metrics']['enable_clinical_metrics']),
            ("可重现性保证", system.config['reproducibility']['enable_environment_tracking']),
        ]

        for feature_name, enabled in features:
            status = "✅" if enabled else "❌"
            print(f"   {status} {feature_name}")

        print(f"\n📂 配置文件: {system.config_path}")
        print(f"📁 可重现性目录: {system.config['reproducibility_dir']}")

    except Exception as e:
        print(f"❌ 获取系统信息失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="增强版学术级智能健康监测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python start_enhanced_academic_system.py --demo
    python start_enhanced_academic_system.py --validate
    python start_enhanced_academic_system.py --quick
    python start_enhanced_academic_system.py --info
        """
    )

    parser.add_argument('--demo', action='store_true',
                       help='运行完整演示实验')
    parser.add_argument('--quick', action='store_true',
                       help='运行快速科学分析')
    parser.add_argument('--validate', action='store_true',
                       help='验证系统配置和依赖')
    parser.add_argument('--info', action='store_true',
                       help='显示系统信息')
    parser.add_argument('--config', type=str,
                       help='指定配置文件路径')

    args = parser.parse_args()

    print("🎯 增强版学术级智能健康监测系统")
    print("=" * 60)
    print("版本: 3.0.0 (Scientific Research Edition)")
    print("时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    if args.validate:
        if not validate_system_setup():
            sys.exit(1)

    if args.info:
        show_system_info(args.config)

    if args.demo:
        if not validate_system_setup():
            print("⚠️ 系统验证失败，但将尝试运行演示...")
        run_demo_experiment()

    if args.quick:
        run_quick_analysis()

    if not any([args.demo, args.quick, args.validate, args.info]):
        # 默认运行交互式菜单
        while True:
            print("\n📋 请选择操作:")
            print("1. 🧪 运行演示实验")
            print("2. ⚡ 快速科学分析")
            print("3. 🔍 验证系统配置")
            print("4. ℹ️ 显示系统信息")
            print("5. 🚪 退出")

            try:
                choice = input("\n请输入选项 (1-5): ").strip()

                if choice == '1':
                    run_demo_experiment()
                elif choice == '2':
                    run_quick_analysis()
                elif choice == '3':
                    validate_system_setup()
                elif choice == '4':
                    show_system_info(args.config)
                elif choice == '5':
                    print("👋 再见！")
                    break
                else:
                    print("❌ 无效选项，请重新选择")

            except KeyboardInterrupt:
                print("\n\n👋 用户中断，退出系统")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                logger.exception("主程序异常")

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'validate_system_setup'", "'run_demo_experiment'", "'run_quick_analysis'", "'show_system_info'", "'main'"]

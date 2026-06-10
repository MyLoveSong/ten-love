#!/usr/bin/env python3
"""
数据驱动微调主入口脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据驱动微调')

    # 通用参数
    parser.add_argument('--mode', type=str, required=True,
                        choices=['gluformer', 'cultural', 'online'],
                        help='微调模式：gluformer(血糖预测), cultural(文化适配), online(在线学习)')
    parser.add_argument('--base_model', type=str, required=True,
                        help='基础模型路径')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')

    # 血糖预测微调参数
    parser.add_argument('--finetune_mode', type=str, default='head_only',
                        choices=['head_only', 'lora', 'full'],
                        help='微调模式：head_only(仅头部), lora(LoRA), full(全模型)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮次')

    # 文化适配微调参数
    parser.add_argument('--num_regions', type=int, default=34,
                        help='地区数量')

    # 在线学习参数
    parser.add_argument('--feedback_dir', type=str, default=None,
                        help='反馈数据目录')
    parser.add_argument('--run_time', type=int, default=3600,
                        help='运行时间（秒）')

    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()

    # 根据模式选择微调方法
    if args.mode == 'gluformer':
        run_gluformer_finetune(args)
    elif args.mode == 'cultural':
        run_cultural_adaptation(args)
    elif args.mode == 'online':
        run_online_learning(args)
    else:
        logger.error(f"不支持的模式: {args.mode}")
        sys.exit(1)

def run_gluformer_finetune(args):
    """运行血糖预测微调"""
    from .models.gluformer_finetune import GluFormerHeadFineTuner
    from .utils.data_utils import CGMDataset, create_data_loaders, load_config

    # 设置默认配置路径
    if args.config is None:
        args.config = os.path.join(os.path.dirname(__file__), 'configs', 'gluformer_finetune.yaml')

    # 设置默认输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'gluformer_finetune')

    # 加载配置
    config = load_config(args.config)

    # 更新配置
    if args.batch_size is not None:
        config['finetune']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['finetune']['num_epochs'] = args.epochs
    config['output']['save_dir'] = args.output_dir

    # 创建数据集
    dataset = CGMDataset(
        data_path=args.data_path,
        sequence_length=config['data'].get('sequence_length', 24),
        prediction_horizon=config['data'].get('prediction_horizon', 6),
        use_personal_features=True,
        use_multimodal=False
    )

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=config['finetune'].get('batch_size', 32),
        train_ratio=config['data'].get('train_val_test_split', [0.7, 0.15, 0.15])[0],
        val_ratio=config['data'].get('train_val_test_split', [0.7, 0.15, 0.15])[1],
        test_ratio=config['data'].get('train_val_test_split', [0.7, 0.15, 0.15])[2],
        num_workers=config['hardware'].get('num_workers', 4) if 'hardware' in config else 4
    )

    # 创建微调器
    finetuner = GluFormerHeadFineTuner(
        base_model_path=args.base_model,
        config_path=args.config,
        finetune_mode=args.finetune_mode
    )

    # 微调模型
    training_summary = finetuner.finetune(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    # 打印结果
    print("\n" + "="*50)
    print("GluFormer微调完成!")
    print("="*50)
    print(f"最佳验证损失: {training_summary['best_val_loss']:.6f}")
    print(f"最佳轮次: {training_summary['best_epoch']}")
    if 'test_results' in training_summary:
        print(f"测试损失: {training_summary['test_results']['test_loss']:.6f}")
        print(f"测试MAE: {training_summary['test_results']['test_metrics']['mae']:.6f}")
    print("="*50)

def run_cultural_adaptation(args):
    """运行文化适配微调"""
    from .models.cultural_adaptation import CulturalAdaptationFineTuner
    from .utils.data_utils import CulturalPreferenceDataset, create_data_loaders, load_config

    # 设置默认配置路径
    if args.config is None:
        args.config = os.path.join(os.path.dirname(__file__), 'configs', 'cultural_adaptation.yaml')

    # 设置默认输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'cultural_adaptation')

    # 加载配置
    config = load_config(args.config)

    # 更新配置
    if args.batch_size is not None:
        config['finetune']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['finetune']['num_epochs'] = args.epochs
    config['output']['save_dir'] = args.output_dir

    # 创建数据集
    dataset = CulturalPreferenceDataset(
        data_path=args.data_path,
        region_embedding=True
    )

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=config['finetune'].get('batch_size', 64),
        train_ratio=config['data'].get('train_val_test_split', [0.7, 0.15, 0.15])[0],
        val_ratio=config['data'].get('train_val_test_split', [0.7, 0.15, 0.15])[1],
        test_ratio=config['data'].get('train_val_test_split', [0.7, 0.15, 0.15])[2],
        num_workers=config['hardware'].get('num_workers', 4) if 'hardware' in config else 4
    )

    # 创建微调器
    finetuner = CulturalAdaptationFineTuner(
        base_model_path=args.base_model if args.base_model != 'none' else None,
        config_path=args.config,
        num_regions=args.num_regions,
        finetune_mode='lora' if args.finetune_mode == 'lora' else 'full'
    )

    # 微调模型
    training_summary = finetuner.finetune(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    # 打印结果
    print("\n" + "="*50)
    print("文化适配微调完成!")
    print("="*50)
    print(f"最佳验证损失: {training_summary['best_val_loss']:.6f}")
    print(f"最佳轮次: {training_summary['best_epoch']}")
    if 'test_results' in training_summary:
        print(f"测试损失: {training_summary['test_results']['test_loss']:.6f}")
    print("="*50)

def run_online_learning(args):
    """运行在线学习管道"""
    import time
    from .models.online_learning import OnlineLearningPipeline
    from .utils.data_utils import load_config

    # 设置默认配置路径
    if args.config is None:
        args.config = os.path.join(os.path.dirname(__file__), 'configs', 'online_learning.yaml')

    # 设置默认输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'online_learning')

    # 加载配置
    config = load_config(args.config)

    # 更新配置
    config['output']['save_dir'] = args.output_dir

    # 创建在线学习管道
    pipeline = OnlineLearningPipeline(
        base_model_path=args.base_model,
        config_path=args.config,
        feedback_dir=args.feedback_dir
    )

    # 启动管道
    pipeline.start()

    try:
        # 运行指定时间
        logger.info(f"在线学习管道将运行 {args.run_time} 秒")
        time.sleep(args.run_time)
    except KeyboardInterrupt:
        logger.info("接收到中断信号，停止管道")
    finally:
        # 停止管道
        pipeline.stop()

        # 保存最终模型
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(
            args.output_dir,
            f"online_learning_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        pipeline.save_model(save_path)

        # 打印模型信息
        model_info = pipeline.get_model_info()
        print("\n" + "="*50)
        print("在线学习管道运行完成!")
        print("="*50)
        print(f"模型版本: {model_info['model_version']}")
        print(f"更新次数: {model_info['update_count']}")
        print(f"最终模型已保存到: {save_path}")
        print("="*50)

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型血糖预测系统启动脚本
快速启动和测试系统功能
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_glucose_system import EnhancedGlucosePredictionSystem

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_system_startup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SystemLauncher:
    """系统启动器"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or "enhanced_config.json"
        self.system = None

    async def start_system(self):
        """启动系统"""
        try:
            print("🚀 启动增强型血糖预测系统...")
            print("=" * 60)

            # 创建系统实例
            self.system = EnhancedGlucosePredictionSystem(self.config_path)

            # 初始化系统
            print("📋 正在初始化系统组件...")
            if not await self.system.initialize_system():
                print("❌ 系统初始化失败")
                return False

            print("✅ 系统初始化成功")

            # 显示系统状态
            await self.show_system_status()

            # 运行演示
            await self.run_demo()

            return True

        except Exception as e:
            print(f"❌ 系统启动失败: {e}")
            logger.error(f"系统启动失败: {e}")
            return False

    async def show_system_status(self):
        """显示系统状态"""
        print("\n📊 系统状态:")
        print("-" * 40)

        status = await self.system.get_system_status()

        print(f"系统初始化: {'✅' if status['system_initialized'] else '❌'}")
        print(f"数据源连接: {'✅' if status['data_sources_connected'] else '❌'}")

        modules = status.get('modules_enabled', {})
        print(f"数据集成: {'✅' if modules.get('data_integration') else '❌'}")
        print(f"数据增强: {'✅' if modules.get('data_augmentation') else '❌'}")
        print(f"反馈优化: {'✅' if modules.get('feedback_optimization') else '❌'}")

        # 显示优化统计
        if 'optimization_stats' in status:
            stats = status['optimization_stats']
            print(f"\n📈 优化统计:")
            print(f"反馈数量: {stats.get('total_feedback_count', 0)}")
            print(f"预测数量: {stats.get('total_predictions_count', 0)}")
            print(f"反馈率: {stats.get('feedback_rate', 0):.2%}")

    async def run_demo(self):
        """运行演示"""
        print("\n🎯 运行系统演示...")
        print("-" * 40)

        # 演示用户数据
        demo_users = [
            {
                "user_id": "demo_user_1",
                "profile": {
                    "age": 45,
                    "bmi": 25.5,
                    "exercise_level": 3,
                    "stress_level": 2,
                    "sleep_hours": 7
                },
                "input_data": {
                    "age": 45,
                    "bmi": 25.5,
                    "glucose": 6.0,
                    "carbohydrates": 50,
                    "exercise": 30,
                    "hour": 12,
                    "day_of_week": 1
                }
            },
            {
                "user_id": "demo_user_2",
                "profile": {
                    "age": 35,
                    "bmi": 28.0,
                    "exercise_level": 2,
                    "stress_level": 4,
                    "sleep_hours": 6
                },
                "input_data": {
                    "age": 35,
                    "bmi": 28.0,
                    "glucose": 7.5,
                    "carbohydrates": 80,
                    "exercise": 15,
                    "hour": 18,
                    "day_of_week": 5
                }
            }
        ]

        for i, user in enumerate(demo_users, 1):
            print(f"\n👤 演示用户 {i}: {user['user_id']}")

            # 获取预测
            try:
                prediction = await self.system.get_enhanced_prediction(
                    user['user_id'],
                    user['input_data'],
                    use_real_time_data=False  # 演示模式不使用实时数据
                )

                print(f"🔮 预测结果:")
                if 'error' in prediction:
                    print(f"   ❌ 预测失败: {prediction['error']}")
                else:
                    print(f"   📊 预测血糖值: {prediction.get('predictions', ['N/A'])[0]:.2f} mmol/L")
                    print(f"   🎯 血糖状态: {prediction.get('glucose_status', ['N/A'])[0]}")
                    print(f"   📈 置信度: {prediction.get('confidence', 0):.2%}")

                    if prediction.get('recommendations'):
                        print(f"   💡 建议:")
                        for j, rec in enumerate(prediction['recommendations'][:3], 1):
                            print(f"      {j}. {rec}")

                # 模拟用户反馈
                if 'error' not in prediction:
                    actual_glucose = float(prediction.get('predictions', [6.0])[0]) + 0.2
                    satisfaction_score = 4 if actual_glucose < 7.0 else 3

                    await self.system.process_user_feedback(
                        user['user_id'],
                        datetime.now().isoformat(),
                        actual_glucose,
                        satisfaction_score,
                        user['profile']
                    )

                    print(f"   📝 模拟反馈: 实际血糖 {actual_glucose:.2f}, 满意度 {satisfaction_score}/5")

            except Exception as e:
                print(f"   ❌ 演示失败: {e}")
                logger.error(f"演示失败: {e}")

            # 获取个性化推荐
            try:
                recommendations = await self.system.get_personalized_recommendations(
                    user['user_id'],
                    user['profile']
                )

                print(f"🎯 个性化推荐:")
                if 'error' in recommendations:
                    print(f"   ❌ 推荐失败: {recommendations['error']}")
                else:
                    print(f"   💡 推荐: {recommendations.get('recommendation', 'N/A')}")
                    print(f"   📊 置信度: {recommendations.get('confidence', 0):.2%}")
                    print(f"   🎨 个性化程度: {recommendations.get('personalization_level', 0):.2%}")

            except Exception as e:
                print(f"   ❌ 推荐失败: {e}")
                logger.error(f"推荐失败: {e}")

    async def interactive_mode(self):
        """交互模式"""
        print("\n🎮 进入交互模式...")
        print("输入 'help' 查看可用命令")

        while True:
            try:
                command = input("\n> ").strip().lower()

                if command == 'help':
                    self.show_help()
                elif command == 'status':
                    await self.show_system_status()
                elif command == 'demo':
                    await self.run_demo()
                elif command == 'predict':
                    await self.interactive_predict()
                elif command == 'feedback':
                    await self.interactive_feedback()
                elif command == 'recommend':
                    await self.interactive_recommend()
                elif command == 'quit' or command == 'exit':
                    print("👋 退出系统")
                    break
                else:
                    print("❌ 未知命令，输入 'help' 查看帮助")

            except KeyboardInterrupt:
                print("\n👋 退出系统")
                break
            except Exception as e:
                print(f"❌ 命令执行失败: {e}")

    def show_help(self):
        """显示帮助信息"""
        print("\n📖 可用命令:")
        print("  help      - 显示帮助信息")
        print("  status    - 显示系统状态")
        print("  demo      - 运行演示")
        print("  predict   - 交互式预测")
        print("  feedback  - 提交反馈")
        print("  recommend - 获取推荐")
        print("  quit      - 退出系统")

    async def interactive_predict(self):
        """交互式预测"""
        print("\n🔮 交互式血糖预测")

        try:
            user_id = input("用户ID: ").strip() or "interactive_user"

            print("请输入预测参数 (直接回车使用默认值):")
            age = float(input("年龄 (默认: 45): ") or "45")
            bmi = float(input("BMI (默认: 25.5): ") or "25.5")
            glucose = float(input("当前血糖 (默认: 6.0): ") or "6.0")
            carbs = float(input("碳水化合物摄入 (默认: 50): ") or "50")
            exercise = float(input("运动时间/分钟 (默认: 30): ") or "30")

            input_data = {
                'age': age,
                'bmi': bmi,
                'glucose': glucose,
                'carbohydrates': carbs,
                'exercise': exercise,
                'hour': datetime.now().hour,
                'day_of_week': datetime.now().weekday()
            }

            prediction = await self.system.get_enhanced_prediction(
                user_id, input_data, use_real_time_data=False
            )

            if 'error' in prediction:
                print(f"❌ 预测失败: {prediction['error']}")
            else:
                print(f"✅ 预测成功:")
                print(f"   预测血糖值: {prediction.get('predictions', ['N/A'])[0]:.2f} mmol/L")
                print(f"   血糖状态: {prediction.get('glucose_status', ['N/A'])[0]}")
                print(f"   置信度: {prediction.get('confidence', 0):.2%}")

        except ValueError:
            print("❌ 输入格式错误")
        except Exception as e:
            print(f"❌ 预测失败: {e}")

    async def interactive_feedback(self):
        """交互式反馈"""
        print("\n📝 提交用户反馈")

        try:
            user_id = input("用户ID: ").strip() or "interactive_user"
            prediction_id = input("预测ID (默认: 当前时间): ").strip() or datetime.now().isoformat()
            actual_glucose = float(input("实际血糖值: "))
            satisfaction = int(input("满意度评分 (1-5): "))

            await self.system.process_user_feedback(
                user_id, prediction_id, actual_glucose, satisfaction
            )

            print("✅ 反馈提交成功")

        except ValueError:
            print("❌ 输入格式错误")
        except Exception as e:
            print(f"❌ 反馈提交失败: {e}")

    async def interactive_recommend(self):
        """交互式推荐"""
        print("\n💡 获取个性化推荐")

        try:
            user_id = input("用户ID: ").strip() or "interactive_user"

            print("请输入用户档案 (直接回车使用默认值):")
            age = float(input("年龄 (默认: 45): ") or "45")
            bmi = float(input("BMI (默认: 25.5): ") or "25.5")
            exercise_level = int(input("运动水平 (1-5, 默认: 3): ") or "3")
            stress_level = int(input("压力水平 (1-5, 默认: 3): ") or "3")
            sleep_hours = float(input("睡眠时间/小时 (默认: 7): ") or "7")

            user_profile = {
                'age': age,
                'bmi': bmi,
                'exercise_level': exercise_level,
                'stress_level': stress_level,
                'sleep_hours': sleep_hours
            }

            recommendations = await self.system.get_personalized_recommendations(
                user_id, user_profile
            )

            if 'error' in recommendations:
                print(f"❌ 推荐失败: {recommendations['error']}")
            else:
                print(f"✅ 推荐成功:")
                print(f"   推荐内容: {recommendations.get('recommendation', 'N/A')}")
                print(f"   置信度: {recommendations.get('confidence', 0):.2%}")
                print(f"   个性化程度: {recommendations.get('personalization_level', 0):.2%}")

        except ValueError:
            print("❌ 输入格式错误")
        except Exception as e:
            print(f"❌ 推荐失败: {e}")

async def main():
    """主函数"""
    print("🎯 增强型血糖预测系统启动器")
    print("=" * 60)

    # 检查配置文件
    config_path = "enhanced_config.json"
    if not Path(config_path).exists():
        print(f"⚠️  配置文件不存在: {config_path}")
        print("使用默认配置启动...")
        config_path = None

    # 创建启动器
    launcher = SystemLauncher(config_path)

    # 启动系统
    if await launcher.start_system():
        print("\n🎉 系统启动成功!")

        # 询问是否进入交互模式
        try:
            choice = input("\n是否进入交互模式? (y/n, 默认: n): ").strip().lower()
            if choice in ['y', 'yes']:
                await launcher.interactive_mode()
            else:
                print("👋 系统启动完成，可以开始使用API接口")
        except KeyboardInterrupt:
            print("\n👋 退出系统")
    else:
        print("❌ 系统启动失败")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出系统")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        logger.error(f"启动失败: {e}")
        sys.exit(1)

__all__ = ["'project_root'", "'logger'", "'SystemLauncher'"]

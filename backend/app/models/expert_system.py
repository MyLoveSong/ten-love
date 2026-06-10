"""
混合专家决策系统和强化学习优化
实现多任务混合专家微调和强化学习膳食推荐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import random
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class RecommendationState:
    """推荐状态"""
    glucose_level: float
    meal_history: List[str]
    cultural_preference: Dict
    health_metrics: Dict
    time_of_day: int

class ExpertNetwork(nn.Module):
    """专家网络"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, expert_type: str):
        super().__init__()
        self.expert_type = expert_type

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class GatingNetwork(nn.Module):
    """门控网络"""

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

        self.gating = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gating(x)

class MixtureOfExperts(nn.Module):
    """混合专家模型"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 3
    ):
        super().__init__()
        self.num_experts = num_experts

        # 专家网络
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, output_dim, f"expert_{i}")
            for i in range(num_experts)
        ])

        # 门控网络
        self.gating_network = GatingNetwork(input_dim, num_experts)

        # 专家类型定义
        self.expert_types = [
            "glucose_control",  # 血糖控制专家
            "nutrition_balance",  # 营养均衡专家
            "cultural_adaptation"  # 文化适配专家
        ]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        混合专家前向传播
        Args:
            x: 输入特征 [batch_size, input_dim]
        Returns:
            输出和门控权重
        """
        # 计算门控权重
        gating_weights = self.gating_network(x)  # [batch_size, num_experts]

        # 计算专家输出
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)  # [batch_size, output_dim]
            expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]

        # 加权融合
        gating_weights = gating_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        final_output = torch.sum(expert_outputs * gating_weights, dim=1)  # [batch_size, output_dim]

        return final_output, gating_weights.squeeze(-1)

class ReinforcementLearningAgent(nn.Module):
    """强化学习智能体"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q网络
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

        # 目标Q网络
        self.target_q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

        # 复制权重到目标网络
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_network(state)

    def get_action(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """epsilon-贪婪策略选择动作"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return q_values.argmax().item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())

class MultiObjectiveReward:
    """多目标奖励函数"""

    def __init__(self):
        # 奖励权重
        self.weights = {
            'glucose_control': 0.4,  # 血糖控制
            'nutrition_balance': 0.3,  # 营养均衡
            'cultural_adaptation': 0.2,  # 文化适配
            'user_satisfaction': 0.1  # 用户满意度
        }

    def calculate_reward(
        self,
        state: RecommendationState,
        action: str,
        next_state: RecommendationState
    ) -> float:
        """
        计算多目标奖励
        Args:
            state: 当前状态
            action: 采取的动作（推荐的食物）
            next_state: 下一状态
        Returns:
            奖励值
        """
        total_reward = 0.0

        # 1. 血糖控制奖励
        glucose_reward = self._calculate_glucose_reward(
            state.glucose_level,
            next_state.glucose_level
        )

        # 2. 营养均衡奖励
        nutrition_reward = self._calculate_nutrition_reward(action, state.health_metrics)

        # 3. 文化适配奖励
        cultural_reward = self._calculate_cultural_reward(action, state.cultural_preference)

        # 4. 用户满意度奖励
        satisfaction_reward = self._calculate_satisfaction_reward(action, state.meal_history)

        # 加权求和
        total_reward = (
            self.weights['glucose_control'] * glucose_reward +
            self.weights['nutrition_balance'] * nutrition_reward +
            self.weights['cultural_adaptation'] * cultural_reward +
            self.weights['user_satisfaction'] * satisfaction_reward
        )

        return total_reward

    def _calculate_glucose_reward(self, current_glucose: float, next_glucose: float) -> float:
        """计算血糖控制奖励"""
        target_glucose = 5.5  # 目标血糖值 mmol/L

        current_deviation = abs(current_glucose - target_glucose)
        next_deviation = abs(next_glucose - target_glucose)

        # 血糖向目标值靠近给正奖励
        if next_deviation < current_deviation:
            return 1.0
        elif next_deviation > current_deviation:
            return -0.5
        else:
            return 0.0

    def _calculate_nutrition_reward(self, action: str, health_metrics: Dict) -> float:
        """计算营养均衡奖励"""
        # 简化的营养评分
        nutrition_score = 0.0

        # 基于食物类型给分
        if "蔬菜" in action:
            nutrition_score += 0.3
        if "蛋白质" in action:
            nutrition_score += 0.3
        if "全谷物" in action:
            nutrition_score += 0.2
        if "水果" in action:
            nutrition_score += 0.2

        return nutrition_score

    def _calculate_cultural_reward(self, action: str, cultural_preference: Dict) -> float:
        """计算文化适配奖励"""
        cultural_score = 0.0

        # 检查菜系匹配
        preferred_cuisine = cultural_preference.get('cuisine_type', '')
        if preferred_cuisine in action:
            cultural_score += 0.5

        # 检查口味偏好
        spice_tolerance = cultural_preference.get('spice_tolerance', 0.5)
        if spice_tolerance > 0.7 and "辣" in action:
            cultural_score += 0.3
        elif spice_tolerance < 0.3 and "辣" not in action:
            cultural_score += 0.3

        return cultural_score

    def _calculate_satisfaction_reward(self, action: str, meal_history: List[str]) -> float:
        """计算用户满意度奖励"""
        # 避免重复推荐
        recent_meals = meal_history[-5:]  # 最近5餐
        if action in recent_meals:
            return -0.2
        else:
            return 0.1

class MixedExpertReinforcementLearning:
    """混合专家强化学习系统"""

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 100,  # 100种可推荐的食物
        hidden_dim: int = 256
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 混合专家网络
        self.moe_network = MixtureOfExperts(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim
        ).to(self.device)

        # 强化学习智能体
        self.rl_agent = ReinforcementLearningAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # 多目标奖励函数
        self.reward_calculator = MultiObjectiveReward()

        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=10000)

        # 优化器
        self.moe_optimizer = torch.optim.Adam(self.moe_network.parameters(), lr=0.001)
        self.rl_optimizer = torch.optim.Adam(self.rl_agent.parameters(), lr=0.001)

        # 食物数据库
        self.food_database = self._create_food_database()

    def _create_food_database(self) -> List[str]:
        """创建食物数据库"""
        return [
            "清蒸鲈鱼", "宫保鸡丁", "麻婆豆腐", "红烧肉", "蒸蛋羹",
            "番茄鸡蛋", "青椒土豆丝", "糖醋排骨", "白切鸡", "凉拌黄瓜",
            "小白菜", "胡萝卜", "西兰花", "菠菜", "白萝卜",
            "苹果", "香蕉", "橙子", "葡萄", "草莓",
            # 更多食物...
        ] + [f"食物_{i}" for i in range(20, 100)]  # 填充到100种

    def encode_state(self, state: RecommendationState) -> torch.Tensor:
        """编码状态"""
        features = [
            state.glucose_level / 20.0,  # 归一化血糖值
            len(state.meal_history) / 10.0,  # 归一化餐食历史长度
            state.time_of_day / 24.0,  # 归一化时间
            state.cultural_preference.get('spice_tolerance', 0.5),
            state.cultural_preference.get('salt_preference', 0.5),
            state.health_metrics.get('bmi', 25) / 40.0,  # 归一化BMI
        ]

        # 填充到20维
        features.extend([0.0] * (20 - len(features)))

        return torch.tensor(features[:20], dtype=torch.float32).to(self.device)

    def get_recommendation(self, state: RecommendationState) -> Tuple[str, Dict]:
        """获取推荐"""
        state_tensor = self.encode_state(state).unsqueeze(0)

        # 混合专家预测
        moe_output, gating_weights = self.moe_network(state_tensor)

        # 强化学习动作选择
        rl_action = self.rl_agent.get_action(state_tensor)

        # 融合两种方法的结果
        moe_action = moe_output.argmax().item()

        # 选择最终动作（这里简化为RL动作）
        final_action = rl_action
        recommended_food = self.food_database[final_action]

        return recommended_food, {
            'moe_output': moe_output,
            'gating_weights': gating_weights,
            'rl_action': rl_action,
            'moe_action': moe_action,
            'expert_contributions': {
                'glucose_control': gating_weights[0, 0].item(),
                'nutrition_balance': gating_weights[0, 1].item(),
                'cultural_adaptation': gating_weights[0, 2].item()
            }
        }

    def train_step(self, batch_size: int = 32):
        """训练步骤"""
        if len(self.replay_buffer) < batch_size:
            return

        # 从经验回放中采样
        batch = random.sample(self.replay_buffer, batch_size)

        states = torch.stack([item[0] for item in batch])
        actions = torch.tensor([item[1] for item in batch])
        rewards = torch.tensor([item[2] for item in batch])
        next_states = torch.stack([item[3] for item in batch])
        dones = torch.tensor([item[4] for item in batch])

        # 训练强化学习智能体
        current_q_values = self.rl_agent(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.rl_agent.target_q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * (1 - dones.float()))

        rl_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.rl_optimizer.zero_grad()
        rl_loss.backward()
        self.rl_optimizer.step()

        # 训练混合专家网络（简化）
        moe_output, _ = self.moe_network(states)
        moe_loss = F.cross_entropy(moe_output, actions)

        self.moe_optimizer.zero_grad()
        moe_loss.backward()
        self.moe_optimizer.step()

    def add_experience(
        self,
        state: RecommendationState,
        action: int,
        reward: float,
        next_state: RecommendationState,
        done: bool
    ):
        """添加经验到回放缓冲区"""
        state_tensor = self.encode_state(state)
        next_state_tensor = self.encode_state(next_state)

        self.replay_buffer.append((
            state_tensor, action, reward, next_state_tensor, done
        ))

class ExpertSystemService:
    """专家系统服务"""

    def __init__(self):
        self.merl_system = MixedExpertReinforcementLearning()

    def get_personalized_recommendation(
        self,
        glucose_level: float,
        meal_history: List[str],
        cultural_preference: Dict,
        health_metrics: Dict,
        time_of_day: int = 12
    ) -> Dict:
        """获取个性化推荐"""

        # 构建状态
        state = RecommendationState(
            glucose_level=glucose_level,
            meal_history=meal_history,
            cultural_preference=cultural_preference,
            health_metrics=health_metrics,
            time_of_day=time_of_day
        )

        # 获取推荐
        recommended_food, recommendation_info = self.merl_system.get_recommendation(state)

        return {
            'recommended_food': recommended_food,
            'confidence_score': max(recommendation_info['expert_contributions'].values()),
            'expert_analysis': recommendation_info['expert_contributions'],
            'reasoning': self._generate_reasoning(recommendation_info),
            'alternative_options': self._get_alternative_options(state, 3)
        }

    def _generate_reasoning(self, recommendation_info: Dict) -> str:
        """生成推荐理由"""
        expert_contributions = recommendation_info['expert_contributions']

        dominant_expert = max(expert_contributions, key=expert_contributions.get)

        reasoning_map = {
            'glucose_control': "基于血糖控制需求，推荐低GI食物",
            'nutrition_balance': "基于营养均衡需求，推荐营养丰富的食物",
            'cultural_adaptation': "基于文化偏好，推荐符合您口味的食物"
        }

        return reasoning_map.get(dominant_expert, "综合考虑多个因素")

    def _get_alternative_options(self, state: RecommendationState, num_options: int) -> List[str]:
        """获取备选推荐"""
        alternatives = []
        for _ in range(num_options):
            food, _ = self.merl_system.get_recommendation(state)
            if food not in alternatives:
                alternatives.append(food)
        return alternatives

def create_expert_system_service() -> ExpertSystemService:
    """创建专家系统服务"""
    return ExpertSystemService()

if __name__ == "__main__":
    # 创建服务
    service = create_expert_system_service()

    # 测试推荐
    recommendation = service.get_personalized_recommendation(
        glucose_level=7.2,
        meal_history=["米饭", "鸡蛋"],
        cultural_preference={'cuisine_type': '川菜', 'spice_tolerance': 0.8},
        health_metrics={'bmi': 24.5}
    )

    print("个性化推荐结果:")
    print(f"推荐食物: {recommendation['recommended_food']}")
    print(f"置信度: {recommendation['confidence_score']:.2f}")
    print(f"推荐理由: {recommendation['reasoning']}")
    print(f"专家分析: {recommendation['expert_analysis']}")
    print("混合专家决策系统创建成功！")

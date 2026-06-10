# 数据库模块说明

## 概述

本数据库模块为学术级智能健康监测集成系统提供完整的数据持久化支持，支持SQLite开发环境和PostgreSQL生产环境。

## 功能特性

### 🗄️ 数据库支持
- **SQLite**: 开发环境，轻量级，无需额外配置
- **PostgreSQL**: 生产环境，功能强大，支持复杂查询
- **MySQL**: 可选支持，适用于特定部署需求

### 📊 数据模型
- **用户管理**: 用户信息、角色权限、登录记录
- **文化档案**: 用户文化偏好、饮食限制、口味特征
- **血糖预测**: 预测记录、反馈数据、模型性能
- **食物数据库**: 营养成分、文化标签、过敏信息
- **文化推荐**: 个性化推荐、用户满意度、营养分析
- **工作流管理**: 工作流定义、执行记录、性能指标
- **系统监控**: 指标收集、告警管理、性能统计
- **反馈系统**: 用户反馈、满意度评分、改进建议
- **实验管理**: 实验结果、模型对比、性能评估
- **数据增强**: 增强记录、质量指标、处理统计

### 🔄 迁移管理
- **版本控制**: 数据库架构版本管理
- **自动迁移**: 支持数据库结构升级
- **回滚支持**: 支持迁移回滚操作
- **一致性检查**: 自动检查架构一致性

### 🛠️ 管理功能
- **CRUD操作**: 完整的增删改查功能
- **复杂查询**: 支持多条件搜索和过滤
- **统计分析**: 内置统计分析和报表功能
- **数据清理**: 自动清理过期数据
- **备份恢复**: 支持数据库备份和恢复

## 快速开始

### 1. 安装依赖

```bash
pip install sqlalchemy alembic psycopg2-binary
```

### 2. 初始化数据库

```bash
# 运行数据库初始化脚本
python database/init_database.py
```

### 3. 运行示例

```bash
# 运行数据库功能演示
python database/example_usage.py
```

### 4. 启动系统

```bash
# 启动集成系统
python academic_integrated_system.py
```

## 配置说明

### 环境变量配置

创建 `.env` 文件并配置以下参数：

```bash
# 环境设置
ENVIRONMENT=development

# 数据库类型
DB_TYPE=sqlite

# SQLite配置 (开发环境)
DB_SQLITE_PATH=academic_system.db

# PostgreSQL配置 (生产环境)
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=academic_user
DB_PASSWORD=academic_password
DB_NAME=academic_system

# 数据库调试
DB_ECHO=false
```

### 配置切换

```python
from database.config import get_database_config

# 获取开发环境配置
dev_config = get_database_config("development")

# 获取生产环境配置
prod_config = get_database_config("production")

# 获取当前环境配置
current_config = get_database_config()
```

## API端点

### 数据库管理
- `GET /database/stats` - 获取数据库统计信息
- `GET /database/migration/status` - 获取迁移状态

### 用户管理
- `GET /users` - 获取用户列表
- `GET /users/{user_id}/predictions` - 获取用户预测历史
- `GET /users/{user_id}/cultural-profile` - 获取用户文化档案

### 食物管理
- `GET /food-items` - 搜索食物项目

### 系统监控
- `GET /stats/comprehensive` - 获取综合统计信息
- `GET /health` - 系统健康检查

## 使用示例

### 基本使用

```python
from database.models import initialize_database, SessionLocal
from database.database_manager import DatabaseManager

# 初始化数据库
engine, SessionLocal = initialize_database()

# 创建会话
session = SessionLocal()
db_manager = DatabaseManager(session)

try:
    # 创建用户
    user = db_manager.create_user(
        username="test_user",
        email="test@example.com",
        password_hash="hashed_password",
        role="patient"
    )

    # 创建血糖预测
    prediction = db_manager.create_glucose_prediction(
        user_id=user.user_id,
        predicted_glucose=120.0,
        confidence=0.85,
        model_type="GluFormer",
        model_version="1.0.0",
        input_features={"glucose_history": [100, 110, 120]}
    )

    # 查询用户预测历史
    predictions = db_manager.get_user_glucose_predictions(user.user_id)

finally:
    session.close()
```

### 高级查询

```python
# 搜索食物项目
food_items = db_manager.search_food_items(
    query="豆腐",
    category="主菜",
    is_vegetarian=True,
    max_gi=50
)

# 获取模型性能统计
performance = db_manager.get_model_performance_stats("GluFormer", days=30)

# 获取反馈统计
feedback_stats = db_manager.get_feedback_statistics(days=7)
```

## 数据模型关系

```
User (用户)
├── CulturalProfile (文化档案)
├── GlucosePrediction (血糖预测)
├── CulturalRecommendation (文化推荐)
├── FeedbackRecord (反馈记录)
├── WorkflowExecution (工作流执行)
├── ExperimentResult (实验结果)
└── DataAugmentationRecord (数据增强记录)

WorkflowDefinition (工作流定义)
└── WorkflowExecution (工作流执行)

SystemMetrics (系统指标)
Alert (告警记录)
FoodItem (食物项目)
```

## 迁移管理

### 查看迁移状态

```python
from database.migrations import MigrationManager

session = SessionLocal()
migration_manager = MigrationManager(session)

# 获取迁移状态
status = migration_manager.get_migration_status()
print(f"当前版本: {status['current_version']}")
print(f"待处理迁移: {status['pending_migrations']}")
```

### 手动运行迁移

```python
# 升级到最新版本
migration_manager.upgrade_database()

# 升级到指定版本
migration_manager.upgrade_database("1.1.0")
```

## 性能优化

### 索引优化
- 所有外键字段已建立索引
- 时间字段已建立索引
- 常用查询字段已建立复合索引

### 查询优化
- 使用连接查询减少数据库往返
- 实现分页查询避免大量数据加载
- 使用批量操作提高写入性能

### 连接池配置
```python
# PostgreSQL连接池配置
config = DatabaseConfig(
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=300
)
```

## 监控和维护

### 数据库统计
```python
# 获取数据库统计信息
stats = db_manager.get_database_stats()
print(f"用户数量: {stats['users']}")
print(f"预测记录: {stats['glucose_predictions']}")
print(f"活跃用户: {stats['active_users_30d']}")
```

### 数据清理
```python
# 清理90天前的旧数据
cleanup_result = db_manager.cleanup_old_data(days=90)
print(f"清理了 {cleanup_result['cleaned_metrics']} 条指标记录")
```

### 备份和恢复
```python
# 备份数据库
migration_manager.backup_database("backup_20240101.db")

# 恢复数据库
migration_manager.restore_database("backup_20240101.db")
```

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查数据库配置是否正确
   - 确认数据库服务是否运行
   - 验证用户权限

2. **迁移失败**
   - 检查数据库版本兼容性
   - 查看错误日志获取详细信息
   - 考虑手动执行迁移SQL

3. **性能问题**
   - 检查索引是否建立
   - 分析慢查询日志
   - 考虑增加连接池大小

### 日志配置

```python
import logging

# 配置数据库日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('database')
```

## 扩展开发

### 添加新表

1. 在 `models.py` 中定义新模型
2. 在 `database_manager.py` 中添加操作方法
3. 创建迁移脚本
4. 更新API端点

### 自定义查询

```python
# 在DatabaseManager中添加自定义方法
def get_custom_statistics(self, custom_params):
    query = self.session.query(CustomModel)
    # 自定义查询逻辑
    return query.all()
```

## 安全考虑

- 使用参数化查询防止SQL注入
- 实施适当的用户权限控制
- 定期备份重要数据
- 监控数据库访问日志
- 使用连接加密（生产环境）

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

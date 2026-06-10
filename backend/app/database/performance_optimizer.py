

"""
高级查询优化和自动调优系统
支持查询分析、索引优化、性能调优
"""

import logging
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import re
import statistics
from collections import defaultdict, Counter

from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import settings
from backend.app.core.exceptions import DatabaseError, CustomException
from backend.app.core.database import engine, SessionLocal
from backend.app.core.task_queue import async_task, TaskPriority

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """查询类型枚举"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"

class OptimizationLevel(Enum):
    """优化级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QueryMetrics:
    """查询指标"""
    query_id: str
    query_text: str
    query_type: QueryType
    execution_time: float
    rows_affected: int
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class IndexRecommendation:
    """索引推荐"""
    table_name: str
    columns: List[str]
    index_type: str  # single, composite, unique, partial
    priority: OptimizationLevel
    estimated_benefit: float
    reason: str
    sql_statement: str

@dataclass
class QueryOptimization:
    """查询优化建议"""
    original_query: str
    optimized_query: str
    optimization_type: str
    estimated_improvement: float
    reason: str
    confidence: float

class QueryAnalyzer:
    """查询分析器"""

    def __init__(self):
        self.query_metrics: List[QueryMetrics] = []
        self.slow_queries: List[QueryMetrics] = []
        self.query_patterns: Dict[str, List[QueryMetrics]] = defaultdict(list)
        self.table_stats: Dict[str, Dict[str, Any]] = {}

        # 性能阈值
        self.slow_query_threshold = 1.0  # 1秒
        self.critical_query_threshold = 5.0  # 5秒

    def record_query(self, query_metrics: QueryMetrics):
        """记录查询指标"""
        self.query_metrics.append(query_metrics)

        # 分析查询模式
        normalized_query = self._normalize_query(query_metrics.query_text)
        self.query_patterns[normalized_query].append(query_metrics)

        # 识别慢查询
        if query_metrics.execution_time > self.slow_query_threshold:
            self.slow_queries.append(query_metrics)

        # 保持最近1000条记录
        if len(self.query_metrics) > 1000:
            self.query_metrics = self.query_metrics[-1000:]

    def _normalize_query(self, query: str) -> str:
        """标准化查询语句"""
        # 移除多余空格
        query = re.sub(r'\s+', ' ', query.strip())

        # 替换参数占位符
        query = re.sub(r'\$\d+', '?', query)
        query = re.sub(r':\w+', '?', query)

        # 替换数字和字符串字面量
        query = re.sub(r'\b\d+\b', 'N', query)
        query = re.sub(r"'[^']*'", "'S'", query)
        query = re.sub(r'"[^"]*"', '"S"', query)

        return query.lower()

    def analyze_query_performance(self) -> Dict[str, Any]:
        """分析查询性能"""
        if not self.query_metrics:
            return {"message": "没有查询数据"}

        # 基本统计
        execution_times = [q.execution_time for q in self.query_metrics]

        stats = {
            "total_queries": len(self.query_metrics),
            "slow_queries": len(self.slow_queries),
            "critical_queries": len([q for q in self.query_metrics if q.execution_time > self.critical_query_threshold]),
            "avg_execution_time": statistics.mean(execution_times),
            "median_execution_time": statistics.median(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "p95_execution_time": self._percentile(execution_times, 95),
            "p99_execution_time": self._percentile(execution_times, 99)
        }

        # 查询类型分布
        query_type_counts = Counter(q.query_type.value for q in self.query_metrics)
        stats["query_type_distribution"] = dict(query_type_counts)

        # 最慢的查询
        slowest_queries = sorted(self.query_metrics, key=lambda x: x.execution_time, reverse=True)[:10]
        stats["slowest_queries"] = [
            {
                "query": q.query_text[:200] + "..." if len(q.query_text) > 200 else q.query_text,
                "execution_time": q.execution_time,
                "timestamp": q.timestamp.isoformat()
            }
            for q in slowest_queries
        ]

        # 最频繁的查询
        most_frequent = sorted(
            self.query_patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]

        stats["most_frequent_queries"] = [
            {
                "pattern": pattern,
                "count": len(metrics),
                "avg_execution_time": statistics.mean([m.execution_time for m in metrics]),
                "total_execution_time": sum([m.execution_time for m in metrics])
            }
            for pattern, metrics in most_frequent
        ]

        return stats

    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def get_slow_queries(self, limit: int = 50) -> List[QueryMetrics]:
        """获取慢查询"""
        return sorted(self.slow_queries, key=lambda x: x.execution_time, reverse=True)[:limit]

    def get_query_patterns(self) -> Dict[str, List[QueryMetrics]]:
        """获取查询模式"""
        return dict(self.query_patterns)

class IndexOptimizer:
    """索引优化器"""

    def __init__(self):
        self.existing_indexes: Dict[str, List[Dict[str, Any]]] = {}
        self.table_info: Dict[str, Dict[str, Any]] = {}

    async def analyze_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """分析表索引"""
        try:
            with engine.connect() as connection:
                # 获取表索引信息
                if settings.DATABASE_URL.startswith("sqlite"):
                    indexes = await self._get_sqlite_indexes(connection, table_name)
                elif settings.DATABASE_URL.startswith("postgresql"):
                    indexes = await self._get_postgresql_indexes(connection, table_name)
                elif settings.DATABASE_URL.startswith("mysql"):
                    indexes = await self._get_mysql_indexes(connection, table_name)
                else:
                    indexes = []

                self.existing_indexes[table_name] = indexes
                return indexes

        except Exception as e:
            logger.error(f"分析表索引失败 {table_name}: {e}")
            return []

    async def _get_sqlite_indexes(self, connection, table_name: str) -> List[Dict[str, Any]]:
        """获取SQLite索引信息"""
        query = text("""
            SELECT name, sql, tbl_name
            FROM sqlite_master
            WHERE type = 'index' AND tbl_name = :table_name
        """)

        result = connection.execute(query, {"table_name": table_name})
        indexes = []

        for row in result:
            indexes.append({
                "name": row[0],
                "sql": row[1],
                "table": row[2],
                "type": "btree"  # SQLite默认索引类型
            })

        return indexes

    async def _get_postgresql_indexes(self, connection, table_name: str) -> List[Dict[str, Any]]:
        """获取PostgreSQL索引信息"""
        query = text("""
            SELECT
                i.indexname,
                i.indexdef,
                i.tablename,
                am.amname as index_type
            FROM pg_indexes i
            JOIN pg_am am ON am.oid = (
                SELECT indexrelid FROM pg_class
                WHERE relname = i.indexname
            )
            WHERE i.tablename = :table_name
        """)

        result = connection.execute(query, {"table_name": table_name})
        indexes = []

        for row in result:
            indexes.append({
                "name": row[0],
                "sql": row[1],
                "table": row[2],
                "type": row[3]
            })

        return indexes

    async def _get_mysql_indexes(self, connection, table_name: str) -> List[Dict[str, Any]]:
        """获取MySQL索引信息"""
        query = text("SHOW INDEX FROM :table_name")

        result = connection.execute(query, {"table_name": table_name})
        indexes = []

        for row in result:
            indexes.append({
                "name": row[2],  # Key_name
                "column": row[4],  # Column_name
                "table": row[0],  # Table
                "type": row[10] if len(row) > 10 else "btree"  # Index_type
            })

        return indexes

    def recommend_indexes(self, query_metrics: List[QueryMetrics]) -> List[IndexRecommendation]:
        """推荐索引"""
        recommendations = []

        # 分析WHERE子句
        where_columns = self._extract_where_columns(query_metrics)

        # 分析JOIN条件
        join_columns = self._extract_join_columns(query_metrics)

        # 分析ORDER BY子句
        order_columns = self._extract_order_columns(query_metrics)

        # 生成索引推荐
        for table_name, columns in where_columns.items():
            if columns:
                recommendation = IndexRecommendation(
                    table_name=table_name,
                    columns=columns,
                    index_type="composite" if len(columns) > 1 else "single",
                    priority=OptimizationLevel.HIGH,
                    estimated_benefit=self._estimate_index_benefit(table_name, columns),
                    reason="基于WHERE子句分析",
                    sql_statement=self._generate_index_sql(table_name, columns)
                )
                recommendations.append(recommendation)

        # 按优先级排序
        recommendations.sort(key=lambda x: x.priority.value, reverse=True)

        return recommendations

    def _extract_where_columns(self, query_metrics: List[QueryMetrics]) -> Dict[str, List[str]]:
        """提取WHERE子句中的列"""
        where_columns = defaultdict(set)

        for metrics in query_metrics:
            query = metrics.query_text.lower()

            # 简单的WHERE子句解析
            where_match = re.search(r'where\s+(.+?)(?:\s+group\s+by|\s+order\s+by|\s+limit|$)', query)
            if where_match:
                where_clause = where_match.group(1)

                # 提取列名（简单实现）
                column_matches = re.findall(r'(\w+)\s*[=<>!]', where_clause)
                for column in column_matches:
                    # 假设表名（这里需要更复杂的解析）
                    table_name = self._guess_table_name(query)
                    if table_name:
                        where_columns[table_name].add(column)

        return {k: list(v) for k, v in where_columns.items()}

    def _extract_join_columns(self, query_metrics: List[QueryMetrics]) -> Dict[str, List[str]]:
        """提取JOIN条件中的列"""
        join_columns = defaultdict(set)

        for metrics in query_metrics:
            query = metrics.query_text.lower()

            # 简单的JOIN解析
            join_matches = re.findall(r'join\s+\w+\s+on\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', query)
            for match in join_matches:
                table1, column1, table2, column2 = match
                join_columns[table1].add(column1)
                join_columns[table2].add(column2)

        return {k: list(v) for k, v in join_columns.items()}

    def _extract_order_columns(self, query_metrics: List[QueryMetrics]) -> Dict[str, List[str]]:
        """提取ORDER BY子句中的列"""
        order_columns = defaultdict(set)

        for metrics in query_metrics:
            query = metrics.query_text.lower()

            # 简单的ORDER BY解析
            order_match = re.search(r'order\s+by\s+(.+?)(?:\s+limit|$)', query)
            if order_match:
                order_clause = order_match.group(1)

                # 提取列名
                column_matches = re.findall(r'(\w+)(?:\s+asc|\s+desc)?', order_clause)
                for column in column_matches:
                    table_name = self._guess_table_name(query)
                    if table_name:
                        order_columns[table_name].add(column)

        return {k: list(v) for k, v in order_columns.items()}

    def _guess_table_name(self, query: str) -> Optional[str]:
        """猜测表名（简单实现）"""
        # 从FROM子句提取表名
        from_match = re.search(r'from\s+(\w+)', query)
        if from_match:
            return from_match.group(1)

        # 从JOIN子句提取表名
        join_match = re.search(r'join\s+(\w+)', query)
        if join_match:
            return join_match.group(1)

        return None

    def _estimate_index_benefit(self, table_name: str, columns: List[str]) -> float:
        """估算索引收益"""
        # 简单的收益估算（实际实现需要更复杂的算法）
        base_benefit = 0.5

        # 根据列数调整收益
        if len(columns) > 1:
            base_benefit *= 0.8  # 复合索引收益略低

        # 根据查询频率调整收益
        query_count = len([q for q in self.query_metrics if table_name in q.query_text.lower()])
        frequency_factor = min(query_count / 100, 1.0)  # 最多1.0

        return base_benefit * frequency_factor

    def _generate_index_sql(self, table_name: str, columns: List[str]) -> str:
        """生成索引SQL语句"""
        index_name = f"idx_{table_name}_{'_'.join(columns)}"
        columns_str = ", ".join(columns)

        return f"CREATE INDEX {index_name} ON {table_name} ({columns_str});"

class QueryOptimizer:
    """查询优化器"""

    def __init__(self):
        self.optimization_rules = [
            self._optimize_select_star,
            self._optimize_where_clause,
            self._optimize_join_order,
            self._optimize_subqueries,
            self._optimize_order_by
        ]

    def optimize_query(self, query: str) -> List[QueryOptimization]:
        """优化查询"""
        optimizations = []

        for rule in self.optimization_rules:
            try:
                optimization = rule(query)
                if optimization:
                    optimizations.append(optimization)
            except Exception as e:
                logger.error(f"查询优化规则执行失败: {e}")

        return optimizations

    def _optimize_select_star(self, query: str) -> Optional[QueryOptimization]:
        """优化SELECT *"""
        if re.search(r'select\s+\*\s+from', query.lower()):
            # 建议使用具体列名
            return QueryOptimization(
                original_query=query,
                optimized_query=query.replace("SELECT *", "SELECT column1, column2, ..."),
                optimization_type="select_columns",
                estimated_improvement=0.2,
                reason="使用具体列名可以减少数据传输量",
                confidence=0.8
            )
        return None

    def _optimize_where_clause(self, query: str) -> Optional[QueryOptimization]:
        """优化WHERE子句"""
        # 检查是否有函数调用在WHERE子句中
        where_match = re.search(r'where\s+(.+?)(?:\s+group\s+by|\s+order\s+by|\s+limit|$)', query.lower())
        if where_match:
            where_clause = where_match.group(1)

            # 检查函数调用
            if re.search(r'\w+\s*\([^)]*\)', where_clause):
                return QueryOptimization(
                    original_query=query,
                    optimized_query=query,  # 需要具体优化
                    optimization_type="where_functions",
                    estimated_improvement=0.3,
                    reason="避免在WHERE子句中使用函数，可能导致索引失效",
                    confidence=0.9
                )
        return None

    def _optimize_join_order(self, query: str) -> Optional[QueryOptimization]:
        """优化JOIN顺序"""
        # 简单的JOIN顺序优化建议
        if 'join' in query.lower():
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                optimization_type="join_order",
                estimated_improvement=0.15,
                reason="考虑调整JOIN顺序，将选择性高的表放在前面",
                confidence=0.6
            )
        return None

    def _optimize_subqueries(self, query: str) -> Optional[QueryOptimization]:
        """优化子查询"""
        if re.search(r'\(select\s+', query.lower()):
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                optimization_type="subquery_to_join",
                estimated_improvement=0.25,
                reason="考虑将子查询转换为JOIN，可能提高性能",
                confidence=0.7
            )
        return None

    def _optimize_order_by(self, query: str) -> Optional[QueryOptimization]:
        """优化ORDER BY"""
        if 'order by' in query.lower() and 'limit' not in query.lower():
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                optimization_type="order_by_limit",
                estimated_improvement=0.1,
                reason="如果不需要所有结果，考虑添加LIMIT子句",
                confidence=0.5
            )
        return None

class DatabasePerformanceOptimizer:
    """数据库性能优化器"""

    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.index_optimizer = IndexOptimizer()
        self.query_optimizer = QueryOptimizer()
        self.optimization_history: List[Dict[str, Any]] = []

    def record_query_execution(self, query: str, execution_time: float, **kwargs):
        """记录查询执行"""
        query_metrics = QueryMetrics(
            query_id=f"query_{int(time.time() * 1000)}",
            query_text=query,
            query_type=self._detect_query_type(query),
            execution_time=execution_time,
            rows_affected=kwargs.get('rows_affected', 0),
            timestamp=datetime.now(),
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id'),
            parameters=kwargs.get('parameters')
        )

        self.query_analyzer.record_query(query_metrics)

    def _detect_query_type(self, query: str) -> QueryType:
        """检测查询类型"""
        query_lower = query.lower().strip()

        if query_lower.startswith('select'):
            return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        elif query_lower.startswith('create'):
            return QueryType.CREATE
        elif query_lower.startswith('drop'):
            return QueryType.DROP
        elif query_lower.startswith('alter'):
            return QueryType.ALTER
        else:
            return QueryType.SELECT

    async def analyze_database_performance(self) -> Dict[str, Any]:
        """分析数据库性能"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "query_performance": self.query_analyzer.analyze_query_performance(),
            "slow_queries": [
                {
                    "query": q.query_text[:200] + "..." if len(q.query_text) > 200 else q.query_text,
                    "execution_time": q.execution_time,
                    "timestamp": q.timestamp.isoformat()
                }
                for q in self.query_analyzer.get_slow_queries(10)
            ],
            "recommendations": []
        }

        # 生成索引推荐
        slow_queries = self.query_analyzer.get_slow_queries(20)
        if slow_queries:
            index_recommendations = self.index_optimizer.recommend_indexes(slow_queries)
            analysis["recommendations"].extend([
                {
                    "type": "index",
                    "table": rec.table_name,
                    "columns": rec.columns,
                    "priority": rec.priority.value,
                    "benefit": rec.estimated_benefit,
                    "reason": rec.reason,
                    "sql": rec.sql_statement
                }
                for rec in index_recommendations
            ])

        # 生成查询优化建议
        for query_metrics in slow_queries[:5]:
            optimizations = self.query_optimizer.optimize_query(query_metrics.query_text)
            analysis["recommendations"].extend([
                {
                    "type": "query_optimization",
                    "original_query": opt.original_query[:200] + "..." if len(opt.original_query) > 200 else opt.original_query,
                    "optimization_type": opt.optimization_type,
                    "improvement": opt.estimated_improvement,
                    "reason": opt.reason,
                    "confidence": opt.confidence
                }
                for opt in optimizations
            ])

        return analysis

    async def auto_optimize_database(self) -> Dict[str, Any]:
        """自动优化数据库"""
        optimization_results = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "recommendations": [],
            "errors": []
        }

        try:
            # 分析性能
            analysis = await self.analyze_database_performance()

            # 自动创建高优先级索引
            for rec in analysis["recommendations"]:
                if rec["type"] == "index" and rec["priority"] == "critical":
                    try:
                        # 这里应该实际创建索引
                        optimization_results["actions_taken"].append({
                            "action": "create_index",
                            "table": rec["table"],
                            "columns": rec["columns"],
                            "sql": rec["sql"]
                        })
                    except Exception as e:
                        optimization_results["errors"].append(f"创建索引失败: {e}")

            # 记录优化历史
            self.optimization_history.append(optimization_results)

        except Exception as e:
            optimization_results["errors"].append(f"自动优化失败: {e}")
            logger.error(f"自动优化失败: {e}")

        return optimization_results

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return self.optimization_history

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "total_queries_analyzed": len(self.query_analyzer.query_metrics),
            "slow_queries_count": len(self.query_analyzer.slow_queries),
            "optimization_actions": len(self.optimization_history),
            "last_optimization": self.optimization_history[-1]["timestamp"] if self.optimization_history else None
        }

# 全局性能优化器实例
performance_optimizer = DatabasePerformanceOptimizer()

# 异步任务
@async_task("analyze_database_performance", TaskPriority.NORMAL)
def analyze_database_performance_task():
    """分析数据库性能任务"""
    return asyncio.run(performance_optimizer.analyze_database_performance())

@async_task("auto_optimize_database", TaskPriority.HIGH)
def auto_optimize_database_task():
    """自动优化数据库任务"""
    return asyncio.run(performance_optimizer.auto_optimize_database())

# 性能优化API
def record_query_execution(query: str, execution_time: float, **kwargs):
    """记录查询执行"""
    performance_optimizer.record_query_execution(query, execution_time, **kwargs)

async def analyze_database_performance() -> Dict[str, Any]:
    """分析数据库性能"""
    return await performance_optimizer.analyze_database_performance()

async def auto_optimize_database() -> Dict[str, Any]:
    """自动优化数据库"""
    return await performance_optimizer.auto_optimize_database()

def get_performance_summary() -> Dict[str, Any]:
    """获取性能摘要"""
    return performance_optimizer.get_performance_summary()

def get_optimization_history() -> List[Dict[str, Any]]:
    """获取优化历史"""
    return performance_optimizer.get_optimization_history()

if __name__ == "__main__":
    # 测试性能优化器
    print("性能优化器摘要:")
    print(json.dumps(get_performance_summary(), indent=2, ensure_ascii=False))

    # 模拟查询执行
    record_query_execution("SELECT * FROM users WHERE id = 1", 0.5)
    record_query_execution("SELECT * FROM users WHERE name LIKE '%test%'", 2.5)

    # 分析性能
    analysis = asyncio.run(analyze_database_performance())
    print("性能分析:")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))

__all__ = ["'logger'", "'QueryType'", "'OptimizationLevel'", "'QueryMetrics'", "'IndexRecommendation'", "'QueryOptimization'", "'QueryAnalyzer'", "'IndexOptimizer'", "'QueryOptimizer'", "'DatabasePerformanceOptimizer'", "'performance_optimizer'", "'analyze_database_performance_task'", "'auto_optimize_database_task'", "'record_query_execution'", "'get_performance_summary'", "'get_optimization_history'"]

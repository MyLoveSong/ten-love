

"""
数据库备份和灾难恢复系统
支持全量备份、增量备份和自动恢复
"""

import os
import shutil
import gzip
import json
import logging
import asyncio
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import threading
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    schedule = None
import time

from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.core.exceptions import DatabaseError, CustomException
from backend.app.core.task_queue import async_task, TaskPriority

logger = logging.getLogger(__name__)

class BackupType(Enum):
    """备份类型枚举"""
    FULL = "full"  # 全量备份
    INCREMENTAL = "incremental"  # 增量备份
    DIFFERENTIAL = "differential"  # 差异备份

class BackupStatus(Enum):
    """备份状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class BackupInfo:
    """备份信息"""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class BackupConfig:
    """备份配置"""
    backup_dir: str
    retention_days: int = 30
    compression: bool = True
    encryption: bool = False
    encryption_key: Optional[str] = None
    max_backup_size: int = 1024 * 1024 * 1024  # 1GB
    parallel_backups: int = 1
    verify_backups: bool = True
    auto_cleanup: bool = True

class DatabaseBackupManager:
    """数据库备份管理器"""

    def __init__(self, config: BackupConfig):
        self.config = config
        self.backups: Dict[str, BackupInfo] = {}
        self.backup_lock = threading.Lock()
        self.is_backing_up = False

        # 确保备份目录存在
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)

        # 加载现有备份信息
        self._load_backup_info()

        # 启动自动清理任务
        if self.config.auto_cleanup:
            self._start_cleanup_scheduler()

    def _load_backup_info(self):
        """加载现有备份信息"""
        backup_info_file = Path(self.config.backup_dir) / "backup_info.json"

        if backup_info_file.exists():
            try:
                with open(backup_info_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for backup_id, backup_data in data.items():
                    backup_info = BackupInfo(
                        backup_id=backup_data["backup_id"],
                        backup_type=BackupType(backup_data["backup_type"]),
                        status=BackupStatus(backup_data["status"]),
                        created_at=datetime.fromisoformat(backup_data["created_at"]),
                        started_at=datetime.fromisoformat(backup_data["started_at"]) if backup_data.get("started_at") else None,
                        completed_at=datetime.fromisoformat(backup_data["completed_at"]) if backup_data.get("completed_at") else None,
                        file_path=backup_data.get("file_path"),
                        file_size=backup_data.get("file_size"),
                        checksum=backup_data.get("checksum"),
                        metadata=backup_data.get("metadata"),
                        error_message=backup_data.get("error_message")
                    )
                    self.backups[backup_id] = backup_info

                logger.info(f"加载了 {len(self.backups)} 个备份记录")

            except Exception as e:
                logger.error(f"加载备份信息失败: {e}")

    def _save_backup_info(self):
        """保存备份信息"""
        backup_info_file = Path(self.config.backup_dir) / "backup_info.json"

        try:
            data = {}
            for backup_id, backup_info in self.backups.items():
                data[backup_id] = {
                    "backup_id": backup_info.backup_id,
                    "backup_type": backup_info.backup_type.value,
                    "status": backup_info.status.value,
                    "created_at": backup_info.created_at.isoformat(),
                    "started_at": backup_info.started_at.isoformat() if backup_info.started_at else None,
                    "completed_at": backup_info.completed_at.isoformat() if backup_info.completed_at else None,
                    "file_path": backup_info.file_path,
                    "file_size": backup_info.file_size,
                    "checksum": backup_info.checksum,
                    "metadata": backup_info.metadata,
                    "error_message": backup_info.error_message
                }

            with open(backup_info_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"保存备份信息失败: {e}")

    def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建备份"""
        if self.is_backing_up:
            raise CustomException("备份正在进行中，请稍后再试")

        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_info = BackupInfo(
            backup_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {}
        )

        with self.backup_lock:
            self.backups[backup_id] = backup_info
            self.is_backing_up = True

        # 异步执行备份
        asyncio.create_task(self._execute_backup(backup_info))

        logger.info(f"创建备份任务: {backup_id}")
        return backup_id

    async def _execute_backup(self, backup_info: BackupInfo):
        """执行备份"""
        try:
            backup_info.status = BackupStatus.RUNNING
            backup_info.started_at = datetime.now()
            self._save_backup_info()

            logger.info(f"开始备份: {backup_info.backup_id}")

            # 根据数据库类型选择备份方法
            if settings.DATABASE_URL.startswith("sqlite"):
                await self._backup_sqlite(backup_info)
            elif settings.DATABASE_URL.startswith("postgresql"):
                await self._backup_postgresql(backup_info)
            elif settings.DATABASE_URL.startswith("mysql"):
                await self._backup_mysql(backup_info)
            else:
                raise CustomException(f"不支持的数据库类型: {settings.DATABASE_URL}")

            # 验证备份
            if self.config.verify_backups:
                await self._verify_backup(backup_info)

            backup_info.status = BackupStatus.COMPLETED
            backup_info.completed_at = datetime.now()

            logger.info(f"备份完成: {backup_info.backup_id}")

        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            logger.error(f"备份失败: {backup_info.backup_id} - {e}")

        finally:
            self.is_backing_up = False
            self._save_backup_info()

    async def _backup_sqlite(self, backup_info: BackupInfo):
        """备份SQLite数据库"""
        db_path = settings.DATABASE_URL.replace("sqlite:///", "")

        if not os.path.exists(db_path):
            raise CustomException(f"数据库文件不存在: {db_path}")

        # 生成备份文件名
        backup_filename = f"{backup_info.backup_id}.db"
        if self.config.compression:
            backup_filename += ".gz"

        backup_path = Path(self.config.backup_dir) / backup_filename

        # 复制数据库文件
        shutil.copy2(db_path, backup_path)

        # 压缩备份文件
        if self.config.compression:
            compressed_path = backup_path.with_suffix(backup_path.suffix + ".gz")
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(backup_path)
            backup_path = compressed_path

        # 更新备份信息
        backup_info.file_path = str(backup_path)
        backup_info.file_size = backup_path.stat().st_size
        backup_info.checksum = self._calculate_checksum(backup_path)

    async def _backup_postgresql(self, backup_info: BackupInfo):
        """备份PostgreSQL数据库"""
        # 使用pg_dump进行备份
        backup_filename = f"{backup_info.backup_id}.sql"
        if self.config.compression:
            backup_filename += ".gz"

        backup_path = Path(self.config.backup_dir) / backup_filename

        # 构建pg_dump命令
        cmd = ["pg_dump", "--verbose", "--no-password"]

        # 添加连接参数
        if "://" in settings.DATABASE_URL:
            # 解析URL
            url_parts = settings.DATABASE_URL.split("://")
            if len(url_parts) == 2:
                auth_host = url_parts[1]
                if "@" in auth_host:
                    auth, host_db = auth_host.split("@", 1)
                    if ":" in auth:
                        username, password = auth.split(":", 1)
                        cmd.extend(["-U", username])
                        os.environ["PGPASSWORD"] = password

                if "/" in host_db:
                    host_port, database = host_db.split("/", 1)
                    if ":" in host_port:
                        host, port = host_port.split(":", 1)
                        cmd.extend(["-h", host, "-p", port])
                    else:
                        cmd.extend(["-h", host_port])
                    cmd.extend(["-d", database])

        # 执行备份
        try:
            with open(backup_path, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                raise CustomException(f"pg_dump失败: {result.stderr}")

            # 压缩备份文件
            if self.config.compression:
                compressed_path = backup_path.with_suffix(backup_path.suffix + ".gz")
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(backup_path)
                backup_path = compressed_path

            # 更新备份信息
            backup_info.file_path = str(backup_path)
            backup_info.file_size = backup_path.stat().st_size
            backup_info.checksum = self._calculate_checksum(backup_path)

        except FileNotFoundError:
            raise CustomException("pg_dump命令未找到，请确保PostgreSQL客户端已安装")
        except Exception as e:
            raise CustomException(f"PostgreSQL备份失败: {e}")

    async def _backup_mysql(self, backup_info: BackupInfo):
        """备份MySQL数据库"""
        # 使用mysqldump进行备份
        backup_filename = f"{backup_info.backup_id}.sql"
        if self.config.compression:
            backup_filename += ".gz"

        backup_path = Path(self.config.backup_dir) / backup_filename

        # 构建mysqldump命令
        cmd = ["mysqldump", "--single-transaction", "--routines", "--triggers"]

        # 添加连接参数
        if "://" in settings.DATABASE_URL:
            # 解析URL
            url_parts = settings.DATABASE_URL.split("://")
            if len(url_parts) == 2:
                auth_host = url_parts[1]
                if "@" in auth_host:
                    auth, host_db = auth_host.split("@", 1)
                    if ":" in auth:
                        username, password = auth.split(":", 1)
                        cmd.extend(["-u", username, f"-p{password}"])

                if "/" in host_db:
                    host_port, database = host_db.split("/", 1)
                    if ":" in host_port:
                        host, port = host_port.split(":", 1)
                        cmd.extend(["-h", host, "-P", port])
                    else:
                        cmd.extend(["-h", host_port])
                    cmd.extend([database])

        # 执行备份
        try:
            with open(backup_path, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                raise CustomException(f"mysqldump失败: {result.stderr}")

            # 压缩备份文件
            if self.config.compression:
                compressed_path = backup_path.with_suffix(backup_path.suffix + ".gz")
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(backup_path)
                backup_path = compressed_path

            # 更新备份信息
            backup_info.file_path = str(backup_path)
            backup_info.file_size = backup_path.stat().st_size
            backup_info.checksum = self._calculate_checksum(backup_path)

        except FileNotFoundError:
            raise CustomException("mysqldump命令未找到，请确保MySQL客户端已安装")
        except Exception as e:
            raise CustomException(f"MySQL备份失败: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    async def _verify_backup(self, backup_info: BackupInfo):
        """验证备份文件"""
        if not backup_info.file_path or not os.path.exists(backup_info.file_path):
            raise CustomException("备份文件不存在")

        # 验证文件大小
        file_size = os.path.getsize(backup_info.file_path)
        if file_size == 0:
            raise CustomException("备份文件为空")

        # 验证校验和
        if backup_info.checksum:
            current_checksum = self._calculate_checksum(Path(backup_info.file_path))
            if current_checksum != backup_info.checksum:
                raise CustomException("备份文件校验和不匹配")

        logger.info(f"备份验证通过: {backup_info.backup_id}")

    async def restore_backup(self, backup_id: str, target_database_url: Optional[str] = None) -> bool:
        """恢复备份"""
        if backup_id not in self.backups:
            raise CustomException(f"备份不存在: {backup_id}")

        backup_info = self.backups[backup_id]

        if backup_info.status != BackupStatus.COMPLETED:
            raise CustomException(f"备份未完成: {backup_id}")

        if not backup_info.file_path or not os.path.exists(backup_info.file_path):
            raise CustomException(f"备份文件不存在: {backup_info.file_path}")

        target_url = target_database_url or settings.DATABASE_URL

        try:
            logger.info(f"开始恢复备份: {backup_id}")

            # 根据数据库类型选择恢复方法
            if target_url.startswith("sqlite"):
                await self._restore_sqlite(backup_info, target_url)
            elif target_url.startswith("postgresql"):
                await self._restore_postgresql(backup_info, target_url)
            elif target_url.startswith("mysql"):
                await self._restore_mysql(backup_info, target_url)
            else:
                raise CustomException(f"不支持的数据库类型: {target_url}")

            logger.info(f"备份恢复完成: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"备份恢复失败: {backup_id} - {e}")
            raise CustomException(f"备份恢复失败: {e}")

    async def _restore_sqlite(self, backup_info: BackupInfo, target_url: str):
        """恢复SQLite备份"""
        target_path = target_url.replace("sqlite:///", "")

        # 解压缩备份文件
        if backup_info.file_path.endswith(".gz"):
            temp_path = backup_info.file_path[:-3]  # 移除.gz后缀
            with gzip.open(backup_info.file_path, 'rb') as f_in:
                with open(temp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            restore_path = temp_path
        else:
            restore_path = backup_info.file_path

        # 复制备份文件到目标位置
        shutil.copy2(restore_path, target_path)

        # 清理临时文件
        if restore_path != backup_info.file_path:
            os.remove(restore_path)

    async def _restore_postgresql(self, backup_info: BackupInfo, target_url: str):
        """恢复PostgreSQL备份"""
        # 构建psql命令
        cmd = ["psql", "--quiet", "--no-password"]

        # 添加连接参数
        if "://" in target_url:
            url_parts = target_url.split("://")
            if len(url_parts) == 2:
                auth_host = url_parts[1]
                if "@" in auth_host:
                    auth, host_db = auth_host.split("@", 1)
                    if ":" in auth:
                        username, password = auth.split(":", 1)
                        cmd.extend(["-U", username])
                        os.environ["PGPASSWORD"] = password

                if "/" in host_db:
                    host_port, database = host_db.split("/", 1)
                    if ":" in host_port:
                        host, port = host_port.split(":", 1)
                        cmd.extend(["-h", host, "-p", port])
                    else:
                        cmd.extend(["-h", host_port])
                    cmd.extend(["-d", database])

        # 解压缩备份文件
        if backup_info.file_path.endswith(".gz"):
            temp_path = backup_info.file_path[:-3]
            with gzip.open(backup_info.file_path, 'rb') as f_in:
                with open(temp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            restore_path = temp_path
        else:
            restore_path = backup_info.file_path

        # 执行恢复
        try:
            with open(restore_path, 'r') as f:
                result = subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                raise CustomException(f"psql恢复失败: {result.stderr}")

            # 清理临时文件
            if restore_path != backup_info.file_path:
                os.remove(restore_path)

        except FileNotFoundError:
            raise CustomException("psql命令未找到，请确保PostgreSQL客户端已安装")

    async def _restore_mysql(self, backup_info: BackupInfo, target_url: str):
        """恢复MySQL备份"""
        # 构建mysql命令
        cmd = ["mysql"]

        # 添加连接参数
        if "://" in target_url:
            url_parts = target_url.split("://")
            if len(url_parts) == 2:
                auth_host = url_parts[1]
                if "@" in auth_host:
                    auth, host_db = auth_host.split("@", 1)
                    if ":" in auth:
                        username, password = auth.split(":", 1)
                        cmd.extend(["-u", username, f"-p{password}"])

                if "/" in host_db:
                    host_port, database = host_db.split("/", 1)
                    if ":" in host_port:
                        host, port = host_port.split(":", 1)
                        cmd.extend(["-h", host, "-P", port])
                    else:
                        cmd.extend(["-h", host_port])
                    cmd.extend([database])

        # 解压缩备份文件
        if backup_info.file_path.endswith(".gz"):
            temp_path = backup_info.file_path[:-3]
            with gzip.open(backup_info.file_path, 'rb') as f_in:
                with open(temp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            restore_path = temp_path
        else:
            restore_path = backup_info.file_path

        # 执行恢复
        try:
            with open(restore_path, 'r') as f:
                result = subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                raise CustomException(f"mysql恢复失败: {result.stderr}")

            # 清理临时文件
            if restore_path != backup_info.file_path:
                os.remove(restore_path)

        except FileNotFoundError:
            raise CustomException("mysql命令未找到，请确保MySQL客户端已安装")

    def get_backup_list(self) -> List[BackupInfo]:
        """获取备份列表"""
        return list(self.backups.values())

    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """获取备份信息"""
        return self.backups.get(backup_id)

    def delete_backup(self, backup_id: str) -> bool:
        """删除备份"""
        if backup_id not in self.backups:
            return False

        backup_info = self.backups[backup_id]

        # 删除备份文件
        if backup_info.file_path and os.path.exists(backup_info.file_path):
            os.remove(backup_info.file_path)

        # 删除备份记录
        del self.backups[backup_id]
        self._save_backup_info()

        logger.info(f"删除备份: {backup_id}")
        return True

    def _start_cleanup_scheduler(self):
        """启动清理调度器"""
        if not SCHEDULE_AVAILABLE:
            logger.warning("schedule模块未安装，跳过自动清理调度")
            return

        def cleanup_job():
            asyncio.create_task(self.cleanup_old_backups())

        # 每天凌晨2点执行清理
        schedule.every().day.at("02:00").do(cleanup_job)

        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)

        thread = threading.Thread(target=run_scheduler, daemon=True)
        thread.start()

        logger.info("备份清理调度器已启动")

    async def cleanup_old_backups(self):
        """清理旧备份"""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)

        backups_to_delete = [
            backup_id for backup_id, backup_info in self.backups.items()
            if backup_info.created_at < cutoff_date
        ]

        for backup_id in backups_to_delete:
            self.delete_backup(backup_id)

        logger.info(f"清理了 {len(backups_to_delete)} 个过期备份")
        return len(backups_to_delete)

    def get_backup_stats(self) -> Dict[str, Any]:
        """获取备份统计信息"""
        total_backups = len(self.backups)
        completed_backups = len([b for b in self.backups.values() if b.status == BackupStatus.COMPLETED])
        failed_backups = len([b for b in self.backups.values() if b.status == BackupStatus.FAILED])

        total_size = sum(
            b.file_size for b in self.backups.values()
            if b.file_size is not None
        )

        return {
            "total_backups": total_backups,
            "completed_backups": completed_backups,
            "failed_backups": failed_backups,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024) if total_size else 0,
            "is_backing_up": self.is_backing_up,
            "retention_days": self.config.retention_days
        }

# 全局备份管理器实例
backup_config = BackupConfig(
    backup_dir="./backups",
    retention_days=30,
    compression=True,
    verify_backups=True,
    auto_cleanup=True
)

backup_manager = DatabaseBackupManager(backup_config)

# 异步任务
@async_task("create_database_backup", TaskPriority.NORMAL)
def create_database_backup_task(backup_type: str = "full", metadata: Dict[str, Any] = None):
    """创建数据库备份任务"""
    backup_type_enum = BackupType(backup_type)
    backup_id = backup_manager.create_backup(backup_type_enum, metadata)
    return {"backup_id": backup_id, "status": "started"}

@async_task("restore_database_backup", TaskPriority.HIGH)
def restore_database_backup_task(backup_id: str, target_database_url: str = None):
    """恢复数据库备份任务"""
    success = asyncio.run(backup_manager.restore_backup(backup_id, target_database_url))
    return {"backup_id": backup_id, "restored": success}

# 备份管理API
def create_backup(backup_type: BackupType = BackupType.FULL, metadata: Optional[Dict[str, Any]] = None) -> str:
    """创建备份"""
    return backup_manager.create_backup(backup_type, metadata)

async def restore_backup(backup_id: str, target_database_url: Optional[str] = None) -> bool:
    """恢复备份"""
    return await backup_manager.restore_backup(backup_id, target_database_url)

def get_backup_list() -> List[BackupInfo]:
    """获取备份列表"""
    return backup_manager.get_backup_list()

def get_backup_info(backup_id: str) -> Optional[BackupInfo]:
    """获取备份信息"""
    return backup_manager.get_backup_info(backup_id)

def delete_backup(backup_id: str) -> bool:
    """删除备份"""
    return backup_manager.delete_backup(backup_id)

def get_backup_stats() -> Dict[str, Any]:
    """获取备份统计"""
    return backup_manager.get_backup_stats()

async def cleanup_old_backups() -> int:
    """清理旧备份"""
    return await backup_manager.cleanup_old_backups()

if __name__ == "__main__":
    # 测试备份管理器
    print("备份管理器统计:")
    print(json.dumps(get_backup_stats(), indent=2, ensure_ascii=False))

    # 创建测试备份
    backup_id = create_backup(BackupType.FULL, {"test": True})
    print(f"创建备份: {backup_id}")

    # 获取备份信息
    backup_info = get_backup_info(backup_id)
    print(f"备份信息: {backup_info}")

__all__ = ["'logger'", "'BackupType'", "'BackupStatus'", "'BackupInfo'", "'BackupConfig'", "'DatabaseBackupManager'", "'backup_config'", "'backup_manager'", "'create_database_backup_task'", "'restore_database_backup_task'", "'create_backup'", "'get_backup_list'", "'get_backup_info'", "'delete_backup'", "'get_backup_stats'"]

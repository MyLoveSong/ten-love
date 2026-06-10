"""pipeline_registry模块\n\n模块描述\n"""

from typing import Callable, Dict, Optional, Any
import pandas as pd

# 可选依赖导入，避免强耦合
try:
    from app.data_processing.data_cleaner import (
        DataPreprocessor, PreprocessingConfig,
        MissingValueStrategy, OutlierDetectionMethod,
        ScalingMethod, EncodingMethod
    )
    _PREPROCESS_AVAILABLE = True
except Exception:
    _PREPROCESS_AVAILABLE = False

try:
    from app.data_processing.self_supervised_learning import (
        SelfSupervisedConfig, SelfSupervisedMethod, train_self_supervised
    )
    _SSL_AVAILABLE = True
except Exception:
    _SSL_AVAILABLE = False

_registry: Dict[str, Callable[..., Any]] = {}

def register_task(name: str, func: Callable[..., Any]) -> None:
    _registry[name] = func

def get_task(name: str) -> Optional[Callable[..., Any]]:
    return _registry.get(name)

# ---------- 默认任务实现（可按需覆盖） ----------

def _default_loader_csv(path: str = "data.csv", **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **{k: v for k, v in kwargs.items() if k != "path"})

def _default_preprocess(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    if not _PREPROCESS_AVAILABLE:
        return df
    from app.data_processing.data_cleaner import make_default_preprocessing_config
    config = make_default_preprocessing_config()
    processor = DataPreprocessor()
    processed_df, _ = processor.preprocess_data(df, config)
    return processed_df

def _default_save_parquet(df: pd.DataFrame, path: str = "output.parquet", **kwargs) -> str:
    df.to_parquet(path, index=False)
    return path

def _default_contrastive_simclr(df: pd.DataFrame, epochs: int = 10, batch_size: int = 64, hidden_dim: int = 128, **kwargs) -> pd.DataFrame:
    if not _SSL_AVAILABLE:
        return df
    import numpy as np
    data_array = df.select_dtypes(include=['number']).to_numpy(dtype=float)
    cfg = SelfSupervisedConfig(
        method=SelfSupervisedMethod("simclr"),
        learning_rate=kwargs.get("learning_rate", 1e-4),
        batch_size=batch_size,
        epochs=epochs,
        hidden_dim=hidden_dim
    )
    result = train_self_supervised(data_array, cfg)
    # 若未来返回嵌入，可将其拼回 df；当前保持 df 不变
    return df

# 预注册默认任务
register_task("loader.csv", _default_loader_csv)
register_task("preprocess.default", _default_preprocess)
register_task("save.parquet", _default_save_parquet)
register_task("contrastive.simclr", _default_contrastive_simclr)

__all__ = ["'register_task'", "'get_task'"]

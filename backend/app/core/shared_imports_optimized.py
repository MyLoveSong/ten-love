"""
优化共享导入模块
消除重复导入，遵循ESM标准
"""

# 标准库导入
import os
import sys
import json
import logging
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import warnings
import hashlib
import re
import ast

# 第三方库导入
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import sessionmaker, relationship
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 项目内部导入
from app.core.mcp_registry import mcp_registry, register_module, register_service, inject
from backend.app.core.exceptions import CustomException
from app.core.configuration import get_configuration
from app.core.structured_logging import get_logger

# 导出所有导入
__all__ = [
    # 标准库
    'os', 'sys', 'json', 'logging', 'asyncio', 'threading', 'time',
    'Path', 'Dict', 'List', 'Any', 'Optional', 'Union', 'Tuple', 'Callable',
    'dataclass', 'field', 'datetime', 'timedelta', 'ABC', 'abstractmethod', 'warnings',
    'hashlib', 're', 'ast',

    # 第三方库
    'np', 'pd', 'FastAPI', 'HTTPException', 'Depends', 'JSONResponse', 'BaseModel',
    'create_engine', 'Column', 'Integer', 'String', 'Float', 'DateTime', 'Boolean', 'Text', 'JSON',
    'sessionmaker', 'relationship', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
    'roc_auc_score', 'mean_squared_error', 'mean_absolute_error', 'r2_score',
    'train_test_split', 'cross_val_score', 'KFold', 'StandardScaler', 'MinMaxScaler',

    # 项目内部
    'mcp_registry', 'register_module', 'register_service', 'inject',
    'CustomException', 'get_configuration', 'get_logger'
]

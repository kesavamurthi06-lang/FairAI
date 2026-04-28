"""
FairAI Guardian - Production Utilities Module
============================================
Centralized utility functions for AI fairness project.

Author: FairAI Guardian Team
Version: 1.0.0
"""
import os
import json
import yaml
import time
import random
import logging
import pickle
from pathlib import Path
from functools import wraps
from typing import Any, Dict, Optional, Callable, List
import numpy as np
import pandas as pd
import joblib
from contextlib import contextmanager


# Module-level logger
_module_logger = logging.getLogger(__name__)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup standardized logger with console and optional file handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
        
    Raises:
        ValueError: Invalid logger name
    """
    if not name or not isinstance(name, str):
        raise ValueError("Logger name must be a non-empty string")
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger


def create_project_directories(
    base_path: str = "."
) -> Dict[str, Path]:
    """
    Create standard project directory structure.
    
    Args:
        base_path: Base project directory
        
    Returns:
        Dictionary of created directory paths
    """
    base = Path(base_path).resolve()
    directories = {
        'data': base / 'data',
        'models': base / 'models',
        'reports': base / 'reports',
        'logs': base / 'logs',
        'artifacts': base / 'artifacts'
    }
    
    for dir_name, dir_path in directories.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        _module_logger.debug(f"Created directory: {dir_path}")
    
    return directories


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Torch seeds (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    _module_logger.info(f"Random seeds set to {seed}")


def save_object(
    obj: Any,
    file_path: str
) -> None:
    """
    Save object using joblib (preferred) with pickle fallback.
    
    Args:
        obj: Object to save
        file_path: Destination file path
        
    Raises:
        IOError: Save operation failed
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        joblib.dump(obj, file_path)
        _module_logger.info(f"Object saved with joblib: {file_path}")
    except Exception:
        # Fallback to pickle
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        _module_logger.warning(f"Object saved with pickle fallback: {file_path}")


def load_object(
    file_path: str
) -> Any:
    """
    Load object using joblib (preferred) with pickle fallback.
    
    Args:
        file_path: Source file path
        
    Returns:
        Loaded object
        
    Raises:
        FileNotFoundError: File does not exist
        IOError: Load operation failed
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Object file not found: {file_path}")
    
    try:
        obj = joblib.load(file_path)
        _module_logger.info(f"Object loaded with joblib: {file_path}")
    except Exception:
        # Fallback to pickle
        try:
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
            _module_logger.info(f"Object loaded with pickle: {file_path}")
        except Exception as e:
            raise IOError(f"Failed to load object from {file_path}: {str(e)}")
    
    return obj


def save_json(
    data: Dict[str, Any],
    file_path: str
) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Destination file path
        
    Raises:
        TypeError: Data is not a dictionary
        IOError: Save operation failed
    """
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary")
    
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    _module_logger.info(f"JSON saved: {file_path}")


def load_json(
    file_path: str
) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        file_path: Source file path
        
    Returns:
        Loaded dictionary
        
    Raises:
        FileNotFoundError: File does not exist
        json.JSONDecodeError: Invalid JSON
        IOError: Load operation failed
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _module_logger.info(f"JSON loaded: {file_path}")
        return data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {file_path}: {str(e)}", e.doc, e.pos)


def load_config(
    file_path: str
) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        file_path: Config file path (.yaml, .yml, or .json)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: Unsupported file format
        FileNotFoundError: File does not exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    ext = path.suffix.lower()
    try:
        if ext in ['.yaml', '.yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        elif ext == '.json':
            return load_json(file_path)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
    except Exception as e:
        raise IOError(f"Failed to load config {file_path}: {str(e)}")


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate pandas DataFrame quality.
    
    Args:
        df: DataFrame to validate
        required_columns: Optional list of required column names
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        TypeError: Input is not DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    issues = []
    
    if df.empty:
        issues.append("DataFrame is empty")
    
    if df.columns.duplicated().any():
        issues.append("Duplicate column names found")
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")
    
    if issues:
        _module_logger.warning(f"DataFrame validation failed: {', '.join(issues)}")
        return False
    
    _module_logger.debug(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to log execution time of functions.
    
    Usage:
        @timing_decorator
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger_name = kwargs.pop('__logger_name__', None) or func.__module__
        logger = logging.getLogger(logger_name)
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} executed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {str(e)}")
            raise
    return wrapper


def export_metrics_report(
    metrics: Dict[str, Any],
    file_path: str
) -> None:
    """
    Export metrics dictionary as JSON and CSV (if tabular).
    
    Args:
        metrics: Metrics dictionary
        file_path: Base file path (JSON extension added)
    """
    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary")
    
    path = Path(file_path)
    json_path = path.with_suffix('.json')
    
    # Save JSON
    save_json(metrics, str(json_path))
    
    # Save CSV if metrics contain tabular data
    if isinstance(list(metrics.values())[0], (dict, list)):
        try:
            df = pd.DataFrame([metrics])
            csv_path = path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            _module_logger.info(f"Metrics CSV saved: {csv_path}")
        except Exception:
            pass
    
    _module_logger.info(f"Metrics report exported: {json_path}")


def get_file_extension(
    file_path: str
) -> str:
    """
    Get file extension (lowercase, without dot).
    
    Args:
        file_path: File path
        
    Returns:
        File extension (e.g., 'json', 'pkl')
    """
    return Path(file_path).suffix.lstrip('.').lower()


# Bonus helper functions
def ensure_directory_exists(path: str) -> Path:
    """Ensure directory exists, create if needed."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def timestamped_filename(prefix: str, extension: str) -> str:
    """Generate timestamped filename."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format float as percentage string."""
    return f"{value*100:.{decimals}f}%"


if __name__ == "__main__":
    """Demonstration block."""
    # Setup
    logger = setup_logger("utils_demo", "logs/utils_demo.log")
    dirs = create_project_directories()
    set_random_seed(42)
    
    # Test data
    test_data = {"accuracy": 0.92, "f1": 0.89, "fairness_score": 0.95}
    test_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
    
    # Save/Load
    save_object(test_df, "models/test_model.pkl")
    loaded_df = load_object("models/test_model.pkl")
    
    # Config
    config = {"seed": 42, "max_iter": 1000}
    save_json(config, "artifacts/config.json")
    
    # Validation
    assert validate_dataframe(loaded_df)
    
    # Metrics
    export_metrics_report(test_data, "reports/demo_metrics")
    
    # Timing demo
    @timing_decorator
    def slow_function():
        time.sleep(0.1)
    
    slow_function()
    
    logger.info("✅ All utilities tested successfully!")
    logger.info(f"Demo metrics: {test_data}")
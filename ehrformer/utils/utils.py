"""Utility functions for EHRFormer project."""
import json
import os
import sys
import re
import pickle
import math
from pathlib import *
import multiprocessing
import random
import numpy as np
import torch
import torch.nn as nn
from collections import *
from itertools import *
from functools import *
from sklearn.metrics import *
import pandas as pd
import hashlib
from PIL import Image
from timeit import default_timer
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pyarrow.parquet as pq
from torch.nn import functional as F


def parse_value(value_str):
    """Parse command line argument value to appropriate Python type"""
    # Handle lists (comma-separated values)
    if ',' in value_str:
        items = [parse_value(item.strip()) for item in value_str.split(',')]
        return items
    
    # Handle booleans
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'
    
    # Handle null/none
    if value_str.lower() in ['null', 'none']:
        return None
    
    # Try to parse as int
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Try to parse as float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str



def parse_unknown_args(unknown_args):
    """Parse unknown command line arguments into a dictionary of overrides.
    
    Args:
        unknown_args: List of unknown command line arguments
        
    Returns:
        Dictionary of parsed key-value pairs
    """
    overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            key = arg[2:]  # Remove '--' prefix
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                # Next argument is the value
                value_str = unknown_args[i + 1]
                overrides[key] = parse_value(value_str)
                i += 2
            else:
                # Flag argument (no value), treat as True
                overrides[key] = True
                i += 1
        else:
            i += 1
    return overrides

def override_config(config, overrides):
    """Override config values with command line arguments
    
    Args:
        config: Base configuration dictionary
        overrides: Dictionary of key-value pairs to override
    
    Returns:
        Updated configuration dictionary
    """
    import copy
    config = copy.deepcopy(config)
    
    for key, value in overrides.items():
        # Support nested keys with dot notation (e.g., "transformer.n_positions")
        keys = key.split('.')
        current_dict = config
        
        # Navigate to the correct nested dictionary
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        
        # Set the final value
        current_dict[keys[-1]] = value
        print(f"Override: {key} = {value} (type: {type(value).__name__})")
    
    return config

def setup_loggers_and_callbacks(config, output_dir, debug=False):
    """Setup loggers and callbacks for PyTorch Lightning training.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
        debug: Whether in debug mode
        is_pretrain: Whether this is pretraining (affects callback monitors)
    
    Returns:
        Tuple of (loggers, callbacks, version_dir)
    """
    from pytorch_lightning.loggers import CSVLogger, WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
    from utils.lr_monitor2 import LearningRateMonitor
    
    log_dir = output_dir
    
    # Determine version for CSV logger
    if not config['train'] and config['test']:
        # version = get_max_version(str(log_dir))
        if 'version_' in config['ckpt_path']:
            version = config['ckpt_path'].split('version_')[1].split('/')[0]
            version = "version_" + str(version)
            print(f"Testing model from version: {version}")
            log_name = 'test'
        else:
            raise Exception(f"Testing model path is not valid !!!")
    else:
        version = None # automatically determine version, add1
        print(f"Training model from version, automatically determine version")
        log_name = 'train'
    # Setup loggers
    logger_csv = CSVLogger(str(log_dir), version=version, name=log_name)
    version_dir = Path(logger_csv.log_dir)
    loggers = [logger_csv]
    
    if not debug:
        wandb_kwargs = {
            'project': config['project'], 
            'name': config['name']}
        logger_wandb = WandbLogger(**wandb_kwargs)
        loggers.append(logger_wandb)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=(version_dir / 'checkpoint'), 
            filename='{epoch}-{val_loss:.3f}', 
            monitor='val_loss', 
            mode='min',
            save_last=True
        ),
        ModelCheckpoint(
            dirpath=(version_dir / 'checkpoint'), 
            filename='{epoch}-{val_mauc:.3f}', 
            monitor='val_mauc', 
            mode='max'
        ),
        ModelCheckpoint(
            dirpath=(version_dir / 'checkpoint'), 
            filename='{epoch}-{val_mpcc:.3f}', 
            monitor='val_mpcc', 
            mode='max', 
        ),
        ModelCheckpoint(
            dirpath=(version_dir / 'checkpoint'), 
            filename='{epoch}-{val_mr2:.3f}', 
            monitor='val_mr2', 
            mode='max', 
        ),
        TQDMProgressBar(refresh_rate=1),
    ]
    
    # Add learning rate monitor
    # 根据是否在debug模式选择logger索引
    if debug:
        # debug模式下只有CSV logger (索引0)
        callbacks.append(LearningRateMonitor(logging_interval='epoch', logger_indexes=0))
    else:
        # 非debug模式下有CSV和WandB logger，使用WandB (索引1)
        callbacks.append(LearningRateMonitor(logging_interval='epoch', logger_indexes=1))
    
    return loggers, callbacks, version_dir

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


def group_by_M_dimension(A_list, B_list, C_list):
    """Group tensors by sequence length for efficient processing."""
    device = A_list[0].device
    M_values = torch.tensor([A.size(1) for A in A_list], device=device)
    unique_M = torch.unique(M_values)
    
    groups = {}
    for M in unique_M.cpu().numpy():
        mask = M_values == M
        indices = torch.where(mask)[0]
        groups[M] = (
            torch.cat([A_list[i] for i in indices], dim=0),
            torch.cat([B_list[i] for i in indices], dim=0),
            torch.cat([C_list[i] for i in indices], dim=0)
        )
    
    return groups

def batch_top5_encode(probs: np.ndarray) -> np.ndarray:
    top5_indices = np.argsort(probs, axis=1)[:, -5:]  
    
    
    encoded = np.zeros(len(probs), dtype=np.int32)
    for i, indices in enumerate(top5_indices):
        
        val = 0
        for idx in indices[::-1]:  
            val = val * 41 + idx
        encoded[i] = val
    
    return encoded

def load_label_info(path):
    label_info = json_load(path)
    cls_label_cols = []
    reg_label_cols = []
    for i in range(len(label_info['label_cols'])):
        if label_info['n_classes'][i] == 1:
            reg_label_cols.append(label_info['label_cols'][i])
        else:
            cls_label_cols.append(label_info['label_cols'][i])
    return cls_label_cols, reg_label_cols


def get_max_version(root_dir):
    if not os.path.isdir(root_dir):
        print("Missing logger folder: %s", root_dir)
        return 0

    existing_versions = []
    for d in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
            existing_versions.append(int(d.split("_")[1]))

    if len(existing_versions) == 0:
        return 0

    return max(existing_versions)


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    
    if isinstance(obj, list):
        obj = [to_device(x, device) for x in obj]
    
    if isinstance(obj, dict):
        for k in obj.keys():
            obj[k] = to_device(obj[k], device)
    
    return obj
    

def isnan(x):
    if not isinstance(x, float):
        return False
    return math.isnan(x)

def to_dataset_mapping(ids, n_fold, salt=''):
    result = {}
    for one_id in ids:
        result[one_id] = int(hashlib.sha256((str(one_id)+salt).encode('utf-8')).hexdigest(), 16) % n_fold
    return result

def str_hash(s, salt=''):
    return int(hashlib.sha256((str(s)+salt).encode('utf-8')).hexdigest(), 16)

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def json_dump(obj, path):
    ensure_file(path)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, sort_keys=True, cls=SetEncoder)


def json_load(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def pkl_dump(obj, path):
    ensure_file(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def pkl_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def np_save(obj, path):
    ensure_file(path)
    with open(path, 'wb') as f:
        np.save(f, obj)


def np_load(path):
    with open(path, 'rb') as f:
        return np.load(f)


def chunk(list, n):
    result = []
    for i in range(n):
        result.append(list[math.floor(i / n * len(list)):math.floor((i + 1) / n * len(list))])
    return result


def df_split(list, ratios):
    results = []
    sum_value = sum(ratios)
    ratios = [x / sum_value for x in ratios]
    current = 0
    for ratio in ratios:
        results.append(list[int(len(list) * current):int(len(list) * (current + ratio))])
        current += ratio
    return results


def list_to_str(list):
    return [str(x) for x in list]


def chunk_sample(list, n):
    result = []
    for i in range(1, n):
        result.append(list[math.floor(i / n * len(list))])
    return result


def chunk_to_batches(list, batch_size):
    result = []
    for i in range(0, len(list), batch_size):
        result.append(list[i:i + batch_size])
    return result


def ensure_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_file(filepath):
    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)


def run_multi_process(item_list, n_proc, func, with_proc_num=False):
    tasks = chunk(item_list, n_proc)
    if with_proc_num:
        for i in range(len(tasks)):
            tasks[i] = (i, tasks[i])
    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.map(func, tasks)
    return results


def bootstrap(func, y_true, y_pred, n=100, random_state=42, ci=(0.025, 0.975), index=None, with_ci=True):
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    val = func(y_true, y_pred)
    if index is not None:
        val = val[index]
    if not with_ci:
        return val
    bootstrapped_scores = []
    rng = np.random.RandomState(random_state)
    for i in range(n):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = func(y_true[indices], y_pred[indices])
        if index is not None:
            score = score[index]
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = sorted_scores[int(ci[0] * len(sorted_scores))]
    ci_upper = sorted_scores[int(ci[1] * len(sorted_scores))]
    return val, ci_lower, ci_upper


def print_df(df, row=2):
    cols = df.columns.tolist()
    pd.set_option('display.max_columns', len(cols))
    pd.set_option('display.max_rows', row)
    print(cols)
    print(len(df))
    # display(df)
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    

def df2map(df,col_key,col_val):
    return df.drop_duplicates(col_key).set_index(col_key)[col_val]


def isnan(x):
    return isinstance(x, float) and math.isnan(x)


def vc(series, to_dict=True, dropna=True):
    result = series.value_counts(dropna=dropna)
    if to_dict:
        return print(result.to_dict())
    print(result)
    

def bp():
    raise Exception()
    

class Benchmark(object):
    def __init__(self, msg, print=True):
        self.msg = msg
        self.print = print

    def print_elapsed(self, add_msg):
        t = default_timer() - self.start
        if self.print:
            print(f"{self.msg}, {add_msg}: {t:.2f} seconds")

    def __enter__(self):
        self.start = default_timer()
        if self.print:
            print(f"{self.msg}: begin")
        return self

    def __exit__(self, *args):
        t = default_timer() - self.start
        if self.print:
            print(f"{self.msg}: {t:.2f} seconds")
        self.time = t


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_parquet(df, path):
    tmp_df = df.copy()
    for col in tqdm(tmp_df.columns):
        if isinstance(tmp_df.iloc[0][col], np.ndarray) and len(tmp_df.iloc[0][col].shape) == 2:
            tmp_df[col] = [x.tolist() for x in tmp_df[col]]
    tmp_df.to_parquet(path)


def load_parquet(path, reset_index=True):
    parquet_file = pq.ParquetFile(path)
    tmp_df = []
    for sub_df in parquet_file.iter_batches(batch_size=100000):
        tmp_df.append(sub_df.to_pandas())
    tmp_df = pd.concat(tmp_df, axis=0)
    for col in tqdm(tmp_df.columns):
        if isinstance(tmp_df.iloc[0][col], np.ndarray) and isinstance(tmp_df.iloc[0][col][0], np.ndarray):
            tmp_df[col] = [np.array(list(x)) for x in tmp_df[col]]
    if reset_index:
        tmp_df = tmp_df.reset_index(drop=True)
    return tmp_df


def get_feat_cols(df, prefix):
    if isinstance(prefix, list):
        result = []
        for p in prefix:
            result += list(filter(lambda x: x.startswith(p), df.columns))
        return result
    return list(filter(lambda x: x.startswith(prefix), df.columns))

def load_checkpoint_with_key_mapping(checkpoint_path, target_model, key_mappings, model_name="Model"):
    """
    加载检查点并应用键值映射转换
    
    Args:
        checkpoint_path (str): 检查点文件路径
        target_model: 目标模型对象
        key_mappings (list): 键值映射列表，格式为 [(old_prefix, new_prefix), ...]
        model_name (str): 模型名称，用于日志输出
    
    Returns:
        bool: 是否成功加载
    """
    from pathlib import Path
    
    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Loading {model_name} from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = {}
        
        # 获取原始状态字典
        source_dict = checkpoint.get('state_dict', checkpoint)
        
        # 应用键值映射
        for old_prefix, new_prefix in key_mappings:
            for key, value in source_dict.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix)
                    state_dict[new_key] = value
        
        if state_dict:
            missing_keys, unexpected_keys = target_model.load_state_dict(
                state_dict, strict=False
            )
            print(f"  ✓ Loaded {model_name}")
            print(f"    - Params loaded: {len(state_dict)}")
            print(f"    - Missing keys: {len(missing_keys)}")
            print(f"    - Unexpected keys: {len(unexpected_keys)}")
            if missing_keys:
                print(f"    - Missing: {missing_keys[:3]}" + ("..." if len(missing_keys) > 3 else ""))
            if unexpected_keys:
                print(f"    - Unexpected: {unexpected_keys[:3]}" + ("..." if len(unexpected_keys) > 3 else ""))
            return True
        else:
            print(f"  ✗ No {model_name} params found")
            return False
            
    except Exception as e:
        print(f"  ✗ Loading failed: {e}")
        return False
from pathlib import Path
import random
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F
import pyarrow.parquet as pq
import pyarrow as pa
import diskcache
import gc
import json
from utils.utils import *

def _row_has_positive_label(row_values):
    """Check if any label array in a row contains positive (==1) labels."""
    for arr in row_values:
        if arr is None:
            continue
        if isinstance(arr, (list, tuple)):
            arr = np.asarray(arr)
        if isinstance(arr, np.ndarray):
            if np.any(arr == 1):
                return True
        else:
            if arr == 1:
                return True
    return False


def negative_sample_df(df, config):
    """Downsample negative samples in training data.

    Keeps all positive samples, and randomly keeps negatives to satisfy:
        n_neg_kept = n_pos * neg_sample_ratio

    Controlled by config:
        - neg_sample_ratio (float, default: None -> no sampling)
        - neg_sample_seed (int, default: 42)
        - neg_sample_shuffle (bool, default: True)
    """
    if config.get('mode') == 'pretrain' or config.get('stage') == 'pretrain':
        return df

    ratio = config.get('neg_sample_ratio', None)
    if ratio is None:
        return df

    label_cols = [c for c in config.get('cls_label_cols', []) if c in df.columns]
    positive_mask = np.zeros(len(df), dtype=bool)

    if label_cols:
        values = df[label_cols].values
        for i in range(len(df)):
            if _row_has_positive_label(values[i]):
                positive_mask[i] = True
    else:
        # Fallback: build a positive pid set by scanning caches once (faster than per-row get)
        if not config.get('use_cache', False):
            print("[negative_sample] Skip: no label columns and cache disabled")
            return df

        if 'pid' not in df.columns:
            print("[negative_sample] Skip: no pid column to fetch cache labels")
            return df

        cache_paths = config.get('df_paths', [])
        if not isinstance(cache_paths, list):
            cache_paths = [cache_paths]
        caches = [diskcache.Cache(path) for path in cache_paths]

        cls_cols = config.get('cls_label_cols', [])
        positive_pids = set()
        total_keys = 0
        for cache in caches:
            try:
                keys = list(cache.iterkeys())
            except Exception:
                keys = []
            total_keys += len(keys)
            log_stride = max(1, len(keys) // 10) if keys else 0
            for idx, pid in enumerate(keys):
                item = cache.get(pid, default=None)
                if item is None:
                    continue
                for col in cls_cols:
                    if col not in item:
                        continue
                    arr = item[col]
                    if isinstance(arr, (list, tuple)):
                        arr = np.asarray(arr)
                    if isinstance(arr, np.ndarray):
                        if np.any(arr == 1):
                            positive_pids.add(pid)
                            break
                    else:
                        if arr == 1:
                            positive_pids.add(pid)
                            break
                if log_stride and (idx + 1) % log_stride == 0:
                    print(f"[negative_sample] cache scan {idx+1}/{len(keys)} (cache size {len(cache)}) pos_pids={len(positive_pids)}")
        print(f"[negative_sample] cache scan done total_keys={total_keys} pos_pids={len(positive_pids)}")

        pid_series = df['pid']
        positive_mask = pid_series.isin(positive_pids).to_numpy()

    n_pos = int(positive_mask.sum())
    n_neg = int(len(df) - n_pos)
    if n_pos == 0:
        print("[negative_sample] No positive samples found. Skip sampling.")
        return df

    if ratio <= 0:
        sampled_df = df[positive_mask].copy()
    else:
        keep_n_neg = int(n_pos * float(ratio))
        if keep_n_neg >= n_neg:
            sampled_df = df.copy()
        else:
            seed = int(config.get('neg_sample_seed', 42))
            neg_df = df[~positive_mask]
            pos_df = df[positive_mask]
            neg_sampled = neg_df.sample(n=keep_n_neg, random_state=seed, replace=False)
            sampled_df = pd.concat([pos_df, neg_sampled], axis=0, ignore_index=True)

    if config.get('neg_sample_shuffle', True):
        seed = int(config.get('neg_sample_seed', 42))
        sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(
        f"[negative_sample] pos={n_pos}, neg={n_neg}, "
        f"keep_neg={len(sampled_df) - n_pos} (ratio={ratio})"
    )
    return sampled_df



def optimize_dtypes(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            col_min, col_max = df[col].min(), df[col].max()
            
            if col_min >= 0:
                if col_max < 2**8:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 2**16:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 2**32:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > -2**7 and col_max < 2**7:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -2**15 and col_max < 2**15:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2**31 and col_max < 2**31:
                    df[col] = df[col].astype(np.int32)
        
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
            
        elif pd.api.types.is_string_dtype(df[col]) and df[col].nunique() / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
            
    return df

def sample_subset_float(mask, prob, seed=42):
    """Sample subset of valid indices for masking, following original logic"""
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    n_total = len(valid_indices)
    n_output = max(1, int(n_total * prob))
    
    # Set random seed for reproducible sampling
    # None is for training, fixed seed (with PID) is for testing
    if seed is not None:
        np.random.seed(seed)

    # Randomly select indices for output (masked prediction)
    # Remaining indices are for input
    output_indices = np.random.choice(valid_indices, size=n_output, replace=False)
    input_indices = np.setdiff1d(valid_indices, output_indices)
    
    return input_indices, output_indices # output_indices is the indices of the features that should be predicted

def sample_subset_cat(mask, prob, seed=42):
    """Sample subset of valid indices for masking, following original logic"""
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    n_total = len(valid_indices)
    n_output = max(1, int(n_total * prob))

    if seed is not None:
        np.random.seed(seed)

    # each element has a probability of prob to be selected, may 0 or all
    # random_values = np.random.random(n_total)
    # output_indices = valid_indices[random_values < prob]
    # input_indices = valid_indices[~(random_values < prob)]
    output_indices = np.random.choice(valid_indices, size=n_output, replace=False)
    input_indices = np.setdiff1d(valid_indices, output_indices)
    
    return input_indices, output_indices


@dataclass
class ChunkedEHRDataset2D(Dataset):
    data_chunks: list  
    mode: str
    config: dict

    def __post_init__(self):
        self.df_paths = self.config['df_paths']
        self.use_cache = self.config['use_cache']
        if self.use_cache:
            
            self.caches = [
                diskcache.Cache(path, eviction_policy='none')
                for path in self.df_paths
            ]
            
        self.chunk_lengths = [len(chunk) for chunk in self.data_chunks]
        self.cumulative_lengths = np.cumsum(self.chunk_lengths)
        self.total_length = self.cumulative_lengths[-1] if self.chunk_lengths else 0
        self._setup_config()

    def _setup_config(self):
        """Config initialization."""
        self.config['task_info'] = json_load(self.config['task_info_path'])
        self.config['reg_label_info'] = self.config['task_info']['float_cols']
        self.config['reg_label_info'] = {k: (v['mean'], v['std']) for k, v in self.config['reg_label_info'].items()}

        self.dataframe_cols = set().union(*[set(chunk.columns) for chunk in self.data_chunks])
        self.config['cls_label_names'] = []
        self.config['reg_label_names'] = []
        
        # Use first row of first chunk to initialize label names
        first_row = self.data_chunks[0].iloc[0]

        # Process classification labels: combine each cls_label_cols with each diag_cols
        for label_cols, cols in zip(
            self.config['cls_label_cols'],
            self.config['cls_label_name_cols'],
        ):
            # Get diagnosis column names list from first row of metadata.parquet
            if cols in first_row and hasattr(first_row[cols], '__iter__'):
                diag_names = first_row[cols]  # Array containing all diagnosis names
                for col in diag_names:
                    if col in self.config['task_info']['category_cols']: 
                        self.config['cls_label_names'].append(f'{label_cols}_{col}')
                print(f"Added {len(self.config['cls_label_names'])} classification tasks for {label_cols}")
        
        # Process regression labels: combine each reg_label_cols with each reg_cols
        for label_cols, cols in zip(
            self.config['reg_label_cols'],
            self.config['reg_label_name_cols']
        ):
            # Get regression column names list from first row of metadata.parquet
            if cols in first_row and hasattr(first_row[cols], '__iter__'):
                reg_names = first_row[cols]  # Array containing all regression names
                for col in reg_names:
                    if col in list(self.config['reg_label_info']):
                        self.config['reg_label_names'].append(f'{label_cols}_{col}')
                print(f"Added {len(self.config['reg_label_names'])} regression tasks for {label_cols}")
        
        print("="*50)        
        print(f"Total configured tasks: {len(self.config['cls_label_names'])} classification + {len(self.config['reg_label_names'])} regression")

    def get_from_cache(self, pid):
        if not self.use_cache:
            return None
            
        for cache in self.caches:
            try:
                data = cache.get(pid, default=None)
                if data is not None:
                    return data
            except:
                continue
        return None

    def read_col(self, chunk_idx: int, within_chunk_idx: int, data: dict, col: str):
        if col in self.dataframe_cols:
            return self.data_chunks[chunk_idx].iloc[within_chunk_idx][col]
        return data[col]

    def read_sample(self, chunk_idx: int, within_chunk_idx: int, data: dict):
        cat_feats = self.read_col(chunk_idx, within_chunk_idx, data, 'tokenized_category_feats')
        float_feats = self.read_col(chunk_idx, within_chunk_idx, data, 'tokenized_float_feats')
        valid_mask = self.read_col(chunk_idx, within_chunk_idx, data, 'valid_mask')
        time_index = self.read_col(chunk_idx, within_chunk_idx, data, 'time_index')

        tensors = {
            'cat_feats': torch.from_numpy(cat_feats.astype(np.int64)),
            'float_feats': torch.from_numpy(float_feats.astype(np.int64)),
            'valid_mask': torch.from_numpy(valid_mask.astype(bool)),
            'time_index': torch.from_numpy(time_index.astype(np.int64))
        }
        
        return tensors

    def read_label(self, chunk_idx: int, within_chunk_idx: int, data: dict):
        labels = {
            'cls': {'values': [], 'masks': []},
            'reg': {'values': [], 'masks': []},
            'time_index': self.read_col(chunk_idx, within_chunk_idx, data, 'time_index').astype(np.int64)
        }
        for value_col in self.config['cls_label_cols']:
            values = self.read_col(chunk_idx, within_chunk_idx, data, value_col)
            if len(values.shape) == 1:
                values = values.reshape(1, -1)
            for value in values:
                value = np.nan_to_num(value, nan=-1)
                mask = (value != -1) & ~np.isnan(value)
                labels['cls']['values'].append(value)
                labels['cls']['masks'].append(mask)
        
        for value_col in self.config['reg_label_cols']:
            values = self.read_col(chunk_idx, within_chunk_idx, data, value_col)
            if len(values.shape) == 1:
                values = values.reshape(1, -1)
            for value, (key, (mean, std)) in zip(values, self.config['reg_label_info'].items()):
                value = (value - mean) / std
                value = np.nan_to_num(value, nan=-1)
                mask = (value != -1) & ~np.isnan(value)
                labels['reg']['values'].append(value)
                labels['reg']['masks'].append(mask)

        if labels['cls']['values']:
            values = torch.tensor(np.stack(labels['cls']['values'], axis=0), dtype=torch.long)
            values[values == -1] = 0
            labels['cls']['values'] = values
            labels['cls']['masks'] = torch.tensor(np.stack(labels['cls']['masks'], axis=0), dtype=torch.bool)

        if labels['reg']['values']:
            labels['reg']['values'] = torch.tensor(np.stack(labels['reg']['values'], axis=0), dtype=torch.float32)
            labels['reg']['masks'] = torch.tensor(np.stack(labels['reg']['masks'], axis=0), dtype=torch.bool)

        labels['time_index'] = torch.tensor(labels['time_index'], dtype=torch.long)
        return labels

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        chunk_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if chunk_idx > 0:
            within_chunk_idx = idx - self.cumulative_lengths[chunk_idx - 1]
        else:
            within_chunk_idx = idx
            
        if self.use_cache:
            pid = self.data_chunks[chunk_idx].iloc[within_chunk_idx]['pid']
            pid = int(pid) if hasattr(pid, 'item') else pid  # numpy.int64 → Python int
            data = self.get_from_cache(pid)
        else:
            data = None
        
        return {
            'pid': self.read_col(chunk_idx, within_chunk_idx, data, 'pid'),
            'data': self.read_sample(chunk_idx, within_chunk_idx, data),
            'label': self.read_label(chunk_idx, within_chunk_idx, data)
        }

class EHRDataModule2D(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.df_paths = config.get('df_paths', [])  
        self.use_cache = config.get('use_cache', False)
        self.dataset_col = config.get('dataset_col', None)
        self.batch_size = config.get('batch_size', None)
        self.train_folds = config.get('train_folds', None)
        self.valid_folds = config.get('valid_folds', None)
        self.test_folds = config.get('test_folds', None)
        self.chunk_size = config.get('chunk_size', 1000000)

    def setup(self, stage=None):
        if not isinstance(self.df_paths, list):
            self.df_paths = [self.df_paths]
        if isinstance(self.df_paths[0], pd.DataFrame):
            self._setup_from_dataframe()
        else:
            self._setup_from_parquet()

    def _setup_from_dataframe(self):
        df = pd.concat(sorted(self.df_paths), axis=0, ignore_index=True)
        df = optimize_dtypes(df)
        print(f"[datamodule] before negative_sample_df rows={len(df)}")
        df = negative_sample_df(df, self.config)
        print(f"[datamodule] after negative_sample_df rows={len(df)}")

        df_train = df[df[self.dataset_col].isin(self.train_folds)].reset_index(drop=True)
        df_valid = df[df[self.dataset_col].isin(self.valid_folds)].reset_index(drop=True)
        df_test = df[df[self.dataset_col].isin(self.test_folds)].reset_index(drop=True)
        
        self.config['df_paths'] = self.df_paths  
        
        self.ds_train = ChunkedEHRDataset2D([df_train], 'train', self.config)
        self.ds_valid = ChunkedEHRDataset2D([df_valid], 'test', self.config)
        self.ds_test = ChunkedEHRDataset2D([df_test], 'test', self.config)

    def _setup_from_parquet(self):
        dfs = []
        for path in sorted(self.df_paths):
            metadata_path = Path(path) / 'metadata.parquet'
            print(f"[datamodule] loading metadata: {metadata_path}")
            df = load_parquet(metadata_path)
            print(f"[datamodule] loaded metadata rows={len(df)} cols={len(df.columns)}")
            dfs.append(df)
    
        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        merged_df = optimize_dtypes(merged_df)
        print(f"[datamodule] before negative_sample_df rows={len(merged_df)}")
        merged_df = negative_sample_df(merged_df, self.config)
        print(f"[datamodule] after negative_sample_df rows={len(merged_df)}")

        df_train = merged_df[merged_df[self.dataset_col].isin(self.train_folds)].reset_index(drop=True)
        df_valid = merged_df[merged_df[self.dataset_col].isin(self.valid_folds)].reset_index(drop=True)
        df_test = merged_df[merged_df[self.dataset_col].isin(self.test_folds)].reset_index(drop=True)
        
        self.config['df_paths'] = self.df_paths
        
        
        self.ds_train = ChunkedEHRDataset2D([df_train], 'train', self.config)
        self.ds_valid = ChunkedEHRDataset2D([df_valid], 'test', self.config)
        
        if self.config.get('test_df', '') != '':
            df_test = pd.read_csv(self.config['test_df'])
            df_test = optimize_dtypes(df_test)
            self.ds_test = ChunkedEHRDataset2D([df_test], 'test', self.config)
        else:
            self.ds_test = ChunkedEHRDataset2D([df_test], 'test', self.config)

    def train_dataloader(self):
        def worker_init_fn(worker_id):
            np.random.seed(1 + worker_id)
            random.seed(1 + worker_id)
            
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=16,
            multiprocessing_context='fork',
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=8,
            multiprocessing_context='fork'
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=8,
            multiprocessing_context='fork'
        )

    def teardown(self, stage=None):
        gc.collect()

@dataclass
class ChunkedEHRDataset1D(Dataset):
    data_chunks: list
    mode: str
    config: dict
    def __post_init__(self):
        self.mask_ratio = self.config.get('mask_ratio', 0.5)
        
        self.feat_float_cols = self.config['feat_float_cols']
        self.feat_cat_cols = self.config['feat_cat_cols']
        self.input_float_cols = self.config['input_float_cols'] # previous [f'tokenized.{x}' for x in self.feat_float_cols]
        self.input_cat_cols = self.config['input_cat_cols'] # previous [f'tokenized.{x}' for x in self.feat_cat_cols]
        self.feat_mean_std = np.array([
            (self.config['feat_info']['float_cols'][x]['mean'], 
             self.config['feat_info']['float_cols'][x]['std']) 
            for x in self.feat_float_cols
        ], dtype=np.float32)
        self.config['cls_label_names'] = self.input_cat_cols
        self.config['reg_label_names'] = self.feat_float_cols
        self.chunk_lengths = [len(chunk) for chunk in self.data_chunks]
        self.cumulative_lengths = np.cumsum(self.chunk_lengths)
        self.total_length = self.cumulative_lengths[-1] if self.chunk_lengths else 0

    def read_sample_label(self, row):
        input_float_feats = row[self.input_float_cols].values.astype(np.int64)
        float_mask = input_float_feats != -1

        cat_feats = np.nan_to_num(row[self.input_cat_cols].values.astype(np.float32), nan=-1).astype(np.int64)
        cat_mask = cat_feats != -1

        assert self.mask_ratio != 0, "mask_ratio should not be 0, during pretraining"
        if len(self.feat_float_cols) > 0: # here when do not use lab float's len is 0
            norm_output_float_values = row[self.feat_float_cols].values.astype(np.float32)
            norm_output_float_values = np.nan_to_num(norm_output_float_values, nan=-1.0)
            norm_output_float_values = (norm_output_float_values - self.feat_mean_std[:, 0]) / (self.feat_mean_std[:, 1] + 1e-10)
        else:
            norm_output_float_values = np.zeros_like(input_float_feats)
        # training do not use specific seed, testing/validation use PID->seed->fixed mask
        # 说明：将验证/测试阶段的掩码固定为基于 pid 的可复现采样，以稳定评估指标
        sample_seed = hash(str(row['pid'])) % (2**32) if self.mode == 'test' else None

        # mask the float features
        input_float_indices, float_output_indices = sample_subset_float(float_mask, self.mask_ratio, seed=sample_seed)
        input_float_mask = np.zeros_like(float_mask, dtype=bool)
        input_float_mask[input_float_indices] = True
        input_float_values = np.zeros_like(input_float_feats) - 1
        input_float_values[input_float_indices] = input_float_feats[input_float_indices]
        output_float_mask = np.zeros_like(float_mask, dtype=bool)
        output_float_mask[float_output_indices] = True
        output_float_values = np.zeros_like(norm_output_float_values)
        output_float_values[float_output_indices] = norm_output_float_values[float_output_indices]
        # mask the cat features
        cat_input_indices, cat_output_indices = sample_subset_cat(cat_mask, self.mask_ratio, seed=sample_seed)
        input_cat_mask = np.zeros_like(cat_mask, dtype=bool)
        input_cat_mask[cat_input_indices] = True
        input_cat_values = np.zeros_like(cat_feats) - 1
        input_cat_values[cat_input_indices] = cat_feats[cat_input_indices]
        output_cat_mask = np.zeros_like(cat_mask, dtype=bool)
        output_cat_mask[cat_output_indices] = True
        output_cat_values = np.zeros_like(cat_feats)
        output_cat_values[cat_output_indices] = cat_feats[cat_output_indices]
        # return the all features
        all_cat_values = output_cat_values
        all_cat_masks = output_cat_mask  
        all_float_values = output_float_values
        all_float_masks = output_float_mask
        
        sample = {
            'cat_feats': torch.tensor(input_cat_values, dtype=torch.long),
            'cat_valid_mask': torch.tensor(input_cat_mask, dtype=torch.bool),
            'float_feats': torch.tensor(input_float_values, dtype=torch.long),
            'float_valid_mask': torch.tensor(input_float_mask, dtype=torch.bool)
        }
        
        label = {
            'cat_feats': torch.tensor(all_cat_values, dtype=torch.long),
            'cat_valid_mask': torch.tensor(all_cat_masks, dtype=torch.bool),
            'float_feats': torch.tensor(all_float_values, dtype=torch.float32),
            'float_valid_mask': torch.tensor(all_float_masks, dtype=torch.bool)
        }
        
        return sample, label

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        chunk_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if chunk_idx > 0:
            within_chunk_idx = idx - self.cumulative_lengths[chunk_idx - 1]
        else:
            within_chunk_idx = idx
        row = self.data_chunks[chunk_idx].iloc[within_chunk_idx]
        sample, label = self.read_sample_label(row)

        return {
            'pid': str(row['pid']),
            'vid': str(row['vid']),
            'data': sample,
            'label': label
        }


class EHRDataModule1D(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.df_path = config.get('df_path', config.get('df_paths', None))
        self.dataset_col = config.get('dataset_col', None)
        self.batch_size = config.get('batch_size', None)
        self.train_folds = config.get('train_folds', None)
        self.valid_folds = config.get('valid_folds', None)
        self.test_folds = config.get('test_folds', None)
        self.feat_float_cols = list(self.config['feat_info']['float_cols'].keys())
        self.feat_cat_cols = self.config['feat_info']['category_cols']
        self.config['feat_float_cols'] = self.feat_float_cols
        self.config['feat_cat_cols'] = self.feat_cat_cols
        
        # 根据 use_info_EHR 决定是否包含诊断特征
        self.use_info_EHR = self.config.get('use_info_EHR', False)
        if self.use_info_EHR:
            self.diag_cols = [x for x in self.config['task_info']['category_cols']]
        else:
            self.diag_cols = []
        self.config['diag_cols'] = self.diag_cols

        # 根据 use_info_LAB 决定是否包含LAB特征
        self.use_info_LAB = self.config.get('use_info_LAB', True)  # 默认启用LAB特征
        if not self.use_info_LAB: # 请注意 float_cols还有sign这种类型的因此不能用lab来筛选
            self.feat_float_cols, self.feat_cat_cols = [], []

        self.input_float_cols = [f'tokenized.{x}' for x in self.feat_float_cols]
        self.input_cat_cols = [f'tokenized.{x}' for x in self.feat_cat_cols]
        self.input_cat_cols.extend(self.diag_cols) # diag 不带tokenized即可

        # we need to update the config
        self.config['input_float_cols'] = self.input_float_cols
        self.config['input_cat_cols'] = self.input_cat_cols
        self.config['feat_float_cols'] = self.feat_float_cols
        self.config['feat_cat_cols'] = self.feat_cat_cols

        self.needed_columns = ['pid', 'vid', self.dataset_col] + \
                                self.input_cat_cols + self.feat_cat_cols + \
                                self.input_float_cols + self.feat_float_cols
        self.chunk_size = config.get('chunk_size', 1000000)

    def setup(self, stage=None):
        if isinstance(self.df_path, pd.DataFrame):
            df = self.df_path
            df = optimize_dtypes(df)
            df = negative_sample_df(df, self.config)
            df_train = df[df[self.dataset_col].isin(self.train_folds)].reset_index(drop=True)
            df_valid = df[df[self.dataset_col].isin(self.valid_folds)].reset_index(drop=True)
            df_test = df[df[self.dataset_col].isin(self.test_folds)].reset_index(drop=True)
            
            self.ds_train = ChunkedEHRDataset1D([df_train], 'train', self.config)
            self.ds_valid = ChunkedEHRDataset1D([df_valid], 'test', self.config)
            self.ds_test = ChunkedEHRDataset1D([df_test], 'test', self.config)
        else:
            parquet_file = pq.ParquetFile(Path(self.df_path))
            df_train_chunks = []
            df_valid_chunks = []
            df_test_chunks = []
            dataset_col_table = pq.read_table(self.df_path, columns=[self.dataset_col])
            dataset_values = dataset_col_table.to_pandas()[self.dataset_col]
            train_mask = dataset_values.isin(self.train_folds)
            valid_mask = dataset_values.isin(self.valid_folds)
            test_mask = dataset_values.isin(self.test_folds)
            train_indices = train_mask[train_mask].index.tolist()
            valid_indices = valid_mask[valid_mask].index.tolist()
            test_indices = test_mask[test_mask].index.tolist()
            del dataset_col_table, dataset_values, train_mask, valid_mask, test_mask
            gc.collect()
            chunk_start_idx = 0
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size, columns=self.needed_columns):
                chunk_df = optimize_dtypes(batch.to_pandas())
                chunk_end_idx = chunk_start_idx + len(chunk_df)
                chunk_train_indices = [i for i in train_indices if chunk_start_idx <= i < chunk_end_idx]
                chunk_valid_indices = [i for i in valid_indices if chunk_start_idx <= i < chunk_end_idx]
                chunk_test_indices = [i for i in test_indices if chunk_start_idx <= i < chunk_end_idx]
                chunk_train_indices = [i - chunk_start_idx for i in chunk_train_indices]
                chunk_valid_indices = [i - chunk_start_idx for i in chunk_valid_indices]
                chunk_test_indices = [i - chunk_start_idx for i in chunk_test_indices]
                if chunk_train_indices:
                    df_train_chunks.append(chunk_df.iloc[chunk_train_indices].reset_index(drop=True))
                if chunk_valid_indices:
                    df_valid_chunks.append(chunk_df.iloc[chunk_valid_indices].reset_index(drop=True))
                if chunk_test_indices:
                    df_test_chunks.append(chunk_df.iloc[chunk_test_indices].reset_index(drop=True))
                
                chunk_start_idx = chunk_end_idx
                del chunk_df
                gc.collect()
            self.ds_train = ChunkedEHRDataset1D(df_train_chunks, 'train', self.config)
            self.ds_valid = ChunkedEHRDataset1D(df_valid_chunks, 'test', self.config)
            
            if self.config.get('test_df', '') != '':
                df_test = pd.read_csv(self.config['test_df'])
                df_test = optimize_dtypes(df_test)
                self.ds_test = ChunkedEHRDataset1D([df_test], 'test', self.config)
            else:
                self.ds_test = ChunkedEHRDataset1D(df_test_chunks, 'test', self.config)

    def train_dataloader(self):
        def worker_init_fn(worker_id):
            np.random.seed(1 + worker_id)
            random.seed(1 + worker_id)
            
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=8,
            multiprocessing_context='fork',
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=8,
            multiprocessing_context='fork'
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=8,
            multiprocessing_context='fork'
        )

    def teardown(self, stage=None):
        gc.collect()


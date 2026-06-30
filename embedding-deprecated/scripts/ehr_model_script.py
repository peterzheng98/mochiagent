"""EHRFormer finetuning module for downstream tasks."""
from collections import defaultdict
from functools import reduce
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, auc
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.nn import functional as F
from torchmetrics import *
from torchmetrics.utilities.data import dim_zero_cat
import torch
import torchmetrics.functional as MF
from tqdm import tqdm
from utils.ddp_utils import *
import numpy as np
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from ehrformer import EHRVAE1D, EHRGPT2
from sklearn.preprocessing import label_binarize
from utils.utils import group_by_M_dimension, batch_top5_encode, FocalLoss
import gc

class EHRVAE1D_pretrain(pl.LightningModule):
    """1D EHR Module for pretraining with VAE architecture"""
    def __init__(self, config):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(config)

        self.config = config
        self.data_type = config.get('data_type', '')
        self.momentum = config.get('momentum', 0.9)
        self.wd = config.get('wd', 1e-6)
        self.lr = config.get('lr', 1e-3)
        self.n_nodes = config.get('n_nodes', 1)
        self.n_gpus = config.get('n_gpus', 1)
        if isinstance(self.n_gpus, list):
            self.n_gpus = len(self.n_gpus)
        self.n_epoch = config.get('n_epoch', None)
        
        # Use only feat_info features for EHRVAE compatibility
        self.cls_label_names = self.config['cls_label_names']
        self.reg_label_names = self.config['reg_label_names']
        
        # Create time-independent 1D model
        self.model = EHRVAE1D(config)
        
        # Initialize Focal Loss for handling class imbalance
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')

        # output directory
        # self.output_dir = config.get('output_dir', None)
        # if self.output_dir is not None:
        #     self.output_dir = Path(self.output_dir) / 'pred'
        #     self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pred_folder = config.get('pred_folder', None)

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )
        scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                                  first_cycle_steps=self.n_epoch,
                                                  max_lr=self.lr, 
                                                  min_lr=1e-8, 
                                                  warmup_steps=int(self.n_epoch * 0.1))
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        scheduler.step()

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def on_train_start(self):
        self.loggers[0].log_hyperparams(self.config)

    def training_step(self, batch, batch_idx):
        
        # get data and label
        data = batch['data']
        label = batch['label']
        cat_feats = data['cat_feats']  # (B, n_cat)
        float_feats = data['float_feats']  # (B, n_float)
        
        # Target values
        cls_label = label['cat_feats']  # (B, n_cat)
        cls_mask = label['cat_valid_mask']  # (B, n_cat)
        reg_label = label['float_feats']  # (B, n_float)
        reg_mask = label['float_valid_mask']  # (B, n_float)

        # Get model output in EHRVAE format
        y_cls, y_reg, mu_z, std_z = self.model(cat_feats, float_feats)
        
        # Regression loss (identical to EHRVAE)
        y_reg_loss = y_reg.reshape(-1)
        reg_label_loss = reg_label.reshape(-1)
        reg_mask_loss = reg_mask.reshape(-1)
        reg_losses = F.mse_loss(y_reg_loss, reg_label_loss, reduction='none')
        reg_loss = (reg_losses * reg_mask_loss).sum() / (reg_mask_loss.sum().clip(1))

        # Classification loss using Focal Loss for handling class imbalance
        y_cls_loss = y_cls.reshape(-1, 2)
        cls_label_loss = cls_label.reshape(-1)
        cls_mask_loss = cls_mask.reshape(-1)
        cls_losses = self.focal_loss(y_cls_loss, cls_label_loss)
        cls_loss = (cls_losses * cls_mask_loss).sum() / (cls_mask_loss.sum().clip(1))

        # KL divergence loss (identical to EHRVAE)
        temp = 1 + std_z - mu_z.pow(2) - std_z.exp()
        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

        # Total loss
        total_loss = cls_loss + reg_loss + loss_kld
        
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_kld_loss", loss_kld, prog_bar=False, sync_dist=True, on_epoch=True) 
        self.log("train_cls_loss", cls_loss, prog_bar=False, sync_dist=True, on_epoch=True) 
        self.log("train_reg_loss", reg_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        
        # 记录学习率
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self.preds = defaultdict(list)
        self.targets = defaultdict(list)
        self.test_outputs = []  # 添加测试输出收集

    def validation_step(self, batch, batch_idx):
        
        # get data and label
        data = batch['data']
        label = batch['label']
        cat_feats = data['cat_feats']  # cat_feats: (b, nc)
        float_feats = data['float_feats']  # float_feats: (b, nf)
        
        # Target values
        cls_label = label['cat_feats']  # cat_feats: (b, nc)
        cls_mask = label['cat_valid_mask']  # cat_feats: (b, nc)
        reg_label = label['float_feats']  # float_feats: (b, nf)
        reg_mask = label['float_valid_mask']  # valid_mask: (b, nf)

        # Get model outputs, handle VAE and non-VAE modes differently
        y_cls, y_reg, mu_z, std_z = self.model(cat_feats, float_feats)

        # Regression loss (identical to EHRVAE)
        y_reg_loss = y_reg.reshape(-1)
        reg_label_loss = reg_label.reshape(-1)
        reg_mask_loss = reg_mask.reshape(-1)
        reg_losses = F.mse_loss(y_reg_loss, reg_label_loss, reduction='none')
        reg_loss = (reg_losses * reg_mask_loss).sum() / (reg_mask_loss.sum().clip(1))

        # Classification loss using Focal Loss for handling class imbalance
        y_cls_loss = y_cls.reshape(-1, 2)
        cls_label_loss = cls_label.reshape(-1)
        cls_mask_loss = cls_mask.reshape(-1)
        cls_losses = self.focal_loss(y_cls_loss, cls_label_loss)
        cls_loss = (cls_losses * cls_mask_loss).sum() / (cls_mask_loss.sum().clip(1))

        # KL divergence loss
        temp = 1 + std_z - mu_z.pow(2) - std_z.exp()
        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

        # Total loss
        total_loss = cls_loss + reg_loss + loss_kld
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_kld_loss", loss_kld, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("val_cls_loss", cls_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("val_reg_loss", reg_loss, prog_bar=False, sync_dist=True, on_epoch=True)

        # gather the output of each task different GPUs
        if not self.config['test'] and self.config['train']:
            tensor_to_gather = [
                cls_label.contiguous(), cls_mask.contiguous(),
                reg_label.contiguous(), reg_mask.contiguous(),
                y_cls.contiguous(), y_reg.contiguous()
            ]
            tensor_gathered = [x.cpu() for x in all_gather(tensor_to_gather)]
            cls_label = tensor_gathered[0]
            cls_mask = tensor_gathered[1]
            reg_label = tensor_gathered[2]
            reg_mask = tensor_gathered[3]
            y_cls = tensor_gathered[4]
            y_reg = tensor_gathered[5]

        # 是的，这个判断是在当前一个验证 batch 内生效的。
        # 含义是：只有当该 batch 中某个任务的有效样本数（掩码后剩余的 pp 长度）大于 2 时，
        # 才把这批数据追加进 self.preds/self.targets。
        # 这样做可以避免把极少样本的波动性很大的片段混进去，
        # 导致后续计算 AUC/F1/PCC 等不稳定。
        for i in range(len(self.cls_label_names)):
            mask = cls_mask[:, i]

            pp = y_cls[:, i, :]
            pp = pp[mask, :]

            yy = cls_label[:, i]
            yy = yy[mask]
            if len(pp) > 2:
                self.preds[f'cls_{i}'].append(pp[:, 1])
                self.targets[f'cls_{i}'].append(yy)

        for i in range(len(self.reg_label_names)):
            mask = reg_mask[:, i]

            pp = y_reg[:, i].squeeze(-1)
            pp = pp[mask]

            yy = reg_label[:, i].squeeze(-1)
            yy = yy[mask]
            if len(pp) > 2:
                self.preds[f'reg_{i}'].append(pp)
                self.targets[f'reg_{i}'].append(yy)

        # 收集测试输出数据（合并test_step功能）
        if 'pid' in batch and 'vid' in batch:
            self.test_outputs.append({
                'pid': batch['pid'], 'vid': batch['vid'],
                'cls_preds': y_cls.float().cpu().numpy(), 
                'cls_label': cls_label.cpu().numpy(), 
                'cls_mask': cls_mask.cpu().numpy(),
                'reg_preds': y_reg.float().cpu().numpy(), 
                'reg_label': reg_label.cpu().numpy(), 
                'reg_mask': reg_mask.cpu().numpy()
            })

        return total_loss

    def on_validation_epoch_end(self) -> None:
        # Classification metrics
        mauc, mf1 = [], []
        for i, k in enumerate(self.cls_label_names):
            if len(self.preds[f'cls_{i}']) != 0:
                pred = dim_zero_cat(self.preds[f'cls_{i}']).double()
                target = dim_zero_cat(self.targets[f'cls_{i}'])
                precision, recall, _ = precision_recall_curve(target.cpu().numpy(), pred.cpu().numpy())
                precision += 1e-10
                recall += 1e-10
                f1 = 2*recall*precision/(recall+precision)
                best_precision = precision[np.argmax(f1)]
                best_recall = recall[np.argmax(f1)]
                best_f1 = np.max(f1)
                mf1.append(best_f1)
                auc_score = MF.auroc(pred, target, task='binary')
                self.log(f"val_{k}_auc", auc_score, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_f1", best_f1, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_precision", best_precision, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_recall", best_recall, prog_bar=False, rank_zero_only=True)
                mauc.append(auc_score.item())
        mauc = sum(mauc) / len(mauc)
        mf1 = sum(mf1) / len(mf1)
        self.log(f"val_mauc", mauc, prog_bar=True, rank_zero_only=True)
        self.log(f"val_mf1", mf1, prog_bar=True, rank_zero_only=True)
        
        # Regression metrics
        mpcc, mr2 = [], []
        for i, k in enumerate(self.reg_label_names):
            if len(self.preds[f'reg_{i}']) != 0:
                pred = dim_zero_cat(self.preds[f'reg_{i}']).double()
                target = dim_zero_cat(self.targets[f'reg_{i}']).double()
                if pred.numel() < 2 or pred.std() == 0 or target.std() == 0:
                    continue
                self.log(f"val_{k}_mse", MF.mean_squared_error(pred, target), prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_pcc", MF.pearson_corrcoef(pred, target), prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_r2", MF.r2_score(pred, target), prog_bar=False, rank_zero_only=True)
                mpcc.append(MF.pearson_corrcoef(pred, target).item())
                mr2.append(MF.r2_score(pred, target).item())
        mpcc = sum(mpcc) / len(mpcc) if len(mpcc) != 0 else 0
        mr2 = sum(mr2) / len(mr2) if len(mr2) != 0 else 0
        self.log(f"val_mpcc", mpcc, prog_bar=True, rank_zero_only=True)
        self.log(f"val_mr2", mr2, prog_bar=True, rank_zero_only=True)

        # 保存测试结果（合并on_test_epoch_end功能）
        if len(self.test_outputs) > 0 and self.config.get('test', False):
            self._save_test_results()

    def _save_test_results(self):
        """保存测试结果到文件"""
        output_dir = Path(self.pred_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output test results
        n_cls = len(self.cls_label_names)
        n_reg = len(self.reg_label_names)
        
        all_pids = []
        all_vids = []
        
        # Initialize storage
        cls_probs = [[] for _ in range(n_cls)]
        cls_mask = [[] for _ in range(n_cls)]
        cls_labels = [[] for _ in range(n_cls)]
        
        reg_preds = [[] for _ in range(n_reg)]
        reg_masks = [[] for _ in range(n_reg)]
        reg_labels = [[] for _ in range(n_reg)]
        
        for output in tqdm(self.test_outputs):
            batch_size = len(output['pid'])
            all_pids.extend(output['pid'])
            all_vids.extend(output['vid'])

            # Process each sample
            for b_idx in range(batch_size):
                for i in range(n_cls):
                    if i < output['cls_preds'].shape[1]:
                        cls_probs[i].append(float(output['cls_preds'][b_idx, i, 1]))
                        cls_mask[i].append(int(output['cls_mask'][b_idx, i]))
                        cls_labels[i].append(int(output['cls_label'][b_idx, i]))

                for i in range(n_reg):
                    if i < output['reg_preds'].shape[1]:
                        reg_preds[i].append(float(output['reg_preds'][b_idx, i]))
                        reg_masks[i].append(int(output['reg_mask'][b_idx, i]))
                        reg_labels[i].append(float(output['reg_label'][b_idx, i]))
        
        # 检查数据长度一致性
        total_samples = len(all_pids)
        # Create DataFrame
        data_dict = {
            'pid': all_pids,
            'vid': all_vids,
        }

        for i, col in enumerate(self.cls_label_names):
            if i < len(cls_probs) and len(cls_probs[i]) == total_samples:
                data_dict[f"{col}_cls_prob"] = cls_probs[i] # 分类任务预测概率
                data_dict[f"{col}_cls_mask"] = cls_mask[i]  # 分类任务有效掩码
                data_dict[col] = cls_labels[i]            # 分类任务真实标签
        
        for i, col in enumerate(self.reg_label_names):
            if i < len(reg_preds) and len(reg_preds[i]) == total_samples:
                data_dict[f"{col}_reg_pred"] = reg_preds[i] # 回归任务预测值
                data_dict[f"{col}_reg_mask"] = reg_masks[i]   # 回归任务有效掩码
                data_dict[col] = reg_labels[i]            # 回归任务真实标签
        
        # 检查最终数据字典的长度
        # for key, value in data_dict.items():
        #     print(f"{key}: length {len(value)}")
        
        df = pd.DataFrame(data_dict)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f'test_pred.{self.global_rank}.{timestamp}.parquet'
        df.to_parquet(output_path)
        self.test_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()


class EHRFormer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(config)
        
        self.config = config
        self.data_type = config.get('data_type', '')
        self.momentum = config.get('momentum', 0.9)
        self.wd = config.get('wd', 1e-6)
        self.lr = config.get('lr', 1e-3)
        self.n_nodes = config.get('n_nodes', 1)
        self.n_gpus = config.get('n_gpus', 1)
        if isinstance(self.n_gpus, list):
            self.n_gpus = len(self.n_gpus)
        self.n_epoch = config.get('n_epoch', None)

        self.cls_label_names = config.get('cls_label_names', [])
        self.reg_label_names = config.get('reg_label_names', [])
        
        # create time-dependent 2D model, 
        # it inherit time-independent 1D model from EHRVAE1D
        # then use GPT2 to form time-dependent embedding
        freezed_pretrained_for_aug = config.get('freezed_pretrained_for_aug', None)
        if freezed_pretrained_for_aug is not None:
            self.model_frozen = EHRGPT2(config)
            state_dict = torch.load(freezed_pretrained_for_aug, weights_only=False)['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.model_frozen.load_state_dict(state_dict, strict=False)
            print(f"Missing keys in loading freezed_pretrained_for_aug: {missing_keys}")
            print(f"Unexpected keys in loading freezed_pretrained_for_aug: {unexpected_keys}")
            for param in self.model_frozen.parameters():
                param.requires_grad = False
        else: self.model_frozen = None
        self.model = EHRGPT2(config)

        self.focal_loss = FocalLoss(alpha=0.25, gamma=2, reduction='none')
        self.pred_folder = config.get('pred_folder', None)

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )
        scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                                  first_cycle_steps=self.n_epoch,
                                                  max_lr=self.lr, 
                                                  min_lr=1e-8, 
                                                  warmup_steps=int(self.n_epoch * 0.1))
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        scheduler.step()

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def on_train_start(self):
        self.loggers[0].log_hyperparams(self.config)

    def forward(self, batch):
        """
        Inference-style forward used by the MCP server.
        Expects keys: category_feats (B, seq_len, n_cat), float_feats (B, seq_len, n_float), mask (B, seq_len).
        Builds simple positional indices and zero diag feats to mimic the validation data flow.
        Returns a dict with cls_logits/reg_preds following server expectations.
        """
        cat_feats = batch['category_feats']            # (B, seq_len, n_cat)
        float_feats = batch.get('float_feats', None)    # (B, seq_len, n_float) or None
        valid_mask = batch.get('mask', None)            # (B, seq_len)

        # transpose to (B, n_cat, seq_len) as the model expects
        cat_feats = cat_feats.permute(0, 2, 1)
        if float_feats is not None:
            float_feats = float_feats.permute(0, 2, 1)

        B, seq_len = cat_feats.shape[0], cat_feats.shape[2]
        device = cat_feats.device

        # simple positional ids 0..seq_len-1, clamped in model to seq_max_len
        time_index = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

        # diagnostic features absent at inference: supply zeros with correct shape
        n_diag = getattr(self.model, 'n_diag_feats', 0)
        diag_feats = torch.zeros(B, n_diag, seq_len, device=device, dtype=cat_feats.dtype)

        # forward through underlying model
        y_cls, y_reg = self.model(cat_feats, float_feats, time_index, diag_feats)

        # stack task outputs for easier downstream handling
        if isinstance(y_cls, (list, tuple)):
            cls_logits = torch.stack(y_cls, dim=1)  # (B, n_cls, seq_len, n_classes_per_task)
        else:
            cls_logits = y_cls

        if isinstance(y_reg, (list, tuple)):
            reg_preds = torch.stack(y_reg, dim=1)  # (B, n_reg, seq_len, 1?)
        else:
            reg_preds = y_reg

        return {
            'cls_logits': cls_logits,
            'reg_preds': reg_preds
        }

    def training_step(self, batch, batch_idx):
        
        # get data and label
        data = batch['data']
        label = batch['label']
        cat_feats = data['cat_feats']  
        float_feats = data['float_feats']
        # valid_mask and time_index is important for GPT2
        valid_mask = data['valid_mask'] # (B, pad_visit_len)
        time_index = data['time_index'] # (B, pad_visit_len)
        max_seq_len = cat_feats.shape[2]
        
        # Target values
        cls_label = label['cls']['values'] # (B, n_cls, pad_visit_len)
        cls_mask = valid_mask.unsqueeze(1).expand_as(label['cls']['masks']) & label['cls']['masks']
        reg_label = label['reg']['values'] # (B, n_reg, pad_visit_len)
        reg_mask = valid_mask.unsqueeze(1).expand_as(label['reg']['masks']) & label['reg']['masks']
        n_cls = cls_label.shape[1]
        n_reg = reg_label.shape[1]

        # Get model outputs, handle VAE and non-VAE modes differently
        diag_feats = batch['label']['cls']['values'][:,:len(self.cls_label_names)//2,:]
        if self.model_frozen is not None:
            _, y_reg_frozen = self.model_frozen(cat_feats, float_feats, time_index, diag_feats)
            reg_preds_frozen = torch.stack([y_reg_frozen[i].reshape(-1, max_seq_len) for i in range(n_reg)], dim=0)
            reg_preds_frozen = reg_preds_frozen.permute(1, 2, 0)
        else: reg_preds_frozen = None
        y_cls, y_reg = self.model(cat_feats, float_feats, time_index, diag_feats, reg_preds_frozen)

        cls_preds = [y.reshape(-1, y.shape[-1]) for y in y_cls]
        cls_labels = cls_label.reshape(cls_label.shape[0], n_cls, -1).permute(1, 0, 2)
        cls_labels = [l.reshape(-1) for l in cls_labels]
        cls_masks = cls_mask.reshape(cls_mask.shape[0], n_cls, -1).permute(1, 0, 2)
        cls_masks = [m.reshape(-1) for m in cls_masks]

        # Safe classification loss calculation
        cls_groups = group_by_M_dimension(cls_preds, cls_labels, cls_masks)
        cls_loss = 0
        for task_idx, (pred, label, mask) in enumerate(cls_groups.values()):
            # Ensure label values in valid range [0, num_classes-1]
            num_classes = pred.shape[-1]
            
            # 多分类任务：使用 CrossEntropyLoss
            if num_classes > 2:
                valid_indices = (label >= 0) & (label < num_classes) & mask.bool()
                if valid_indices.sum() > 0:
                    valid_pred = pred[valid_indices]
                    valid_label = label[valid_indices]
                    loss_i = F.cross_entropy(valid_pred, valid_label, reduction='mean')
                    cls_loss += loss_i
            else:  # 二分类任务：使用 FocalLoss
                valid_indices = (label >= 0) & (label < num_classes) & mask.bool()
                if valid_indices.sum() > 0:
                    valid_pred = pred[valid_indices]
                    valid_label = label[valid_indices]
                    valid_mask = mask[valid_indices]
                    loss_i = self.focal_loss(valid_pred, valid_label)
                    cls_loss += (loss_i * valid_mask).sum() / valid_mask.sum().clip(1)
        
        # Safe regression loss calculation
        reg_preds = torch.stack([y_reg[i].reshape(-1) for i in range(n_reg)], dim=0)
        reg_labels = reg_label.reshape(reg_label.shape[0], n_reg, -1).permute(1, 0, 2).reshape(n_reg, -1)
        reg_masks = reg_mask.reshape(reg_mask.shape[0], n_reg, -1).permute(1, 0, 2).reshape(n_reg, -1)
        reg_losses = F.mse_loss(reg_preds, reg_labels, reduction='none')
        reg_loss = (reg_losses * reg_masks).sum() / reg_masks.sum().clip(1)

        # Calculate total losses
        total_loss = cls_loss + reg_loss
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_cls_loss", cls_loss, prog_bar=True, sync_dist=True, on_epoch=True) 
        self.log("train_reg_loss", reg_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        # 记录学习率
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self.preds = defaultdict(list)
        self.targets = defaultdict(list)
        self.test_outputs = []  # 添加测试输出收集
    
    def validation_step(self, batch, batch_idx):
        
        # get data and label
        data = batch['data']
        label = batch['label']
        cat_feats = data['cat_feats']  
        float_feats = data['float_feats']
        max_seq_len = cat_feats.shape[2]  

        # valid_mask and time_index is important for GPT2
        valid_mask = data['valid_mask']  
        time_index = data['time_index']  

        # Target values
        cls_label = label['cls']['values']
        cls_mask = valid_mask.unsqueeze(1).expand_as(label['cls']['masks']) & label['cls']['masks']
        reg_label = label['reg']['values']
        reg_mask = valid_mask.unsqueeze(1).expand_as(label['reg']['masks']) & label['reg']['masks']
        n_cls = cls_label.shape[1]
        n_reg = reg_label.shape[1]
        

        # Get model outputs, handle VAE and non-VAE modes differently
        # maybenot compatible here
        diag_feats = batch['label']['cls']['values'][:,:len(self.cls_label_names)//2,:]
        if self.model_frozen is not None:
            _, y_reg_frozen = self.model_frozen(cat_feats, float_feats, time_index, diag_feats)
            reg_preds_frozen = torch.stack([y_reg_frozen[i].reshape(-1, max_seq_len) for i in range(n_reg)], dim=0)
            reg_preds_frozen = reg_preds_frozen.permute(1, 2, 0)
        else: reg_preds_frozen = None
        y_cls, y_reg = self.model(cat_feats, float_feats, time_index, diag_feats, reg_preds_frozen)

        # Classification loss calculation with safety checks
        cls_preds = [y.reshape(-1, y.shape[-1]) for y in y_cls]
        cls_labels = cls_label.reshape(cls_label.shape[0], n_cls, -1).permute(1, 0, 2)
        cls_labels = [l.reshape(-1) for l in cls_labels]
        cls_masks = cls_mask.reshape(cls_mask.shape[0], n_cls, -1).permute(1, 0, 2)
        cls_masks = [m.reshape(-1) for m in cls_masks]

        # Safe classification loss calculation
        cls_groups = group_by_M_dimension(cls_preds, cls_labels, cls_masks)
        cls_loss = 0
        for task_idx, (pred, label, mask) in enumerate(cls_groups.values()):
            # Ensure label values in valid range [0, num_classes-1]
            num_classes = pred.shape[-1]
            
            # 多分类任务：使用 CrossEntropyLoss
            if num_classes > 2:
                valid_indices = (label >= 0) & (label < num_classes) & mask.bool()
                if valid_indices.sum() > 0:
                    valid_pred = pred[valid_indices]
                    valid_label = label[valid_indices]
                    loss_i = F.cross_entropy(valid_pred, valid_label, reduction='mean')
                    cls_loss += loss_i
            else:  # 二分类任务：使用 FocalLoss
                valid_indices = (label >= 0) & (label < num_classes) & mask.bool()
                if valid_indices.sum() > 0:
                    valid_pred = pred[valid_indices]
                    valid_label = label[valid_indices]
                    valid_mask = mask[valid_indices]
                    loss_i = self.focal_loss(valid_pred, valid_label)
                    cls_loss += (loss_i * valid_mask).sum() / valid_mask.sum().clip(1)
        
        # Safe regression loss calculation
        reg_loss = 0
        reg_preds = torch.stack([y_reg[i].reshape(-1) for i in range(n_reg)], dim=0)
        reg_labels = reg_label.reshape(reg_label.shape[0], n_reg, -1).permute(1, 0, 2).reshape(n_reg, -1)
        reg_masks = reg_mask.reshape(reg_mask.shape[0], n_reg, -1).permute(1, 0, 2).reshape(n_reg, -1)
        reg_losses = F.mse_loss(reg_preds, reg_labels, reduction='none')
        reg_loss = (reg_losses * reg_masks).sum() / reg_masks.sum().clip(1)

        total_loss = cls_loss + reg_loss
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_cls_loss", cls_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("val_reg_loss", reg_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        
        # for distributed metrics calculation
        if not self.config['test'] and self.config['train']:
            tensor_to_gather = [
                cls_label.contiguous(), cls_mask.contiguous(), 
                reg_label.contiguous(), reg_mask.contiguous()
            ] + y_cls + y_reg
            tensor_gathered = [x.cpu() for x in all_gather(tensor_to_gather)]
            cls_label = tensor_gathered[0]
            cls_mask = tensor_gathered[1]
            reg_label = tensor_gathered[2]
            reg_mask = tensor_gathered[3]
            y_cls = tensor_gathered[4:4+len(y_cls)]
            y_reg = tensor_gathered[4+len(y_cls):]

        # Safe access to classification predictions with boundary checks
        for i in range(len(self.cls_label_names)):
            if i < len(y_cls) and i < cls_mask.shape[1]:
                mask = cls_mask[:, i, :]
                
                pp = y_cls[i]
                pp = pp[mask]
                pp = torch.softmax(pp, dim=-1)

                yy = cls_label[:, i, :]
                yy = yy[mask]
                
                if len(pp) > 2:
                    self.preds[f'cls_{i}'].append(pp)
                    self.targets[f'cls_{i}'].append(yy)

        # Safe access to regression predictions with boundary checks
        for i in range(len(self.reg_label_names)):
            if i < len(y_reg) and i < reg_mask.shape[1]:
                mask = reg_mask[:, i, :]

                pp = y_reg[i].squeeze(-1)
                pp = pp[mask]

                yy = reg_label[:, i, :].squeeze(-1)  # 添加 squeeze(-1)
                yy = yy[mask]
                if len(pp) > 2:
                    self.preds[f'reg_{i}'].append(pp)
                    self.targets[f'reg_{i}'].append(yy)
        
        # 收集测试输出数据（合并test_step功能）
        if 'pid' in batch:
            self.test_outputs.append({
                'pid': batch['pid'],
                'cls_label': cls_label, 'cls_mask': cls_mask, 'cls_preds': y_cls, 
                'reg_label': reg_label, 'reg_mask': reg_mask, 'reg_preds': y_reg,
            })
        
        return total_loss

    def on_validation_epoch_end(self) -> None:
        # classification metrics (分离二分类和多分类)
        mauc, mf1, maucpr = [], [], []  # 二分类指标
        mtop1_acc, mmicro_auc = [], []  # 多分类指标
        
        for i, k in enumerate(self.cls_label_names):
            if len(self.preds[f'cls_{i}']) != 0:
                pred = dim_zero_cat(self.preds[f'cls_{i}']).float()
                target = dim_zero_cat(self.targets[f'cls_{i}'])
                if pred.shape[-1] == 2: # binary classification
                    pred = pred[..., 1]
                    precision, recall, _ = precision_recall_curve(target.cpu().numpy(), pred.cpu().numpy())
                    precision += 1e-10
                    recall += 1e-10
                    f1 = 2*recall*precision/(recall+precision)
                    best_precision = precision[np.argmax(f1)]
                    best_recall = recall[np.argmax(f1)]
                    best_f1 = np.max(2*recall*precision/(recall+precision))
                    auroc_score = MF.auroc(pred, target, task='binary')
                    aucpr_score = average_precision_score(target.cpu().numpy(), pred.cpu().numpy())
                    self.log(f"val_{k}_auc", auroc_score, prog_bar=False, rank_zero_only=True)
                    self.log(f"val_{k}_f1", best_f1, prog_bar=False, rank_zero_only=True)
                    self.log(f"val_{k}_aucpr", aucpr_score, prog_bar=False, rank_zero_only=True)
                    mf1.append(best_f1)
                    mauc.append(auroc_score)
                    maucpr.append(aucpr_score)
                else:  # multiclass classification
                    num_classes = pred.shape[-1]
                    top1_acc = MF.accuracy(pred, target, task='multiclass', num_classes=num_classes, top_k=1)
                    target_onehot = label_binarize(target.cpu().detach().numpy(), classes=range(num_classes))
                    pred_probs = pred.cpu().detach().numpy()
                    micro_auc = roc_auc_score(target_onehot.ravel(), pred_probs.ravel())
                    self.log(f"val_{k}_top1_acc", top1_acc, prog_bar=False, rank_zero_only=True)
                    self.log(f"val_{k}_micro_auc", micro_auc, prog_bar=False, rank_zero_only=True)
                    # Top5只在类别数>5时有意义
                    if num_classes > 5:
                        top5_acc = MF.accuracy(pred, target, task='multiclass', num_classes=num_classes, top_k=5)
                        self.log(f"val_{k}_top5_acc", top5_acc, prog_bar=False, rank_zero_only=True)
                    mtop1_acc.append(top1_acc.item())
                    mmicro_auc.append(micro_auc)
            else:
                self.log(f"val_{k}_auc", torch.nan, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_aucpr", torch.nan, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_f1", torch.nan, prog_bar=False, rank_zero_only=True)
        
        # 二分类平均指标
        if len(mauc) > 0:
            mauc_avg = sum(mauc) / len(mauc)
            mf1_avg = sum(mf1) / len(mf1)
            maucpr_avg = sum(maucpr) / len(maucpr)
            self.log(f"val_binary_mauc", mauc_avg, prog_bar=True, rank_zero_only=True)
            self.log(f"val_binary_mf1", mf1_avg, prog_bar=True, rank_zero_only=True)
            self.log(f"val_binary_maucpr", maucpr_avg, prog_bar=True, rank_zero_only=True)
        
        # 多分类平均指标
        if len(mtop1_acc) > 0:
            mtop1_avg = sum(mtop1_acc) / len(mtop1_acc)
            mmicro_auc_avg = sum(mmicro_auc) / len(mmicro_auc)
            self.log(f"val_multi_top1_acc", mtop1_avg, prog_bar=True, rank_zero_only=True)
            self.log(f"val_multi_micro_auc", mmicro_auc_avg, prog_bar=True, rank_zero_only=True)
        
        # 综合平均指标（兼容旧版checkpoint）
        all_metrics = mauc + [m for m in mtop1_acc]  # 合并二分类AUC和多分类Top1
        if len(all_metrics) > 0:
            overall_mauc = sum(all_metrics) / len(all_metrics)
            self.log(f"val_mauc", overall_mauc, prog_bar=True, rank_zero_only=True)
        if len(mf1) > 0:
            self.log(f"val_mf1", mf1_avg, prog_bar=True, rank_zero_only=True)

        # regression metrics
        mpcc, mr2 = [], []
        for i, k in enumerate(self.reg_label_names):
            if len(self.preds[f'reg_{i}']) >= 2:
                device = self.preds[f'reg_{i}'][0].device
                pred = dim_zero_cat(self.preds[f'reg_{i}']).double().to(device)
                target = dim_zero_cat(self.targets[f'reg_{i}']).double().to(device)
                self.log(f"val_{k}_mse", MF.mean_squared_error(pred, target), prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_pcc", MF.pearson_corrcoef(pred, target), prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_r2", MF.r2_score(pred, target), prog_bar=False, rank_zero_only=True)
                mpcc.append(MF.pearson_corrcoef(pred, target))
                mr2.append(MF.r2_score(pred, target))
                if self.config.get("test", False):
                    os.makedirs(self.pred_folder, exist_ok=True)
                    save_path = os.path.join(self.pred_folder, f"val_{k.replace('/', '_')}_preds.npy")
                    np.save(save_path, np.concatenate([pred.float().cpu().numpy().flatten().reshape(-1,1), target.float().cpu().numpy().flatten().reshape(-1,1)], axis=-1))
                    print('save at ',save_path)
        mpcc = sum(mpcc) / len(mpcc) if len(mpcc) != 0 else 0
        mr2 = sum(mr2) / len(mr2) if len(mr2) != 0 else 0
        self.log(f"val_mpcc", mpcc, prog_bar=True, rank_zero_only=True)
        self.log(f"val_mr2", mr2, prog_bar=True, rank_zero_only=True)

        # 保存测试结果（合并on_test_epoch_end功能）
        if len(self.test_outputs) > 0 and self.config.get('test', False):
            self._save_test_results_ehrformer()

    def _save_test_results_ehrformer(self):
        """保存EHRFormer测试结果到文件"""
        output_dir = Path(self.pred_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_cls = len(self.cls_label_names)
        n_reg = len(self.reg_label_names)

        pred_cls = [[] for _ in range(n_cls)]
        cls_mask = [[] for _ in range(n_cls)]
        cls_label = [[] for _ in range(n_cls)]
        for i in range(len(self.test_outputs)):
            for j in range(n_cls):
                pred_cls[j].append(self.test_outputs[i]['cls_preds'][j])
                cls_mask[j].append(self.test_outputs[i]['cls_mask'][:, j, :])
                cls_label[j].append(self.test_outputs[i]['cls_label'][:, j, :])
        pred_cls = [torch.softmax(torch.cat(x, dim=0), dim=-1).to('cpu') for x in pred_cls]
        cls_mask = [torch.cat(x, dim=0).to('cpu') for x in cls_mask]
        cls_label = [torch.cat(x, dim=0).to('cpu') for x in cls_label]

        pred_reg = [[] for _ in range(n_reg)]
        reg_mask = [[] for _ in range(n_reg)]
        reg_label = [[] for _ in range(n_reg)]
        for i in range(len(self.test_outputs)):
            for j in range(n_reg):
                pred_reg[j].append(self.test_outputs[i]['reg_preds'][j])
                reg_mask[j].append(self.test_outputs[i]['reg_mask'][:, j, :])
                reg_label[j].append(self.test_outputs[i]['reg_label'][:, j, :])
        pred_reg = [torch.cat(x, dim=0).to('cpu') for x in pred_reg]
        reg_mask = [torch.cat(x, dim=0).to('cpu') for x in reg_mask]
        reg_label = [torch.cat(x, dim=0).to('cpu') for x in reg_label]

        # Collect pid as a flat Python list; handle CUDA tensors and numpy arrays safely
        pid_chunks = []
        for x in self.test_outputs:
            p = x['pid']
            if isinstance(p, torch.Tensor): p = p.detach().cpu().tolist()
            elif isinstance(p, np.ndarray): p = p.tolist()
            else: p = list(p)
            pid_chunks.append(p)
        pid = reduce(lambda a, b: a + b, pid_chunks, [])

        n_sample = len(pid)
        df = pd.DataFrame(pid, columns=['pid'])
        for i, col in enumerate(tqdm(self.cls_label_names)):
            if pred_cls[i].shape[-1] == 2:
                tmp1 = pd.DataFrame([{f"{col}_cls_prob": pred_cls[i][j, :, 1].float().numpy()} for j in range(n_sample)])
                tmp2 = pd.DataFrame([{f"{col}_cls_mask": cls_mask[i][j, :].float().numpy()} for j in range(n_sample)])
                tmp3 = pd.DataFrame([{col: cls_label[i][j, :].float().numpy()} for j in range(n_sample)])
                df = pd.concat([df, tmp1, tmp2, tmp3], axis=1)
            else:
                tmp1 = pd.DataFrame([{f"{col}_cls_probs": batch_top5_encode(pred_cls[i][j, :, :].float().numpy().astype(np.int32))} for j in range(n_sample)])
                tmp2 = pd.DataFrame([{f"{col}_clsmask": cls_mask[i][j, :].float().numpy()} for j in range(n_sample)])
                tmp3 = pd.DataFrame([{col: cls_label[i][j, :].float().numpy()} for j in range(n_sample)])
                df = pd.concat([df, tmp1, tmp2, tmp3], axis=1)
        for i, col in enumerate(self.reg_label_names):
            tmp1 = pd.DataFrame([{f"{col}_reg_pred": pred_reg[i][j, :, 0].float().numpy()} for j in range(n_sample)])
            tmp2 = pd.DataFrame([{f"{col}_reg_mask": reg_mask[i][j, :].float().numpy()} for j in range(n_sample)])
            tmp3 = pd.DataFrame([{col: reg_label[i][j, :].float().numpy()} for j in range(n_sample)])
            df = pd.concat([df, tmp1, tmp2, tmp3], axis=1)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f'test_pred.rank_{self.global_rank}.parquet'
        df.to_parquet(output_path)
        self.test_outputs.clear()

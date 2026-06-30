"""Pretraining script."""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from scripts.ehr_dataset_chunk import EHRDataModule1D
from scripts.ehr_model_script import EHRVAE1D_pretrain
from scripts.setup_multistage_config import setup_multistage_config
from pathlib import Path
import torch
from pytorch_lightning import Trainer
from utils.utils import *   
import logging
import warnings
torch.set_float32_matmul_precision('medium')
set_seed(1)

# 设置日志级别，避免警告信息污染 CSV，但允许模型总结显示
# logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
# warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")


def main(config):
    debug = config['debug']
    output_dir = Path(config['output_dir'])
    
    # Setup loggers and callbacks
    loggers, callbacks, _ = \
        setup_loggers_and_callbacks(config, output_dir, debug)
    csv_root_path = loggers[0].log_dir 
    config['pred_folder'] = os.path.join(csv_root_path, config['pred_folder']) # make sure pred folder is the same as the csv logger

    # dataset and model initialization
    data_module = EHRDataModule1D(config)
    data_module.setup()
    config = setup_multistage_config(config)
    model_module = EHRVAE1D_pretrain(config)
    
    
    if config['train']:
        trainer = Trainer(
            accelerator='gpu', 
            devices=config['n_gpus'], 
            max_epochs=config['n_epoch'],
            logger=loggers,
            callbacks=callbacks,
            strategy='ddp_find_unused_parameters_true',
            precision='bf16-mixed',
            sync_batchnorm=True,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=10,
            check_val_every_n_epoch=config.get('check_val_every_n_epoch', 5))  # 从配置文件读取验证频率（每N个epoch验证一次）
        trainer.fit(model_module, datamodule=data_module)
    if config['test']:
        trainer = Trainer(
            inference_mode=True,
            logger=loggers,
            accelerator ='gpu', 
            precision='bf16-mixed',
            devices=[config['n_gpus'][0]])
        if config['ckpt_path'] is not None:
            ckpt = torch.load(config['ckpt_path'], map_location='cpu', weights_only=False)
            model_module.load_state_dict(ckpt['state_dict'], strict=False)
        trainer.validate(model_module, datamodule=data_module)

if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description='EHRFormer pretraining with flexible config override',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--config', '-c', type=str, 
        default='configs/pretrain_copd_woEHR_wz_filter_conversion_selected.json',
        help='Path to JSON configuration file (default: configs/pretrain_mother.json)')
    
    # Parse known arguments to allow for flexible config overrides
    args, unknown = parser.parse_known_args()
    config = json_load(args.config)
    print(f"Loaded configuration from: {args.config}")
    overrides = parse_unknown_args(unknown)
    if overrides:
        print("\nApplying configuration overrides:")
        config = override_config(config, overrides)
    
    # Load feature and task info
    config['n_gpus'] = [0]
    config['feat_info'] = json_load(config['feat_info_path'])
    config['task_info'] = json_load(config['task_info_path'])
    config['n_category_feats'] = len(config['feat_info']['category_cols'])
    config['n_float_feats'] = len(config['feat_info']['float_cols'])

    print("\nFinal configuration:")
    print(json.dumps({k: str(v) if not isinstance(v, (dict, list)) else v for k, v in config.items()}, indent=2, default=str))
    print("\n" + "="*50 + "\n")
    
    main(config)

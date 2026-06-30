"""Finetuning script."""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from scripts.ehr_dataset_chunk import EHRDataModule2D
from scripts.ehr_model_script import EHRFormer
from scripts.setup_multistage_config import setup_multistage_config
from pathlib import Path
from transformers import GPT2Config
import torch
from pytorch_lightning import Trainer
from utils.utils import *
import os, warnings, logging
# 设置日志级别，避免警告信息污染 CSV
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
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
    config['pred_folder'] = os.path.join(csv_root_path,config['pred_folder']) # make sure pred folder is the same as the csv logger

    # dataset and model initialization
    data_module = EHRDataModule2D(config)
    data_module.setup()
    config = setup_multistage_config(config)
    model_module = EHRFormer(config)
    
    # need add safety of GPT2 to load
    torch.serialization.add_safe_globals([GPT2Config])
    trainer = Trainer(
        inference_mode=True,
        logger=loggers,
        accelerator ='gpu', 
        precision='bf16-mixed',
        devices=config['n_gpus'])
    if config['ckpt_path'] is not None:
        ckpt = torch.load(config['ckpt_path'], map_location='cpu', weights_only=False)
        model_module.load_state_dict(ckpt['state_dict'], strict=False)
        ############# save gpt2 embed and register hooks ######################
        if config.get('save_gpt2_embed', False):
            print('save gpt2 embed and register hooks')
            from utils.plot.forward_hook_utils import register_gpt2_output_hooks, register_bert_output_hooks
            gpt2_embed_list, bert_embed_list = [], []
            register_gpt2_output_hooks(model_module, gpt2_embed_list)
            register_bert_output_hooks(model_module, bert_embed_list)
        #######################################################################
        trainer.validate(model_module, datamodule=data_module)
        ##################### save gpt2 embed #################################
        if config.get('save_gpt2_embed', False):
            gpt2_embed = torch.concat(gpt2_embed_list, dim=0).float().cpu().numpy()
            np.save(os.path.join(config['pred_folder'], 'gpt2_embed.npy'), gpt2_embed)
            bert_embed = torch.concat(bert_embed_list, dim=0).float().cpu().numpy()
            np.save(os.path.join(config['pred_folder'], 'bert_embed.npy'), bert_embed)


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description='EHRFormer finetuning with flexible config override',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--config', '-c', type=str, 
        default='configs/finetune_mother_labonly.json',
        help='Path to JSON configuration file (default: configs/finetune_child.json)')
    
    # Parse known arguments to allow for flexible config overrides
    args, unknown = parser.parse_known_args()
    config = json_load(args.config)
    print(f"Loaded configuration from: {args.config}")
    overrides = parse_unknown_args(unknown)
    if overrides:
        print("\nApplying configuration overrides:")
        config = override_config(config, overrides)
    
    config['n_gpus'] = [0]
    config['valid_folds'] = [0]
    config['test_folds'] = [0]
    config['save_gpt2_embed'] = True
    config['train'] = False
    config['test'] = True
    config['ckpt_path'] = "output/finetune_mother_labonly/train/version_0/checkpoint/last.ckpt"
    # Load feature information
    config['feat_info'] = json_load(config['feat_info_path'])
    config['task_info'] = json_load(config['task_info_path'])
    config['n_category_feats'] = len(config['feat_info']['category_cols'])
    config['n_float_feats'] = len(config['feat_info']['float_cols'])
    
    # ######################### finetune mode use GPT2 #########################
    config['transformer'] = GPT2Config.from_pretrained('gpt2')
    config['transformer'].n_positions = 8192
    config['transformer'].attn_implementation='flash_attention_2'
    #############################################################################
    
    print("\nFinal configuration:")
    print(json.dumps({k: str(v) if not isinstance(v, (dict, list)) else v for k, v in config.items()}, indent=2, default=str))
    print("\n" + "="*50 + "\n")
    
    main(config)

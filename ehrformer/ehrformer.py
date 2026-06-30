"""EHRFormer model."""
import torch.nn as nn
import torch
from einops import rearrange
from transformers import BertConfig
# from transformers import GPT2Model, BertModel
# from transformers.models.bert.modeling_bert import BertEncoder
from flash_attn.models.bert import BertEncoder, BertModel
from utils.patch_flash_attn import GPTModel


class MultiTaskHeadsharedhead(nn.Module):
    """1D Multi-task head for feat_info features (shared heads)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['output_dim']
        self.proj_dim = config['proj_dim']
        self.n_cls = len(config.get('cls_label_names', []))
        self.n_reg = len(config.get('reg_label_names', []))

        
        self.cls_fc = nn.Sequential(
            nn.Linear(self.output_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, 2)
        )
        self.reg_fc = nn.Sequential(
            nn.Linear(self.output_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, 1)
        )

    def forward(self, h):
        h_features = h[:, 1:, :]  # Skip CLS token: (B, seq_len-1, hidden_dim)
        # Split into classification and regression features
        h_cls = h_features[:, :self.n_cls, :]  # (B, n_cls, hidden_dim)
        h_reg = h_features[:, self.n_cls:self.n_cls+self.n_reg, :]  # (B, n_reg, hidden_dim)
        y_cls = self.cls_fc(h_cls)
        y_reg = self.reg_fc(h_reg).squeeze(2)
        
        return y_cls, y_reg


class MultiTaskHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['output_dim']
        self.proj_dim = config['proj_dim']
        self.n_cls = len(config.get('cls_label_names', []))
        self.n_reg = len(config.get('reg_label_names', []))
        
        # Classification task heads
        self.cls_fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.output_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, 2)
        ) for _ in range(self.n_cls)])
        
        # Regression task heads
        if self.n_reg > 0:
            self.reg_fcs = nn.ModuleList([nn.Sequential(
                nn.Linear(self.output_dim, self.proj_dim),
                nn.ReLU(),
                nn.Linear(self.proj_dim, 1)
            ) for _ in range(self.n_reg)])
        else:
            self.reg_fcs = nn.ModuleList()

    def forward(self, x):

        B = x.shape[0]
        y_cls_results = []
        y_reg_results = []

        # Classification predictions
        for i, cls_fc in enumerate(self.cls_fcs):
            y_cls = cls_fc(x)
            y_cls_results.append(y_cls)
        
        # Regression predictions
        for i, reg_fc in enumerate(self.reg_fcs):
            y_reg = reg_fc(x)
            y_reg_results.append(y_reg)
        
        return y_cls_results, y_reg_results


class ImprovedMultiTaskHead(nn.Module):
    """改进的多任务头：更深的网络 + LayerNorm + Dropout，增强表达能力和泛化性"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['output_dim']
        self.proj_dim = config.get('proj_dim', 512)  # 默认使用更大的投影维度
        self.n_cls = len(config.get('cls_label_names', []))
        self.n_reg = len(config.get('reg_label_names', []))
        
        # 获取每个分类任务的类别数（默认2分类）
        self.cls_num_classes = config.get('cls_num_classes', [2] * self.n_cls)
        
        # 分类任务头：3层MLP with LayerNorm + Dropout，支持不同类别数
        self.cls_fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.output_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.GELU(),  # GELU 比 ReLU 更平滑
            nn.Dropout(0.3),
            nn.Linear(self.proj_dim, self.proj_dim // 2),
            nn.LayerNorm(self.proj_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.proj_dim // 2, self.cls_num_classes[i])
        ) for i in range(self.n_cls)])
        
        # 回归任务头：同样的架构
        if self.n_reg > 0:
            self.reg_fcs = nn.ModuleList([nn.Sequential(
                nn.Linear(self.output_dim, self.proj_dim),
                nn.LayerNorm(self.proj_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(self.proj_dim, self.proj_dim // 2),
                nn.LayerNorm(self.proj_dim // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.proj_dim // 2, 1)
            ) for _ in range(self.n_reg)])
        else:
            self.reg_fcs = nn.ModuleList()

    def forward(self, x):
        B = x.shape[0]
        y_cls_results = []
        y_reg_results = []

        # Classification predictions
        for i, cls_fc in enumerate(self.cls_fcs):
            y_cls = cls_fc(x)
            y_cls_results.append(y_cls)
        
        # Regression predictions
        for i, reg_fc in enumerate(self.reg_fcs):
            y_reg = reg_fc(x)
            y_reg_results.append(y_reg)
        
        return y_cls_results, y_reg_results

# EHRGPT2 use GPT2 to embed the labtest2D data, use dataset2Dmode
# EHRGPT2 inherit EHRBert encoder from EHRVAE1D
# then use GPT2 to form time-dependent embedding
class EHRGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        assert self.config['mode'] == 'finetune', "EHRGPT2 only supports finetune mode"

        # 诊断特征相关配置
        self.use_info_EHR = config.get('use_info_EHR', False)
        if self.use_info_EHR:
            # 从 cls_label_names 中计算诊断特征数量，这里注意 data generation的2D模式把pregnancy_status放在了第一个位置
            self.diag_cols = config.get('cls_label_names', [])
            self.diag_cols = [x for x in self.diag_cols if x.startswith('c_cls_labels_diag_')]
            self.n_diag_feats = len(self.diag_cols)
            self.diag_start_idx = config['n_category_feats']  # 诊断特征在原始分类特征之后
        else:
            self.diag_cols = []
            self.n_diag_feats = 0
            self.diag_start_idx = 0
        
        # 根据 use_info_LAB 决定是否包含labtest特征
        self.use_info_LAB = self.config.get('use_info_LAB', True)
        if not self.use_info_LAB:
            temp_len_float = len(self.config['feat_info']['float_cols'])
            temp_len_cat = len(self.config['feat_info']['category_cols'])
        else:
            temp_len_float = 0
            temp_len_cat = 0
            
        # update the config and print the result
        self.config['n_category_feats'] = self.config['n_category_feats'] + len(self.diag_cols) - temp_len_cat
        self.config['n_float_feats'] = self.config['n_float_feats'] - temp_len_float
        print(f"EHRGPT2: EHR: Added {len(self.diag_cols)} diagnosis features to categorical features")
        print(f"EHRGPT2: LAB: Minus {temp_len_float} float features and {temp_len_cat} category features of labtest")
        print(f"EHRGPT2: Final n_category_feats: {self.config['n_category_feats']}, Final n_float_feats: {self.config['n_float_feats']}")

        # 2D模式需要pooler_output，所以设置pool_emb=True
        config_2d = {**config, 'pool_emb': True}
        self.ehr_embed = EHRBert(config_2d)
        self.bert_config = self.ehr_embed.bert_config
        self.transformer = GPTModel(config['transformer'])
        # self.head = MultiTaskHead(config)
        self.head = ImprovedMultiTaskHead(config)

        # previous trained knowledge
        if config.get('freezed_pretrained_for_aug', None):
            input_dim = len(self.config.get('reg_label_names', []))
            self.mapping = nn.Sequential(
                nn.Linear(input_dim, config['output_dim']//2),
                nn.ReLU(),
                nn.Linear(config['output_dim']//2, config['output_dim']))
    
    def forward(self, cat_feats, float_feats, time_index, diag_feats, reg_preds_frozen=None):
        # 微调阶段：对诊断特征应用时间步限制
        if self.use_info_EHR and self.n_diag_feats > 0:
            B, n_cat, seq_len = cat_feats.shape 
            cat_feats = torch.cat([cat_feats, diag_feats], dim=1)  # (B, n_cat + n_diag_feats, s
            for t in range(seq_len):
                cat_feats[:, self.diag_start_idx:self.diag_start_idx+self.n_diag_feats, t] = -1
        # 根据 use_info_LAB 决定是否包含labtest特征
        if not self.use_info_LAB:
            cat_feats = cat_feats[:, self.diag_start_idx:self.diag_start_idx+self.n_diag_feats, :]
            float_feats = None

        # ft_emb: (B, seq_len, hidden_dim) seq_len is pad_visit_len
        ft_emb = self.ehr_embed(cat_feats, float_feats)
        # 使用非原地 clamp 避免共享存储导致的 inplace 错误
        h = self.transformer(inputs_embeds=ft_emb, position_ids=time_index.clamp(min=0, max=8191)) # 
        # it is time-dependent embedding, here each visit have aggregated information before this visit
        # Get predictions using separate heads
        if reg_preds_frozen is not None: 
            h = h + self.mapping(reg_preds_frozen)
        y_cls, y_reg = self.head(h)
        return y_cls, y_reg

# EHRVAE1D use EHRBert to embed the labtest1D data, use dataset1Dmode
# EHRVAE1D actually is a VAE model, we train the encoder for EHRGPT2 (discard decoder)
# y_cls, y_reg, mu_z, std_z are the output of EHRVAE1D, pretraining
class EHRVAE1D(nn.Module):
    """1D EHRFormer model for pretraining (Compatible with EHRVAE), always uses VAE"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert self.config['mode'] == 'pretrain', "EHRVAE1D only supports pretrain mode"

        # 根据 use_info_EHR 决定是否包含诊断特征
        self.use_info_EHR = self.config.get('use_info_EHR', False)
        if self.use_info_EHR:
            self.diag_cols = [x for x in self.config['task_info']['category_cols']]
        else:
            self.diag_cols = []

        # 根据 use_info_LAB 决定是否包含labtest特征
        self.use_info_LAB = self.config.get('use_info_LAB', True)
        if not self.use_info_LAB:
            temp_len_float = len(self.config['feat_info']['float_cols'])
            temp_len_cat = len(self.config['feat_info']['category_cols'])
        else:
            temp_len_float = 0
            temp_len_cat = 0

        # update the config and print the result
        self.config['n_category_feats'] = self.config['n_category_feats'] + len(self.diag_cols) - temp_len_cat
        self.config['n_float_feats'] = self.config['n_float_feats'] - temp_len_float
        print(f"EHRVAE1D: EHR: Added {len(self.diag_cols)} diagnosis features to categorical features")
        print(f"EHRVAE1D: LAB: Minus {temp_len_float} float features and {temp_len_cat} category features of labtest")
        print(f"EHRVAE1D: Final n_category_feats: {self.config['n_category_feats']}, Final n_float_feats: {self.config['n_float_feats']}")

        # Base components - 1D模式使用last_hidden_state
        # 因为1D模式把每一个labtest都视为一个单词，pool_emb为false来保存每个labtest的向量，后面也用bert来做提取
        config_1d = {**config, 'pool_emb': False}
        self.ehr_embed = EHRBert(config_1d)
        self.bert_config = self.ehr_embed.bert_config
        self.ehr_mu = BertEncoder(self.bert_config)
        self.ehr_std = BertEncoder(self.bert_config)
        self.decoder = BertEncoder(self.bert_config)
        self.head = MultiTaskHeadsharedhead(config)
        self.type_vocab_size = self.bert_config.type_vocab_size
        # self.pool_embedding = nn.Sequential(
        #     nn.Linear(self.type_vocab_size*config['output_dim'], config['output_dim']), nn.ReLU(),
        #     nn.Linear(config['output_dim'], config['output_dim']), nn.ReLU()
        # )

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, cat_feats, float_feats):
        """Forward pass, returns format compatible with EHRVAE: y_cls, y_reg, mu_z, std_z"""
        # ft_emb: (B, seq_len, hidden_dim) seq_len == type_vocab_size, is 是所有特征类型的数量+1，就是多少列labtest+1（CLS）
        ft_emb = self.ehr_embed(cat_feats, float_feats)
        mu_z = self.ehr_mu(ft_emb)
        std_z = self.ehr_std(ft_emb)
        z = self.reparameterize(mu_z, std_z)
        h = self.decoder(z)
        y_cls, y_reg = self.head(h)
        return y_cls, y_reg, mu_z, std_z

# EHRBert is the most important module for labtest1D, use BERT to embed the labtest1D data
# (b, nc, pad_visit_len) or (b, nc) is the input shape of 2D or 1D data
# anyway, we will reshape the input for time-independent embedding
class EHRBert(nn.Module):
    """统一的EHR BERT嵌入层,支持1D和2D输入,2D的时间维度上没有交互,1D的时间维度上有交互"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("token_type_ids", torch.arange(config['n_category_feats']+config['n_float_feats']+1), persistent=False)
        self.register_buffer("one", torch.ones(1, dtype=torch.bool), persistent=False)
        self.register_buffer("zero", torch.zeros(1, dtype=torch.long), persistent=False)
        
        # 配置选项
        self.pool_emb = config.get('pool_emb', False)
        
        # 为每个分类特征计算token偏移（支持多分类）
        # 默认所有特征是二值的（2个类别），除非在 category_num_classes 中指定
        category_num_classes = config['category_num_classes']
        self.category_num_classes = category_num_classes
        
        # 计算累积偏移：每个特征的起始token位置
        cumsum = [0]
        for num_class in category_num_classes:
            cumsum.append(cumsum[-1] + num_class)
        self.category_token_offsets = torch.tensor(cumsum[:-1], dtype=torch.long)
        self.total_category_tokens = cumsum[-1]
        
        self.register_buffer("category_offsets_buffer", self.category_token_offsets, persistent=False)
        
        # BERT配置
        self.bert_config = BertConfig(
            vocab_size=config['n_float_values']+self.total_category_tokens+2,  # 修正：使用实际的category tokens总数
            hidden_size=config['output_dim'],
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=config['output_dim'] * 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=config['n_category_feats']+config['n_float_feats']+1,  # 0 for CLS
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            position_embedding_type="none",
            use_cache=True,
            classifier_dropout=None)
        
        self.bert = BertModel(self.bert_config)

    def forward(self, cat_feats, float_feats):
        # 检测输入维度
        is_2d = len(cat_feats.shape) == 3  # (b, nc, pad_visit_len) vs (b, nc)
        
        if is_2d:
            # 2D处理：重塑为1D处理
            B = cat_feats.shape[0]
            L = cat_feats.shape[2]
            cat_feats = rearrange(cat_feats, 'b nc l -> (b l) nc')
            # 只有当 float_feats 不为 None 时才重塑
            if float_feats is not None:
                float_feats = rearrange(float_feats, 'b nf l -> (b l) nf')
        else:
            # 1D处理：直接使用
            B = cat_feats.shape[0]
        
        # 统一的数据处理逻辑
        cat_feats_mask = cat_feats == -1
        
        # 将每个位置的值映射为唯一的 input_ids（支持多分类）
        # 每个特征根据其类别数占用不同数量的 token ID
        B_combine, seq_len = cat_feats.shape
        cat_feats = cat_feats.long()  # 确保是整数类型
        
        # 使用累积偏移：每个特征有不同的起始token位置
        position_offset = self.category_offsets_buffer.to(cat_feats.device)
        position_offset = position_offset.unsqueeze(0).expand(B_combine, -1)  # 扩展到 batch 维度
        
        # 映射：cat_feats 中的值（0,1,2,3...）加上该特征的起始偏移
        cat_feats = cat_feats + position_offset
        cat_feats = cat_feats + 2 # cls padded token id
        # 处理 mask：将 -1 位置设置为 1（padded token）
        cat_feats[cat_feats_mask] = 1
        
        # 只有当 float_feats 不为 None 时才处理
        if float_feats is not None:
            float_feats_mask = float_feats == -1
            # float特征的起始位置 = 2 (CLS+PAD) + total_category_tokens
            float_feats = float_feats + 2 + self.total_category_tokens
            float_feats[float_feats_mask] = 1
            input_ids = torch.cat([self.zero.unsqueeze(0).expand(cat_feats.shape[0], -1), cat_feats, float_feats], dim=1)
        else: # 当 float_feats 为 None 时，只使用 cat_feats
            input_ids = torch.cat([self.zero.unsqueeze(0).expand(cat_feats.shape[0], -1), cat_feats], dim=1)
        
        token_type_ids = self.token_type_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
        
        # BERT前向传播
        ft_emb = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        # 简化的输出选择：2D模式总是用pooler_output，1D模式根据pool_emb决定
        # 如果是2d模式，则使用pooler_output，因为这个2D模式是把每个labtest都视为一个单词，所以需要用pooler_output来保存每个labtest的向量
        # 如果是1d模式，则使用last_hidden_state，因为这个1D模式把每个labtest都视为一个单词，所以需要用last_hidden_state来保存每个labtest的向量
        ft_emb = ft_emb.pooler_output if (is_2d or self.pool_emb) else ft_emb.last_hidden_state
        
        # 如果是2D输入，重塑回2D格式
        if is_2d:
            ft_emb = rearrange(ft_emb, '(b l) d -> b l d', b=B)
        
        return ft_emb
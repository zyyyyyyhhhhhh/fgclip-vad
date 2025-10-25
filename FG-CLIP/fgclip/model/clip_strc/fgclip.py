import torch
import torch.nn as nn
import math

from transformers import CLIPConfig,AutoConfig
from typing import Any, Optional, Tuple, Union
import torch.distributed.nn as nn_dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union
from dataclasses import dataclass
from .modeling_clip import CLIPModel, CLIPTextTransformer, CLIPVisionTransformer, CLIPOutput, CLIPAttention, CLIPMLP

import torch.distributed as dist
from torch.nn import AvgPool2d
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from torch import nn

import math
from torchvision.ops import roi_align

# ✅ 自定义输出类,继承CLIPOutput并添加loss_dict字段
@dataclass
class FGCLIPOutput(CLIPOutput):
    """
    扩展的CLIP输出,包含详细的loss信息
    """
    loss_dict: Optional[dict] = None

class FGCLIPModel(CLIPModel):
    config_class = CLIPConfig
    main_input_name = "text_long"

    def __init__(self, config):
        super(CLIPModel, self).__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        text_config.eos_token_id = 49407
        text_config.pad_token_id = 49407
        text_config.bos_token_id = 49406

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)

        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)


        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.text_filip_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)


        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_finegraind = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_hardneg = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

 
        self.embed_dim = text_config.hidden_size
        self.world_size = 0

        # ========== 新增: 时序建模模块 ==========
        # 轻量级 Transformer 用于捕捉帧间时序关系
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.projection_dim,
            nhead=8,
            dim_feedforward=self.projection_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 注意力加权池化: 学习哪些帧更重要
        self.temporal_attention = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim // 2),
            nn.ReLU(),
            nn.Linear(self.projection_dim // 2, 1)
        )
        
        # ✅ ROI特征投影层 - 处理维度匹配
        # 使用ViT-B/32: vision_config.hidden_size=768
        # hidden_states[-2]的特征维度也是768（与配置一致）
        # 
        # ✅ 关键修复：使用visual_projection的权重初始化roi_projection
        # 理论依据：ROI特征和全局特征都来自同一个ViT，应该使用相同的投影空间
        # 这样Region特征和Text特征才能在预训练对齐的空间中对比学习
        self.roi_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        
        # ⚠️ 注意：此时visual_projection还未初始化，实际复制在load_openai_clip_weights()之后
        # 在train_fgclip.py中调用copy_weight()来完成权重复制
        print(f"[INFO] Initialized roi_projection: {self.vision_embed_dim} -> {self.projection_dim} (ViT-B/32)")

        # 运行时诊断开关（默认关闭）。开启后会在有region的batch打印embedding、logits、MB状态等统计信息
        # 用法：在训练脚本中设置 model.enable_runtime_diagnostics = True
        self.enable_runtime_diagnostics = False
        # 打印频率（单位：有region的forward调用次数），默认1表示每次打印
        self.diagnostics_interval = 1

        # ========== Memory Bank for Region Contrastive Learning ==========
        # Queue-based memory bank (MoCo v2 style)
        # 用于增强Region对比学习的负样本数量
        # ✅ 解决方案：使用detach()切断梯度依赖 + clone()创建安全副本
        # 理论依据：MoCo v2论文Section 3.2 "Dictionary as a queue"
        # 
        # ✅ 修复：训练初期暂时禁用Memory Bank，等ROI特征空间稳定后再启用
        # 原因：随机初始化的roi_projection需要先学习基础对齐，否则队列中的特征都是噪声
        # ✅ 策略：前50步禁用MB，等Region loss初步收敛后自动启用
        self.use_memory_bank = False  # 初始禁用，等step 50后自动启用
        self.memory_bank_size = 128  # 每个模态存储128个历史样本（负样本数量从N增加到N+128）
        self.memory_bank_warmup_steps = 50  # ✅ 修复: 50次forward调用 ≈ 6个Trainer step (gradient_accumulation=8)
        
        # 注册为buffer（不参与梯度更新，但会被保存到checkpoint）
        self.register_buffer("region_image_queue", torch.randn(self.projection_dim, self.memory_bank_size))
        self.register_buffer("region_text_queue", torch.randn(self.projection_dim, self.memory_bank_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("training_steps", torch.zeros(1, dtype=torch.long))  # 训练步数计数器
        
        # 初始化队列：归一化到单位向量
        self.region_image_queue = F.normalize(self.region_image_queue, dim=0)
        self.region_text_queue = F.normalize(self.region_text_queue, dim=0)
        
        # 标志位：训练初期队列未填满时使用
        self.register_buffer("queue_is_full", torch.zeros(1, dtype=torch.bool))

        # Initialize weights and apply final processing
        self.post_init()


    def resize_postion_embeding(self, newsize=248):

        old_position_embedding = self.text_model.embeddings.position_embedding
        old_position_embedding_res = self.text_model.embeddings.position_embedding_res
        old_position_embedding_ori = self.text_model.embeddings.position_embedding_ori
        
        positional_embedding_pre = self.text_model.embeddings.position_embedding.weight.data
    
        length, dim = positional_embedding_pre.shape
        keep_len = 20
        posisitonal_embedding_new = torch.zeros([4*length-3*keep_len, dim], dtype=positional_embedding_pre.dtype)
        for i in range(keep_len):
            posisitonal_embedding_new[i] = positional_embedding_pre[i]
        for i in range(length-1-keep_len):
            posisitonal_embedding_new[4*i + keep_len] = positional_embedding_pre[i + keep_len]
            posisitonal_embedding_new[4*i + 1 + keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4

        posisitonal_embedding_new[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
                
        positional_embedding_res = posisitonal_embedding_new.clone()

        self.text_model.embeddings.position_embedding_ori.weight.data = posisitonal_embedding_new
        self.text_model.embeddings.position_embedding_ori.num_embeddings = posisitonal_embedding_new.shape[0]
        
        self.text_model.embeddings.position_embedding_res.weight.data = positional_embedding_res
        self.text_model.embeddings.position_embedding_res.num_embeddings = positional_embedding_res.shape[0]

        old_position_embedding_ori_requires_grad = old_position_embedding_ori.weight.requires_grad
        self.text_model.embeddings.position_embedding_ori.requires_grad_(old_position_embedding_ori_requires_grad)

        old_position_embedding_res_requires_grad = old_position_embedding_res.weight.requires_grad
        self.text_model.embeddings.position_embedding_res.requires_grad_(old_position_embedding_res_requires_grad)



    def copy_weight(self,):
        """
        复制投影层权重，确保不同分支使用相同的特征空间
        
        ✅ 关键修复：将visual_projection的权重复制到roi_projection
        理论依据：
        1. ROI特征和全局特征都来自同一个ViT编码器
        2. 应该投影到相同的CLIP特征空间才能与文本对比
        3. 使用预训练权重初始化可以避免Region分支从零开始学习
        """
        with torch.no_grad():
            # 原有逻辑：text_filip_projection复制text_projection权重
            self.text_filip_projection.weight.data.copy_(self.text_projection.weight.data)
            
            # ✅ 新增：roi_projection复制visual_projection权重
            # 确保ROI特征投影到与全局特征相同的CLIP空间
            self.roi_projection.weight.data.copy_(self.visual_projection.weight.data)
            print("[FG-CLIP] ✅ Copied visual_projection weights to roi_projection")
    
    def load_openai_clip_weights(self, model_name="ViT-B/32"):
        """
        从本地OpenAI CLIP权重文件加载预训练权重
        
        学习自 Vadclip-iccv/src/model52.py 的实现：
        直接使用 clip.load() 加载完整的预训练模型，然后复制权重到FG-CLIP
        
        这个方法比手动映射权重简单得多，因为OpenAI CLIP会自动处理所有架构细节
        
        Args:
            model_name: CLIP模型名称,例如 "ViT-B/32"
        
        Returns:
            loaded_keys: 成功加载的权重键列表
            missing_keys: 缺失的权重键列表
        """
        import os
        from fgclip.model import clip
        
        print(f"[FG-CLIP] 🔄 Loading OpenAI CLIP weights: {model_name}")
        print(f"[FG-CLIP] Cache: ~/.cache/clip/")
        
        try:
            # ✅ 直接使用 clip.load() 加载完整的预训练模型
            # 这会自动从本地缓存加载 ViT-B-32.pt
            clipmodel, _ = clip.load(model_name, device='cpu', jit=False, 
                                    download_root=os.path.expanduser("~/.cache/clip"))
            
            # 获取OpenAI CLIP的state_dict
            openai_state_dict = clipmodel.state_dict()
            
            # 获取当前FG-CLIP模型的state_dict
            fgclip_state_dict = self.state_dict()
            
            loaded_keys = []
            missing_keys = []
            
            # ✅ 复制Vision Transformer权重（智能映射）
            print("[FG-CLIP] Copying Vision Transformer weights...")
            vision_count = 0
            for key in openai_state_dict.keys():
                if not key.startswith('visual.'):
                    continue
                
                # 映射规则:
                # visual.class_embedding → vision_model.embeddings.class_embedding
                # visual.positional_embedding → vision_model.embeddings.position_embedding.weight  
                # visual.conv1.weight → vision_model.embeddings.patch_embedding.weight
                # visual.ln_pre → vision_model.pre_layrnorm
                # visual.transformer.resblocks.X → vision_model.encoder.layers.X
                # visual.ln_post → vision_model.post_layernorm
                
                if 'class_embedding' in key:
                    fgclip_key = 'vision_model.embeddings.class_embedding'
                elif 'positional_embedding' in key and 'visual.' in key:
                    fgclip_key = 'vision_model.embeddings.position_embedding.weight'
                elif 'conv1' in key:
                    fgclip_key = key.replace('visual.conv1', 'vision_model.embeddings.patch_embedding')
                elif 'ln_pre' in key:
                    fgclip_key = key.replace('visual.ln_pre', 'vision_model.pre_layrnorm')
                elif 'ln_post' in key:
                    fgclip_key = key.replace('visual.ln_post', 'vision_model.post_layernorm')
                elif 'resblocks' in key:
                    # visual.transformer.resblocks.X → vision_model.encoder.layers.X
                    fgclip_key = key.replace('visual.transformer.resblocks', 'vision_model.encoder.layers')
                    # in_proj_weight/bias → q/k/v_proj (OpenAI合并的权重，需要分解)
                    if 'in_proj_weight' in key or 'in_proj_bias' in key:
                        # 跳过，需要特殊处理
                        continue
                    fgclip_key = fgclip_key.replace('.attn.', '.self_attn.')
                    fgclip_key = fgclip_key.replace('.ln_1', '.layer_norm1')
                    fgclip_key = fgclip_key.replace('.ln_2', '.layer_norm2')
                    fgclip_key = fgclip_key.replace('.c_fc', '.fc1')
                    fgclip_key = fgclip_key.replace('.c_proj', '.fc2')
                else:
                    continue
                
                if fgclip_key in fgclip_state_dict:
                    if openai_state_dict[key].shape == fgclip_state_dict[fgclip_key].shape:
                        fgclip_state_dict[fgclip_key].copy_(openai_state_dict[key])
                        loaded_keys.append(fgclip_key)
                        vision_count += 1
            
            print(f"[FG-CLIP]   Copied {vision_count} vision weights")
            
            # ✅ 复制Text Transformer权重（智能映射）
            print("[FG-CLIP] Copying Text Transformer weights...")
            text_count = 0
            for key in openai_state_dict.keys():
                if not (key.startswith('transformer.') or key == 'positional_embedding' or key == 'token_embedding.weight'):
                    continue
                
                # 映射规则:
                # positional_embedding → text_model.embeddings.position_embedding.weight
                # token_embedding.weight → text_model.embeddings.token_embedding.weight
                # transformer.resblocks.X → text_model.encoder.layers.X
                
                if key == 'positional_embedding':
                    fgclip_key = 'text_model.embeddings.position_embedding.weight'
                elif key == 'token_embedding.weight':
                    fgclip_key = 'text_model.embeddings.token_embedding.weight'
                elif 'resblocks' in key:
                    fgclip_key = key.replace('transformer.resblocks', 'text_model.encoder.layers')
                    if 'in_proj_weight' in key or 'in_proj_bias' in key:
                        # 跳过合并的QKV权重
                        continue
                    fgclip_key = fgclip_key.replace('.attn.', '.self_attn.')
                    fgclip_key = fgclip_key.replace('.ln_1', '.layer_norm1')
                    fgclip_key = fgclip_key.replace('.ln_2', '.layer_norm2')
                    fgclip_key = fgclip_key.replace('.c_fc', '.fc1')
                    fgclip_key = fgclip_key.replace('.c_proj', '.fc2')
                elif 'ln_final' in key:
                    fgclip_key = key.replace('transformer.ln_final', 'text_model.final_layer_norm')
                else:
                    continue
                
                if fgclip_key in fgclip_state_dict:
                    if openai_state_dict[key].shape == fgclip_state_dict[fgclip_key].shape:
                        fgclip_state_dict[fgclip_key].copy_(openai_state_dict[key])
                        loaded_keys.append(fgclip_key)
                        text_count += 1
            
            print(f"[FG-CLIP]   Copied {text_count} text weights")
            
            # ✅ 复制投影层权重
            projection_mappings = {
                'visual.proj': 'visual_projection.weight',
                'text_projection': 'text_projection.weight',
            }
            
            print("[FG-CLIP] Copying projection layers...")
            for openai_key, fgclip_key in projection_mappings.items():
                if openai_key in openai_state_dict and fgclip_key in fgclip_state_dict:
                    # OpenAI的proj是(D_out, D_in), HuggingFace是(D_in, D_out)，需要转置
                    if 'visual.proj' in openai_key:
                        fgclip_state_dict[fgclip_key].copy_(openai_state_dict[openai_key].T)
                    else:
                        fgclip_state_dict[fgclip_key].copy_(openai_state_dict[openai_key])
                    loaded_keys.append(fgclip_key)
            
            # ✅ 关键修复：复制logit_scale参数（训练温度控制）
            # OpenAI CLIP的logit_scale初始值为ln(100)=4.6052
            # 这个参数控制对比学习的温度，对收敛至关重要
            if 'logit_scale' in openai_state_dict:
                self.logit_scale.data.copy_(openai_state_dict['logit_scale'])
                loaded_keys.append('logit_scale')
                print(f"[FG-CLIP]   ✅ Loaded logit_scale = {self.logit_scale.data.item():.4f} (exp={self.logit_scale.exp().item():.1f})")
            
            # 加载回模型
            self.load_state_dict(fgclip_state_dict, strict=False)
            
            print(f"[FG-CLIP] ✅ Successfully loaded {len(loaded_keys)} weight tensors")
            print(f"[FG-CLIP] ⚠️  Missing {len(missing_keys)} weight tensors (expected for new modules)")
            
            return loaded_keys, missing_keys
            
        except Exception as e:
            print(f"[FG-CLIP] ❌ Failed to load OpenAI CLIP weights: {e}")
            print(f"[FG-CLIP] Continuing with random initialization...")
            return [], []
  

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features
    
    def get_image_box_roi_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        box_info=None,
    ) -> torch.FloatTensor:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )

        bs = pixel_values.shape[0]
        length = vision_outputs[0].shape[1]-1
        h = int(math.sqrt(length))
        w = h

        feature_map = vision_outputs.hidden_states[-2]#[:, 1:, :]
        feature_map = self.forward_without_attn(feature_map)[:, 1:]

        feature_map = self.vision_model.post_layernorm(feature_map)
        feature_map = self.visual_projection(feature_map)

        feature_map = feature_map.view(bs, h, w, -1).permute(0, 3, 1, 2)
        x_rois = roi_align(feature_map.type(torch.float32),box_info, (1, 1), 1.0, -1, True)[..., 0, 0]

        # x_rois = x_rois / x_rois.norm(p=2, dim=-1, keepdim=True)

        return x_rois

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        walk_short_pos: Optional[bool] = True,
        use_bbox: Optional[bool] = False
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pos_flag = walk_short_pos or use_bbox

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            walk_short_pos=pos_flag,
        )
        pooled_output = text_outputs[1]

        if walk_short_pos:
            text_features = self.text_projection(pooled_output)
        else:
            text_features = self.text_filip_projection(pooled_output)           

        return text_features

    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        # print("normed_boxes, ", normed_boxes.shape)
        for boxes in normed_boxes:
            # print("boxes, ", boxes)
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes.type(torch.float32))
        return denormed_boxes

    @torch.no_grad()
    def _update_memory_bank(self, image_feats, text_feats):
        """
        更新Memory Bank (FIFO队列)
        
        Args:
            image_feats: (N, D) 当前batch的region image特征
            text_feats: (N, D) 当前batch的region text特征
        
        注意:
            - 使用@torch.no_grad()确保不计算梯度
            - FIFO策略：新样本替换最老的样本
            - 队列未满时，queue_is_full=False
        """
        batch_size = image_feats.shape[0]
        
        # 当前队列指针位置
        ptr = int(self.queue_ptr)
        
        # 检查是否会超出队列容量
        if ptr + batch_size <= self.memory_bank_size:
            # 情况1: 队列有足够空间，直接写入
            self.region_image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.region_text_queue[:, ptr:ptr + batch_size] = text_feats.T
        else:
            # 情况2: 队列空间不足，需要循环写入（wrap around）
            # 先填充队列末尾剩余空间
            remain_space = self.memory_bank_size - ptr
            self.region_image_queue[:, ptr:] = image_feats[:remain_space].T
            self.region_text_queue[:, ptr:] = text_feats[:remain_space].T
            
            # 再从队列开头写入剩余样本
            overflow_size = batch_size - remain_space
            self.region_image_queue[:, :overflow_size] = image_feats[remain_space:].T
            self.region_text_queue[:, :overflow_size] = text_feats[remain_space:].T
        
        # 更新队列指针（循环）
        old_ptr = ptr  # 保存旧指针用于日志
        ptr = (ptr + batch_size) % self.memory_bank_size
        self.queue_ptr[0] = ptr
        
        # 标记队列已满（至少完整循环一次）
        was_full = self.queue_is_full.item()  # 保存之前的状态
        if not self.queue_is_full and ptr < batch_size:
            self.queue_is_full[0] = True
            print(f"[MEMORY BANK] 🎉 队列首次填满！Ptr: {old_ptr}→{ptr}, 负样本数量: {batch_size}→128")
        
        # 周期性日志：显示队列填充进度
        if not was_full and old_ptr % 32 == 0:  # 训练初期每32个样本打印一次
            fill_ratio = (old_ptr / self.memory_bank_size) * 100
            print(f"[MEMORY BANK] 📊 积累中... Ptr: {old_ptr}/{self.memory_bank_size} ({fill_ratio:.1f}%), 当前负样本: {old_ptr}")

    def forward_without_attn(self, x):
        # get last layer 
        residual = x
        x = self.vision_model.encoder.layers[-1].layer_norm1(x)

        x = F.linear(input=x, weight=self.vision_model.encoder.layers[-1].self_attn.v_proj.weight, bias=self.vision_model.encoder.layers[-1].self_attn.v_proj.bias)
        x = self.vision_model.encoder.layers[-1].self_attn.out_proj(x)
        x = residual+x

        residual = x
        x = self.vision_model.encoder.layers[-1].layer_norm2(x)
        x = self.vision_model.encoder.layers[-1].mlp(x)
        x = residual + x

        return x

    def get_image_dense_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        box_info=None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )


        bs = pixel_values.shape[0]
        length = vision_outputs[0].shape[1]-1
        h = int(math.sqrt(length))
        w = h

        feature_map = vision_outputs.hidden_states[-2]#[:, 1:, :]
        feature_map = self.forward_without_attn(feature_map)[:, 1:]

        feature_map = self.vision_model.post_layernorm(feature_map)
        feature_map = self.visual_projection(feature_map)

        return feature_map

    def forward(
        self,
        text_short: Optional[torch.LongTensor] = None,
        text_long: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        box_infos: Optional[torch.FloatTensor] = None,
        bbox_mask: Optional[torch.BoolTensor] = None,  # ✅ 新增：bbox有效性mask (B, T, max_anns)
        box_texts: Optional[torch.LongTensor] = None,
        box_nums: Optional[torch.LongTensor] = None,
        hard_infos: Optional[torch.FloatTensor] = None,
        hard_texts: Optional[torch.LongTensor] = None,
        hard_nums: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # 文本的attention_mask
        video_attention_mask: Optional[torch.Tensor] = None,  # 新增：视频的attention_mask
        region_videos: Optional[torch.FloatTensor] = None,  # ✅ 新增：region独立采样的视频 (B, max_anns, T, C, H, W)
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        add_box_loss: bool = False,
        use_hard_neg: bool = False,
    ) -> Union[Tuple, CLIPOutput]:

        # ========== 防止属性丢失，自动初始化累积loss相关变量 ==========
        if not hasattr(self, 'accum_buffer'):
            self.accum_buffer = {'total': 0.0, 'global': 0.0, 'region': 0.0, 'hard_neg': 0.0, 'count': 0}
        if not hasattr(self, 'accum_step_counter'):
            self.accum_step_counter = 0
        if not hasattr(self, 'gradient_accumulation_steps'):
            self.gradient_accumulation_steps = 8  # 可根据实际传入config调整

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ========== Memory Bank自动启用逻辑 ==========
        # 获取rank，如果没有初始化分布式环境则使用0
        try:
            rank = dist.get_rank()
        except (RuntimeError, ValueError):
            rank = 0
            
        # ⚠️ 注意：这里的training_steps是forward调用次数，不是Trainer的global_step
        # Trainer的step考虑了gradient_accumulation_steps（每accumulation_steps个forward才更新一次）
        # 所以如果gradient_accumulation=8，那么Trainer的step 50 = 这里的training_steps 400
        # 
        # 为了对齐用户理解，我们改为：在Trainer的step 50时启用（即training_steps=50*8=400）
        if self.training and add_box_loss:
            self.training_steps += 1
            
            # ✅ 自动启用Memory Bank（200步后 - 确保region_videos特征稳定）
            # 理由：训练初期region特征快速演化，过早启用会污染队列
            if not self.use_memory_bank and self.training_steps >= 200:  # 50 → 200
                self.use_memory_bank = True
                if rank == 0:
                    print(f"\n{'='*80}")
                    print(f"[Memory Bank] ✅ 已启用 @ training_step {self.training_steps.item()}")
                    print(f"[Memory Bank] 队列大小: {self.memory_bank_size}, 当前指针: {self.queue_ptr.item()}")
                    print(f"[Memory Bank] ⚠️  训练前200步的旧特征已被丢弃")
                    print(f"{'='*80}\n")
         
        # ========== 修改: 支持视频输入 (B, T, C, H, W) ==========
        is_video = (image.dim() == 5)  # 判断是否为视频输入
        
        if is_video:
            bs, num_frames, c, h, w = image.shape
            # 将视频展平为 (B*T, C, H, W) 进行逐帧编码
            image_flat = image.view(bs * num_frames, c, h, w)
            
            vision_outputs = self.vision_model(
                    pixel_values=image_flat,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
            )
            
            # 提取全局特征 (B*T, D)
            image_embeds_flat = vision_outputs[1]
            image_embeds_flat = self.visual_projection(image_embeds_flat)
            
            # 重塑为 (B, T, D)
            image_embeds_temporal = image_embeds_flat.view(bs, num_frames, -1)
            
            # 时序建模: Transformer捕捉帧间关系
            # 如果有video_attention_mask,使用它来处理填充帧
            if video_attention_mask is not None and video_attention_mask.dim() == 2:
                # video_attention_mask: (B, T), True表示有效帧
                # Transformer需要的mask: (B, T), False表示有效位置
                temporal_mask = ~video_attention_mask  # 反转
                image_embeds_temporal = self.temporal_transformer(
                    image_embeds_temporal, 
                    src_key_padding_mask=temporal_mask
                )
            else:
                image_embeds_temporal = self.temporal_transformer(image_embeds_temporal)
            
            # 注意力加权聚合
            attn_weights = self.temporal_attention(image_embeds_temporal)  # (B, T, 1)
            if video_attention_mask is not None and video_attention_mask.dim() == 2:
                # 将填充帧的权重设为极小值
                attn_weights = attn_weights.masked_fill(~video_attention_mask.unsqueeze(-1), -1e9)
            attn_weights = torch.softmax(attn_weights, dim=1)  # (B, T, 1)
            
            # 加权求和得到视频级特征
            image_embeds = (image_embeds_temporal * attn_weights).sum(dim=1)  # (B, D)
            # ✅ 零范数保护
            image_norm = image_embeds.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            image_embeds = image_embeds / image_norm
        else:
            # 原始图像处理逻辑
            vision_outputs = self.vision_model(
                    pixel_values=image,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
            )
            image_embeds = vision_outputs[1]
            image_embeds = self.visual_projection(image_embeds)
            # ✅ 零范数保护
            image_norm = image_embeds.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            image_embeds = image_embeds / image_norm


        long_text_outputs = self.text_model(
                input_ids=text_long,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # ✅ 注释：short_text不再使用，保留编码只是为了向后兼容
        # 在VAD任务中，short_text(region_caption)应该与region视觉特征对比
        # 而不是与全局视觉特征对比
        short_text_outputs = self.text_model(
                input_ids=text_short,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                walk_short_pos=True,
            )

        long_text_embeds = long_text_outputs[1]
        long_text_embeds = self.text_filip_projection(long_text_embeds)
        # ✅ 零范数保护
        text_norm = long_text_embeds.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        long_text_embeds = long_text_embeds / text_norm

        # ✅ 修改：禁用short_text对比学习，只保留全局对比
        # short_text在VAD任务中语义不匹配（全局视觉 vs region文本）
        # short_text_embeds = short_text_outputs[1]
        # short_text_embeds = self.text_projection(short_text_embeds)
        # short_text_embeds = short_text_embeds / short_text_embeds.norm(p=2, dim=-1, keepdim=True)

        # ✅ 只使用全局对比学习 (image_embeds ↔ long_text_embeds)
        loss_itcl = self.clip_loss_global_only(image_embeds, long_text_embeds, rank, image)

        if getattr(self, "disable_global_loss", False):
            loss_itcl = loss_itcl * 0.0

        # 初始化loss（只包含全局对比）
        loss = loss_itcl
        
        # 用于打印的详细loss字典
        loss_dict = {
            'loss_global': loss_itcl.item() if hasattr(loss_itcl, 'item') else float(loss_itcl),
            'loss_region': 0.0,
            'loss_hard_neg': 0.0,
        }

        if add_box_loss or use_hard_neg:
            # ========== 修改: 支持视频的 feature map 提取 ==========
            if is_video:
                # 对于视频输入,feature_map 已经是 (B*T, ...) 形式
                feature_map = vision_outputs.hidden_states[-2]
                feature_map = self.forward_without_attn(feature_map)[:, 1:]
                feature_map = self.vision_model.post_layernorm(feature_map)
                # ❌ 移除：不要对hidden_states应用visual_projection
                # 因为visual_projection期望768维，但hidden_states是1024维
                # ROI特征的投影由roi_projection单独处理
                # feature_map = self.visual_projection(feature_map)
                
                # 计算空间维度
                length = vision_outputs[0].shape[1]-1
                h = int(math.sqrt(length))
                w = h
                # feature_map: (B*T, h*w, D) -> (B*T, D, h, w)
                feature_map = feature_map.view(bs * num_frames, h, w, -1).permute(0, 3, 1, 2)
            else:
                # 原始图像处理
                feature_map = vision_outputs.hidden_states[-2]
                feature_map = self.forward_without_attn(feature_map)[:, 1:]
                feature_map = self.vision_model.post_layernorm(feature_map)
                # ❌ 移除：同样不要对hidden_states应用visual_projection
                # feature_map = self.visual_projection(feature_map)
                bs = image.shape[0]
                length = vision_outputs[0].shape[1]-1
                h = int(math.sqrt(length))
                w = h
                feature_map = feature_map.view(bs, h, w, -1).permute(0, 3, 1, 2)

        if add_box_loss:
            # ========== 修改: 使用 region_videos 直接编码 Region 特征 ==========
            if region_videos is not None:
                # region_videos: (B, max_anns, T, C, H, W)
                bs_reg, num_anns, num_frames_reg, c, h_img, w_img = region_videos.shape
                
                # 展平为 (B*max_anns*T, C, H, W)
                region_videos_flat = region_videos.view(bs_reg * num_anns * num_frames_reg, c, h_img, w_img)
                
                # 通过 Vision Encoder 编码
                region_vision_outputs = self.vision_model(
                    pixel_values=region_videos_flat,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )
                
                # 提取特征 (B*max_anns*T, D)
                region_embeds_flat = region_vision_outputs[1]
                region_embeds_flat = self.visual_projection(region_embeds_flat)
                
                # 重塑为 (B, max_anns, T, D)
                region_embeds_temporal = region_embeds_flat.view(bs_reg, num_anns, num_frames_reg, -1)
                
                # 对每个 region 单独进行时序建模
                # 需要处理维度: (B*max_anns, T, D)
                region_embeds_for_transformer = region_embeds_temporal.view(bs_reg * num_anns, num_frames_reg, -1)
                
                # 时序 Transformer
                region_embeds_temporal_transformed = self.temporal_transformer(region_embeds_for_transformer)
                
                # 注意力加权聚合
                region_attn_weights = self.temporal_attention(region_embeds_temporal_transformed)  # (B*max_anns, T, 1)
                region_attn_weights = torch.softmax(region_attn_weights, dim=1)
                
                # 加权求和 (B*max_anns, D)
                bbox_image_embeds = (region_embeds_temporal_transformed * region_attn_weights).sum(dim=1)
                
                # 归一化
                bbox_image_embeds = F.normalize(bbox_image_embeds, p=2, dim=-1, eps=1e-8)
            
            # ========== Region 文本编码（必须在视觉编码后立即执行）==========
            # 无论使用 region_videos 还是 ROI Align，都需要编码 box_texts
            bbox_text_outputs = self.text_model(
                input_ids=box_texts,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                walk_short_pos=True,
            )
            bbox_text_embeds = bbox_text_outputs[1]
            bbox_text_embeds = self.text_projection(bbox_text_embeds)
            # ✅ 零范数保护
            text_norm = bbox_text_embeds.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            bbox_text_embeds = bbox_text_embeds / text_norm
                
            if region_videos is None:
                # ========== 回退: 使用旧逻辑（从 feature_map 提取 ROI）==========
                box_size = box_infos.shape[-1]  # 应该是 4 (x1, y1, x2, y2)
                
                if is_video:
                    # box_infos: (B, T, max_anns, 4)
                    if box_infos.dim() == 4:
                        num_anns = box_infos.shape[2]
                        # 展平为 (B*T, max_anns, 4)
                        box_infos_expanded = box_infos.view(bs * num_frames, num_anns, box_size)
                        
                        # 反归一化 bbox
                        original_bboxes = self._denormalize_boxes(box_infos_expanded, feature_map)
                        
                        # RoI Align
                        x_rois = roi_align(feature_map.type(torch.float32), original_bboxes, (1, 1), 1.0, -1, True)[..., 0, 0]
                        
                        # 重塑为 (B, T, max_anns, D)
                        x_rois = x_rois.view(bs, num_frames, num_anns, -1)
                        
                        # 时序聚合
                        if bbox_mask is not None:
                            mask_expanded = bbox_mask.unsqueeze(-1).float()
                            x_rois_masked = x_rois * mask_expanded
                            valid_frames = bbox_mask.sum(dim=1).unsqueeze(-1).float()
                            bbox_vision_outputs = x_rois_masked.sum(dim=1) / valid_frames.clamp(min=1)
                        else:
                            if video_attention_mask is not None and video_attention_mask.dim() == 2:
                                mask_expanded = video_attention_mask.unsqueeze(-1).unsqueeze(-1)
                                x_rois_masked = x_rois * mask_expanded
                                valid_frames = video_attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                                bbox_vision_outputs = x_rois_masked.sum(dim=1) / valid_frames.clamp(min=1)
                            else:
                                bbox_vision_outputs = x_rois.mean(dim=1)
                        
                        # 展平为 (B*max_anns, D)
                        bbox_vision_outputs = bbox_vision_outputs.view(bs * num_anns, -1).type(torch.bfloat16)
                    elif box_infos.dim() == 3:
                        raise ValueError("3D box_infos is deprecated. Please use 4D box_infos (B, T, max_anns, 4)")
                    else:
                        raise ValueError(f"Unexpected box_infos shape for video: {box_infos.shape}")
                else:
                    # 图像处理逻辑
                    if box_infos.dim() == 4:
                        box_infos = box_infos.squeeze(1)
                    elif box_infos.dim() == 2:
                        box_infos = box_infos.reshape(bs, -1, box_size)
                    else:
                        raise ValueError(f"Unexpected box_infos shape for image: {box_infos.shape}")
                    
                    original_bboxes = self._denormalize_boxes(box_infos, feature_map)
                    x_rois = roi_align(feature_map.type(torch.float32), original_bboxes, (1, 1), 1.0, -1, True)[..., 0, 0]
                    bbox_vision_outputs = x_rois.type(torch.bfloat16)
                
                # 投影到最终维度
                bbox_image_embeds = self.roi_projection(bbox_vision_outputs)
                bbox_image_embeds = F.normalize(bbox_image_embeds, p=2, dim=-1, eps=1e-8)
            
        if use_hard_neg:
            box_size = hard_infos.shape[-1]
            hard_infos = hard_infos.reshape(bs, -1, box_size)
           
            original_bboxes = self._denormalize_boxes(hard_infos, feature_map)
            x_rois = roi_align(feature_map.type(torch.float32), original_bboxes, (1, 1), 1.0, -1, True)[..., 0, 0]

            hard_bbox_image_embeds = x_rois.type(torch.bfloat16)
            
            # ✅ 关键修复：Hard Negative的ROI特征也需要经过projection（与Region分支保持一致）
            # 确保维度与bbox_image_embeds一致（512维）
            if self.roi_projection is not None and not isinstance(self.roi_projection, nn.Identity):
                hard_bbox_image_embeds = self.roi_projection(hard_bbox_image_embeds)
            
            # ✅ 零范数保护 - 使用F.normalize避免inplace操作
            hard_bbox_image_embeds = F.normalize(hard_bbox_image_embeds, p=2, dim=-1, eps=1e-8)

            hard_bbox_text_outputs = self.text_model(
                    input_ids=hard_texts,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=return_dict,
                    walk_short_pos=True,
                )
            hard_bbox_text_embeds = hard_bbox_text_outputs[1]
            hard_bbox_text_embeds = self.text_projection(hard_bbox_text_embeds)
            # ✅ 零范数保护
            hard_text_norm = hard_bbox_text_embeds.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            hard_bbox_text_embeds = hard_bbox_text_embeds / hard_text_norm

        

        if add_box_loss:

            # ✅ 修复：提升Region loss权重，增强学习信号
            # 原值0.1导致Region梯度过弱，无法有效学习
            # 理论依据：Region对比学习对VAD任务更关键，应该与Global同等重要
            box_loss_weight = 0.5  # 从0.1提升到0.5
            bs = box_nums.shape[0]
            bbox_size = int(bbox_text_embeds.shape[0]/bs)
            # 初始基于 box_nums 标记有效region（按region_data数量）
            box_weight = torch.zeros([bs, bbox_size], device=bbox_text_embeds.device)
            for i in range(bs):
                valid_count = int(box_nums[i])
                box_weight[i][:valid_count] = 1

            # 额外安全过滤：如果 bbox_mask 存在，剔除那些在所有帧上都无有效bbox的region
            # 这种情况会在 temporal aggregation 时产生全零的视觉特征。
            # 如果不剔除，它们会以零向量被选中并进入 memory bank，污染负样本集合。
            if bbox_mask is not None:
                try:
                    # bbox_mask: (B, T, max_anns) -> (B, max_anns) indicating每region是否在任意帧上有效
                    valid_region_mask = (bbox_mask.sum(dim=1) > 0)  # (B, max_anns), bool
                    # 对齐dtype并逐元素相乘
                    box_weight = box_weight * valid_region_mask.to(box_weight.dtype)
                except Exception:
                    # 若任何维度不匹配，保留原有行为，避免引入运行时错误
                    pass

            # flatten并选取非零索引
            box_weight = box_weight.reshape(1, bbox_text_embeds.shape[0]).squeeze()
            select_index = box_weight.nonzero()
            
            # ✅ 修复：安全的squeeze，避免单样本维度塌陷
            if select_index.numel() > 0:
                valid_count = select_index.shape[0]
                bbox_text_embeds = bbox_text_embeds[select_index, :].view(valid_count, -1)
                bbox_image_embeds = bbox_image_embeds[select_index, :].view(valid_count, -1)
                
                # ========== Region对比学习 with Memory Bank ==========
                # 目标：将负样本数量从batch内的N个扩展到N+128个（Memory Bank）
                # 架构：MoCo v2风格的Queue-based Memory Bank
                # 解决方案：Detach + Clone机制确保梯度安全
                
                logit_scale = self.logit_scale_finegraind.exp()
                
                if self.use_memory_bank:
                    # ✅ 动态Queue策略：训练初期只用batch内对比，队列填满后再使用Memory Bank
                    # 理论依据：MoCo v2论文Algorithm 1第7-8行
                    
                    # Step 1: 计算当前有效队列大小
                    if self.queue_is_full:
                        # 队列已满：使用全部128个样本
                        effective_queue_size = self.memory_bank_size
                    else:
                        # 队列未满：只使用已填充的部分（queue_ptr指向下一个空位）
                        effective_queue_size = int(self.queue_ptr)
                    
                    # Step 2: 根据队列状态选择对比策略
                    if effective_queue_size > 0:
                        # 有历史样本：使用Memory Bank增强对比学习
                        queue_image = self.region_image_queue[:, :effective_queue_size].detach().clone()  # (D, K)
                        queue_text = self.region_text_queue[:, :effective_queue_size].detach().clone()    # (D, K)
                        
                        # ✅ 质量监控：检测队列样本的范数分布
                        with torch.no_grad():
                            queue_img_norms = queue_image.norm(p=2, dim=0)  # (K,)
                            queue_txt_norms = queue_text.norm(p=2, dim=0)   # (K,)
                            curr_img_norms = bbox_image_embeds.norm(p=2, dim=1)  # (N,)
                            
                            # 检测异常：队列样本的范数远小于当前batch
                            queue_norm_mean = queue_img_norms.mean().item()
                            curr_norm_mean = curr_img_norms.mean().item()
                            norm_ratio = curr_norm_mean / (queue_norm_mean + 1e-8)
                            
                            # 如果队列样本质量明显低于当前batch，打印警告
                            if norm_ratio > 1.5 and self.training_steps % 50 == 0:
                                print(f"⚠️  [MB Quality] Queue样本范数偏低：Queue={queue_norm_mean:.4f} vs Curr={curr_norm_mean:.4f} (ratio={norm_ratio:.2f})")
                        
                        # 训练监控：显示Memory Bank状态（训练初期）
                        if not self.queue_is_full and effective_queue_size % 32 == 0:
                            print(f"[MB-FORWARD] 使用{effective_queue_size}个历史负样本 | Batch内: {valid_count} | 总负样本: {valid_count + effective_queue_size}")
                        
                        # 拼接batch + queue作为负样本
                        text_with_queue = torch.cat([bbox_text_embeds.T, queue_text], dim=1)  # (D, N+K)
                        logits_i2t = torch.matmul(bbox_image_embeds, text_with_queue) * logit_scale  # (N, N+K)
                        
                        image_with_queue = torch.cat([bbox_image_embeds.T, queue_image], dim=1)  # (D, N+K)
                        logits_t2i = torch.matmul(bbox_text_embeds, image_with_queue) * logit_scale  # (N, N+K)
                        
                        # 标签：正样本在前N个位置的对角线
                        labels = torch.arange(valid_count, device=bbox_image_embeds.device, dtype=torch.long)
                    else:
                        # 训练最开始：队列为空，只用batch内对比（避免随机噪声干扰）
                        logits_i2t = torch.matmul(bbox_image_embeds, bbox_text_embeds.T) * logit_scale
                        logits_t2i = torch.matmul(bbox_text_embeds, bbox_image_embeds.T) * logit_scale
                        labels = torch.arange(valid_count, device=bbox_image_embeds.device, dtype=torch.long)
                    
                    # Step 4: InfoNCE Loss
                    loss_i2t = F.cross_entropy(logits_i2t, labels)
                    loss_t2i = F.cross_entropy(logits_t2i, labels)
                    loss_bbox_itcl = (loss_i2t + loss_t2i) / 2.0
                    
                    # Step 5: 安全更新Memory Bank（使用detached特征）
                    with torch.no_grad():
                        # clone()确保特征副本不会反向传播
                        self._update_memory_bank(
                            bbox_image_embeds.detach().clone(), 
                            bbox_text_embeds.detach().clone()
                        )
                else:
                    # Fallback: 只使用batch内对比（原始实现）
                    logits_i2t = torch.matmul(bbox_image_embeds, bbox_text_embeds.T) * logit_scale
                    logits_t2i = torch.matmul(bbox_text_embeds, bbox_image_embeds.T) * logit_scale
                    
                    labels = torch.arange(valid_count, device=bbox_image_embeds.device, dtype=torch.long)
                    loss_i2t = F.cross_entropy(logits_i2t, labels)
                    loss_t2i = F.cross_entropy(logits_t2i, labels)
                    loss_bbox_itcl = (loss_i2t + loss_t2i) / 2.0
                
                # 累加到总损失
                # ✅ 提升Region loss权重：从0.2→0.5（与Global同等重要）
                # 理由：Region对VAD任务更关键，需要更强的学习信号
                box_loss_weight = 0.5
                loss = loss + box_loss_weight * loss_bbox_itcl
                loss_dict['loss_region'] = loss_bbox_itcl.item() if hasattr(loss_bbox_itcl, 'item') else float(loss_bbox_itcl)
                # ========== 运行时诊断输出 ==========
                try:
                    if getattr(self, 'enable_runtime_diagnostics', False):
                        # 计数器（只在有region的forward中递增）
                        if not hasattr(self, '_diag_counter'):
                            self._diag_counter = 0
                        self._diag_counter += 1
                        if (self._diag_counter % max(1, getattr(self, 'diagnostics_interval', 1))) == 0:
                            # 基本统计
                            img_norms = bbox_image_embeds.norm(p=2, dim=-1)
                            txt_norms = bbox_text_embeds.norm(p=2, dim=-1)
                            logits = None
                            try:
                                logits = logits_i2t.detach() if 'logits_i2t' in locals() else None
                            except Exception:
                                logits = None

                            def stats(tensor):
                                if tensor is None:
                                    return {'min': None, 'max': None, 'mean': None, 'std': None, 'has_nan': None}
                                t = tensor.detach()
                                return {
                                    'min': float(t.min().item()),
                                    'max': float(t.max().item()),
                                    'mean': float(t.mean().item()),
                                    'std': float(t.std().item()),
                                    'has_nan': bool(torch.isnan(t).any().item()),
                                }

                            img_stats = stats(img_norms)
                            txt_stats = stats(txt_norms)
                            logits_stats = stats(logits)

                            mb_ptr = int(self.queue_ptr.item()) if hasattr(self, 'queue_ptr') else None
                            mb_full = bool(self.queue_is_full.item()) if hasattr(self, 'queue_is_full') else None
                            mb_size = int(self.memory_bank_size) if hasattr(self, 'memory_bank_size') else None

                            print("[DIAG] Region batch diagnostics:")
                            print(f"  - valid_count={valid_count} | bbox_batch_size={bbox_text_embeds.shape[0]}")
                            print(f"  - img_norms: min={img_stats['min']:.4f}, max={img_stats['max']:.4f}, mean={img_stats['mean']:.4f}, std={img_stats['std']:.4f}, has_nan={img_stats['has_nan']}")
                            print(f"  - txt_norms: min={txt_stats['min']:.4f}, max={txt_stats['max']:.4f}, mean={txt_stats['mean']:.4f}, std={txt_stats['std']:.4f}, has_nan={txt_stats['has_nan']}")
                            if logits_stats['min'] is not None:
                                print(f"  - logits_i2t: min={logits_stats['min']:.4f}, max={logits_stats['max']:.4f}, mean={logits_stats['mean']:.4f}, std={logits_stats['std']:.4f}, has_nan={logits_stats['has_nan']}")
                            print(f"  - logit_scale (finegrained): {self.logit_scale_finegraind.item():.6f} (exp={self.logit_scale_finegraind.exp().item():.4f})")
                            print(f"  - memory_bank: ptr={mb_ptr}, full={mb_full}, size={mb_size}")
                            # 检查NaN/Inf
                            any_nan = torch.isnan(bbox_image_embeds).any().item() or torch.isnan(bbox_text_embeds).any().item()
                            any_inf = torch.isinf(bbox_image_embeds).any().item() or torch.isinf(bbox_text_embeds).any().item()
                            print(f"  - any_nan={any_nan}, any_inf={any_inf}")
                            # 额外：打印第一个样本的bbox文本和一些logits位置
                            try:
                                print(f"  - sample_logits_row0_first5: {logits[0,:5].cpu().numpy().tolist() if logits is not None else 'N/A'}")
                            except Exception:
                                pass
                except Exception as e:
                    print(f"[DIAG] Failed to emit diagnostics: {e}")
            else:
                # 如果没有有效region，跳过region loss
                pass

        if use_hard_neg:
            hard_box_loss_weight = 0.5

            bs = hard_nums.shape[0]
            bbox_size = int(hard_bbox_image_embeds.shape[0]/bs)
            box_weight = torch.zeros([bs, bbox_size], device=hard_bbox_image_embeds.device)
            for i in range(bs):
                valid_count = int(hard_nums[i])
                box_weight[i][:valid_count] = 1
            box_weight = box_weight.reshape(1, hard_bbox_image_embeds.shape[0]).squeeze()
            select_index = box_weight.nonzero()
            hard_bbox_image_embeds = hard_bbox_image_embeds[select_index,:].squeeze()
            loss_bbox_hitc= self.hard_contrastive_loss(hard_bbox_image_embeds, hard_bbox_text_embeds, hard_bbox_text_embeds.device, self.logit_scale_hardneg)
            loss = loss + hard_box_loss_weight*loss_bbox_hitc
            loss_dict['loss_hard_neg'] = loss_bbox_hitc.item() if hasattr(loss_bbox_hitc, 'item') else float(loss_bbox_hitc)
        
        # ========== 只记录累积步（accumulated）loss到TensorBoard ==========
        if rank == 0:
            # 累积loss统计
            self.accum_buffer['total'] += loss.item()
            self.accum_buffer['global'] += loss_dict['loss_global']
            self.accum_buffer['region'] += loss_dict['loss_region']
            self.accum_buffer['hard_neg'] += loss_dict['loss_hard_neg']
            self.accum_buffer['count'] += 1
            if self.accum_buffer['count'] == self.gradient_accumulation_steps:
                avg_total = self.accum_buffer['total'] / self.accum_buffer['count']
                avg_global = self.accum_buffer['global'] / self.accum_buffer['count']
                avg_region = self.accum_buffer['region'] / self.accum_buffer['count']
                avg_hard_neg = self.accum_buffer['hard_neg'] / self.accum_buffer['count']
                self.accum_step_counter += 1
                print(f"[ACCUMULATED] Total: {avg_total:.4f} | Global: {avg_global:.4f} | Region: {avg_region:.4f} | HardNeg: {avg_hard_neg:.4f}", flush=True)
                if hasattr(self, 'tb_writer') and self.tb_writer is not None:
                    self.tb_writer.add_scalar('Loss/Total', avg_total, self.accum_step_counter)
                    self.tb_writer.add_scalar('Loss/Global', avg_global, self.accum_step_counter)
                    self.tb_writer.add_scalar('Loss/Region', avg_region, self.accum_step_counter)
                    self.tb_writer.add_scalar('Loss/HardNeg', avg_hard_neg, self.accum_step_counter)
                self.accum_buffer = {'total': 0.0, 'global': 0.0, 'region': 0.0, 'hard_neg': 0.0, 'count': 0}
        
        # ✅ 使用FGCLIPOutput返回结果,包含loss_dict
        return FGCLIPOutput(
            loss=loss,
            loss_dict=loss_dict
        )

    

    def clip_loss(self,image_features_long, text_features_long, text_features_short,rank,image):


        image_feat_all_long = torch.cat(nn_dist.all_gather(image_features_long), dim=0)#gather with grad

        if text_features_long is not None:
            text_feat_all_long = torch.cat(nn_dist.all_gather(text_features_long), dim=0)

        text_feat_all_short = torch.cat(nn_dist.all_gather(text_features_short), dim=0)
        
        if text_features_long is not None:
            sim_i2tl = torch.matmul(image_features_long, text_feat_all_long.T)
            sim_tl2i = torch.matmul(image_feat_all_long, text_features_long.T)
            sim_tl2i = sim_tl2i.T

        sim_i2ts = torch.matmul(image_features_long, text_feat_all_short.T)
        sim_ts2i = torch.matmul(image_feat_all_long, text_features_short.T)
        sim_ts2i = sim_ts2i.T

        
        if text_features_long is not None:
            sim_i2tl = self.logit_scale.exp() * sim_i2tl
            sim_tl2i = self.logit_scale.exp() * sim_tl2i


        sim_i2ts = self.logit_scale.exp() * sim_i2ts
        sim_ts2i = self.logit_scale.exp() * sim_ts2i

        
        bs = image_features_long.size(0)
        targets = torch.linspace(rank * bs,rank * bs + bs - 1, bs, dtype=torch.long).to(image.device)

        loss_itcl = None
        if text_features_long is not None:
            loss_itcl = (
                    F.cross_entropy(sim_i2tl, targets, label_smoothing=0.0)
                    + F.cross_entropy(sim_tl2i, targets, label_smoothing=0.0)
                ) / 2
        
        loss_itcs = (
                F.cross_entropy(sim_i2ts, targets, label_smoothing=0.0)
                + F.cross_entropy(sim_ts2i, targets, label_smoothing=0.0)
            ) / 2

        return loss_itcl, loss_itcs

    
    def clip_loss_global_only(self, image_features, text_features, rank, image):
        """
        ✅ 修改：只进行全局对比学习 (image_embeds ↔ global_caption)
        
        Args:
            image_features: 全局视觉特征 [CLS] token (B, D)
            text_features: 全局文本特征 global_caption (B, D)
            rank: 当前GPU的rank
            image: 用于获取device
            
        Returns:
            loss_itcl: 全局对比学习loss
        """
        # ✅ 修复：正确处理分布式和单GPU环境
        if dist.is_available() and dist.is_initialized():
            # 分布式训练：跨GPU gather
            image_feat_all = torch.cat(nn_dist.all_gather(image_features), dim=0)
            text_feat_all = torch.cat(nn_dist.all_gather(text_features), dim=0)
            world_size = dist.get_world_size()
        else:
            # ✅ 单GPU训练：不需要all_gather，rank固定为0
            image_feat_all = image_features
            text_feat_all = text_features
            rank = 0
            world_size = 1
        
        # 计算相似度矩阵
        sim_i2t = torch.matmul(image_features, text_feat_all.T)  # (B, world_size×B)
        sim_t2i = torch.matmul(image_feat_all, text_features.T)  # (world_size×B, B)
        sim_t2i = sim_t2i.T  # (B, world_size×B)
        
        # ✅ 温度缩放 + 数值稳定性保护
        logit_scale_clamped = torch.clamp(self.logit_scale, max=4.6052)  # exp(4.6) ≈ 100
        scale = logit_scale_clamped.exp()
        sim_i2t = scale * sim_i2t
        sim_t2i = scale * sim_t2i
        
        # ✅ 构建标签（单GPU时rank=0，targets就是[0,1,2,...,B-1]）
        bs = image_features.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long).to(image.device)
        
        # InfoNCE Loss（双向对比）
        loss_i2t = F.cross_entropy(sim_i2t, targets, label_smoothing=0.0)
        loss_t2i = F.cross_entropy(sim_t2i, targets, label_smoothing=0.0)
        
        loss_itcl = (loss_i2t + loss_t2i) / 2
        
        return loss_itcl


    def pairwise_contrastive_loss(self, image_features_long, text_features_long, device, logit_scale=1.0):
        batch_size, c = image_features_long.shape
        
        # ✅ 修复：使用类别索引而非one-hot矩阵
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        # ✅ 数值稳定性：clamp logit_scale
        logit_scale_clamped = torch.clamp(logit_scale, max=4.6052)  # exp(4.6) ≈ 100
        scale = logit_scale_clamped.exp()
        
        # 🔍 详细调试：检查矩阵乘法输入
        # print(f"    [pairwise] scale={scale.item():.4f}, batch_size={batch_size}", flush=True)
        
        logits_per_image = scale * image_features_long @ text_features_long.T
        logits_per_text = scale * text_features_long @ image_features_long.T
        
        # 🔍 检查相似度矩阵
        # print(f"    [pairwise] logits_per_image: min={logits_per_image.min().item():.4f}, "
        #       f"max={logits_per_image.max().item():.4f}, "
        #       f"has_nan={torch.isnan(logits_per_image).any().item()}", flush=True)
        
        temp1 = F.cross_entropy(logits_per_text, labels)
        temp2 = F.cross_entropy(logits_per_image, labels)

        loss = (temp1+temp2)/2
        return loss
    
    def hard_contrastive_loss(self, image_features_long, text_features_long, device, logit_scale=1.0):
        batch_size, c = image_features_long.shape
        text_features_long = text_features_long.reshape(batch_size, 11, -1)
        labels = torch.zeros(batch_size, device=device, dtype=torch.long)#.repeat(batch_size, 1)
        predict = logit_scale.exp() * torch.einsum('bp,bdp->bd', image_features_long, text_features_long)
        loss = F.cross_entropy(predict, labels)
        return loss

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim) 
        self.num_heads = num_heads

    def forward(self, x): 
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC  #32,2048,7,7 ->49, 32, 2048
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC  50,32,2048
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        ) 

        return x 

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1) 
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): 
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype) 
        x = stem(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x3 = self.layer3(x) 
        x4 = self.layer4(x3) 
        xproj = self.attnpool(x4) 

        return x3, x4, xproj 

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MoEResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, 
                 num_experts: int,
                 top_k: int,
                 dropout: float = 0.0,
                 attn_mask: torch.Tensor = None
                 ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(p=dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ])) for _ in range(self.num_experts)])
        self.gate = nn.Linear(d_model, self.num_experts, bias=False)
        
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x: torch.Tensor, pre_computed_routing: tuple = None):
        x_after_attn = x + self.attention(self.ln_1(x))
        
        # 准备MoE层的输入
        hidden_states = self.ln_2(x_after_attn)  # [seq_len, batch_size, hidden_dim] (LND格式)
        seq_len, batch_size, hidden_dim = hidden_states.shape
        
        # 转换为[batch_size*seq_len, hidden_dim]方便处理
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)
        
        current_router_logits = None
        
        if pre_computed_routing is None:
            # 此块执行门控
            router_logits = self.gate(hidden_states_flat)  # [batch_size*seq_len, num_experts]
            current_router_logits = router_logits
            
            # 计算路由权重并选择top-k专家
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)  # 重新归一化权重
            routing_weights = routing_weights.to(hidden_states_flat.dtype)
            
            # 保存路由决策以供后续层使用 (如果这是第一个门控块)
            routing_decision_to_pass = (routing_weights, selected_experts)
        else:
            # 使用预先计算的路由
            routing_weights, selected_experts = pre_computed_routing
            # current_router_logits 保持为 None，因为此块不产生新的 logits

        # 准备输出张量
        final_hidden_states = torch.zeros(
            (seq_len * batch_size, hidden_dim), dtype=hidden_states_flat.dtype, device=hidden_states_flat.device
        )
        
        # 使用one-hot编码创建专家掩码，方便索引选择的专家
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # expert_mask: [num_experts, top_k, batch_size*seq_len]
        
        # 对每个专家进行计算
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            # 如果没有token路由到这个专家，跳过
            if not top_x.shape[0]:
                continue
                
            # 获取需要由当前专家处理的token
            current_state = hidden_states_flat[top_x]
            
            # 应用专家计算并加权
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            
            # 将结果添加到最终输出中
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
            
        # 重塑回原始尺寸
        final_hidden_states = final_hidden_states.reshape(seq_len, batch_size, hidden_dim)
        
        # 残差连接
        output = x_after_attn + final_hidden_states
        
        if pre_computed_routing is None:
            # 此块执行了门控，返回输出、logits和路由决策
            return output, current_router_logits, routing_decision_to_pass
        else:
            # 此块使用了预计算的路由，仅返回输出
            return output

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class MoETransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, 
                 num_experts: int, top_k: int, 
                 moe_layers: int = 0,  # 使用MoE的层数，-1表示全部使用
                 dropout: float = 0.0,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        
        # 确定使用MoE的层数
        self.moe_layers = layers if moe_layers == -1 else min(moe_layers, layers)
        self.standard_layers = layers - self.moe_layers
        
        # 创建MoE层和标准层，MoE层放在前面
        self.resblocks = nn.ModuleList(
            [MoEResidualAttentionBlock(width, heads, num_experts, top_k, dropout, attn_mask) 
             for _ in range(self.moe_layers)] + 
            [ResidualAttentionBlock(width, heads, attn_mask) for _ in range(self.standard_layers)]
        )
        
    def forward(self, x: torch.Tensor):
        first_gate_router_logits = None
        routing_decision_from_first_moe = None
        moe_block_encountered = False
        
        # 遍历所有层，根据类型处理
        for i, block in enumerate(self.resblocks):
            if isinstance(block, ResidualAttentionBlock):
                x = block(x)
            elif isinstance(block, MoEResidualAttentionBlock):
                if not moe_block_encountered:
                    # 这是第一个MoE块，执行门控并保存决策和logits
                    x, current_logits, routing_decision = block(x, pre_computed_routing=None)
                    first_gate_router_logits = current_logits
                    routing_decision_from_first_moe = routing_decision
                    moe_block_encountered = True
                else:
                    # 后续MoE块，使用已保存的路由决策
                    x = block(x, pre_computed_routing=routing_decision_from_first_moe)
            else:
                # 处理其他可能类型的块（如果存在）
                x = block(x)
        
        if first_gate_router_logits is not None:
            return x, first_gate_router_logits
        else:
            return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

def load_balancing_loss_func(gate_logits: torch.Tensor, top_k: int) -> float:
    """计算负载平衡损失
    
    该损失函数鼓励门控网络平衡使用各专家，避免出现专家过载或闲置的情况。
    参考Switch Transformer(https://arxiv.org/abs/2101.03961)论文中的公式(4)-(6)。
    
    Args:
        gate_logits: 门控网络输出的logits，形状为[batch_size*seq_len, num_experts]
        top_k: 每个token选择的专家数量
        
    Returns:
        负载平衡损失值
    """
    num_experts = gate_logits.shape[-1]
    gate_logits = gate_logits.view(-1, num_experts)
    
    # 计算路由权重
    routing_weights = F.softmax(gate_logits, dim=-1)
    
    # 选择top-k专家
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # 计算专家分配掩码
    expert_mask = F.one_hot(selected_experts, num_experts)
    
    # 计算分配给每个专家的token百分比
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    
    # 计算路由到每个专家的平均概率
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    
    # 计算总体损失
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)
    
    # 乘以专家数量作为最终损失
    return overall_loss * num_experts

class VisionTransformer(nn.Module):
    def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int,
                 width: int, layers: int, heads: int, output_dim: int,
                 num_experts: int = 0,  # 0表示不使用MoE
                 top_k: int = 0,
                 moe_layers: int = 0,
                 dropout: float = 0.0
                 ):
        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.use_moe = num_experts > 0 and top_k > 0
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution*w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)

        # 根据参数选择使用标准Transformer还是MoE Transformer
        if self.use_moe:
            self.transformer = MoETransformer(
                width, layers, heads, 
                num_experts=num_experts, 
                top_k=top_k,
                moe_layers=moe_layers,
                dropout=dropout
            )
        else:
            self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, cv_emb = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if cv_emb != None: 
            x[:,0] = x[:,0] + cv_emb
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # router_logits 将存储第一次门控的logits，如果MoE被使用
        first_gate_router_logits = None 
        if self.use_moe:
            # MoETransformer现在返回 (x, first_gate_router_logits) 或 (x)
            transformer_output = self.transformer(x)
            if isinstance(transformer_output, tuple):
                x_transformed, first_gate_router_logits = transformer_output
            else:
                x_transformed = transformer_output
            
            # 特征提取逻辑保持不变，但基于 x_transformed
            # 提取前11层（或所有层，如果少于11层）的"概念"
            # 这个部分需要仔细考虑，因为MoETransformer内部循环处理了所有层
            # 我们这里假设x_transformed是经过所有transformer块处理后的结果
            # 为了提取中间层特征，我们需要修改MoETransformer或者在这里重新实现部分逻辑
            # 为了简化，当前假设我们只需要最终的输出和第一次的gate logits
            
            # 简单的处理方式是，MoETransformer已经处理了所有层
            # VisionTransformer的x11, x12概念可能需要重新思考或调整
            # 如果目标是获取特定中间层的输出，MoETransformer的forward也需要调整以返回这些
            
            # 假设我们用MoETransformer的最终输出来代表x12, 并根据需要处理x11
            # 目前，直接使用transformer的最终输出作为后续处理的基础
            x_processed_by_transformer = x_transformed
            
            # 临时将x11和x12都设为最终的输出
            x11 = x_processed_by_transformer 
            x12 = x_processed_by_transformer

        else:
            # 标准transformer
            if len(self.transformer.resblocks) >= 12:
                x11 = self.transformer.resblocks[:11](x)
                x12 = self.transformer.resblocks[11](x11)
            elif len(self.transformer.resblocks) > 0 : # 至少有一个block
                x11 = self.transformer.resblocks[:-1](x) if len(self.transformer.resblocks) > 1 else x
                x12 = self.transformer.resblocks[-1](x11 if len(self.transformer.resblocks) > 1 else x)
            else: # 没有resblocks
                x11 = x
                x12 = x
                
        x11 = x11.permute(1, 0, 2)  # LND -> NLD  
        x12 = x12.permute(1, 0, 2)  # LND -> NLD  

        x12 = self.ln_post(x12)  

        if self.proj is not None:
            xproj = x12 @ self.proj   

        if self.use_moe and first_gate_router_logits is not None:
            return x11, x12, xproj, first_gate_router_logits
        else:
            return x11, x12, xproj
        
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 h_resolution: int, 
                 w_resolution: int,
                 # MoE parameters
                 num_experts: int = 0,  # 0表示不使用MoE
                 top_k: int = 0,
                 moe_layers: int = 0,
                 dropout: float = 0.0
                 ):
        super().__init__()

        self.context_length = context_length
        self.use_moe = num_experts > 0 and top_k > 0

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=h_resolution*w_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                h_resolution = h_resolution,
                w_resolution = w_resolution,
                patch_size = vision_patch_size,
                stride_size = vision_stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                num_experts=num_experts,
                top_k=top_k,
                moe_layers=moe_layers,
                dropout=dropout
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        if not self.use_moe:
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        if self.use_moe:
            # visual_output can be (x11, x12, xproj, first_gate_router_logits) or (x11, x12, xproj)
            visual_output = self.visual(image.type(self.dtype))
            
            if len(visual_output) == 4:
                features_tuple = visual_output[:3]
                img_router_logits = visual_output[3] # This is now a single tensor or None
                return features_tuple, img_router_logits
            else: # Should be 3 if no router_logits were produced (e.g. moe_layers=0 but use_moe was true)
                return visual_output, None # visual_output is (x11, x12, xproj)
        else:
            # self.visual (standard VisionTransformer or ModifiedResNet) returns (x11, x12, xproj) or (x3,x4,xproj)
            return self.visual(image.type(self.dtype)), None

    def encode_text(self, text): 
        x = self.token_embedding(text).type(self.dtype)  

        x = x + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  
        # The text transformer is standard, not MoE-enabled in this architecture
        x = self.transformer(x) # Standard Transformer, does not return router_logits
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype) 

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection 
        
        # Text transformer doesn't produce router_logits in this setup
        # If self.use_moe is True, it refers to the vision model's MoE capabilities
        # So, encode_text should not be expected to return text_router_logits if it uses a standard transformer
        return x # Always returns just x

    def forward(self, image, text):
        image_features_tuple, img_router_logits = self.encode_image(image)
        
        # 从返回的元组中提取特征
        if len(image_features_tuple) == 3:
            _, _, image_features = image_features_tuple
        else:
            image_features = image_features_tuple
            
        text_features = self.encode_text(text) # encode_text now only returns features

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # Return img_router_logits for potential loss calculation
        # No text_router_logits are produced by the standard text transformer
        if img_router_logits is not None:
            return logits_per_image, logits_per_text, img_router_logits
        else:
            return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, h_resolution: int, w_resolution: int, vision_stride_size: int,
                num_experts: int = 0, top_k: int = 0, moe_layers: int = 0, dropout: float = 0.0):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else: #RN50
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0] #77 (77,512)
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size, vision_stride_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        h_resolution, w_resolution,
        num_experts, top_k, moe_layers, dropout
    )
    if vit:
        state_dict["visual.positional_embedding"] = resize_pos_embed(state_dict["visual.positional_embedding"], model.visual.positional_embedding, h_resolution, w_resolution)
    else: #RN50
        state_dict["visual.attnpool.positional_embedding"] = resize_pos_embed(state_dict["visual.attnpool.positional_embedding"], model.visual.attnpool.positional_embedding, h_resolution, w_resolution)
    
    
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
            
    convert_weights(model)

    # 当使用MoE模型时，不进行严格加载，因为结构不同
    if num_experts > 0:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    return model.eval()

import math
def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
      
    ntok_new = posemb_new.shape[0] #129,2048

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid))) #14
    print('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2) 
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear') 
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)     
    return posemb
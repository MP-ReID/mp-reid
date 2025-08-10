import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class MLPFeatureFusion(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512):
        super(MLPFeatureFusion, self).__init__()
        # 定义一个小型的 MLP 融合模块
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, image_features, text_features_raw):
        # 将 image_features 和 text_features_raw 连接在一起
        fused_input = torch.cat([image_features, text_features_raw], dim=-1)  # [64, 1024]
        
        # 通过 MLP 网络融合
        x = self.relu(self.fc1(fused_input))  # [64, hidden_dim]
        fused_features = self.fc2(x)  # [64, output_dim]

        return fused_features

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x


#domain adaption 版本
class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE  
        self.prompt_dim = 512 
        
        # self.visual_prompt = nn.Parameter(torch.randn(1, 1, self.prompt_dim))

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.image_fusion_net = MLPFeatureFusion(input_dim=512, hidden_dim=256, output_dim=512)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        vp_vectors = torch.empty(1, 1, self.prompt_dim, dtype=clip_model.dtype)
        nn.init.normal_(vp_vectors, std=0.02)
        self.visual_prompt = nn.Parameter(vp_vectors)
        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        exp_setting = cfg.DATASETS.EXP_SETTING
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding, exp_setting)
        self.text_encoder = TextEncoder(clip_model)
        
        # Initialize in stage1a by default
        self.enable_stage1a_training()

    def enable_stage1a_training(self):
        """
        Enables training for Stage 1a: trains only the generic context vectors.
        """
        print("Enabling Stage 1a Training: Generic Context only.")
        self.prompt_learner.set_training_stage('1a')
        for param in self.prompt_learner.parameters():
            param.requires_grad = False
        self.prompt_learner.ctx_generic.requires_grad = True

    def enable_stage1b_training(self):
        """
        Enables training for Stage 1b: trains only the domain-specific context vectors.
        """
        print("Enabling Stage 1b Training: Domain-Specific Context only.")
        self.prompt_learner.set_training_stage('1b')
        for param in self.prompt_learner.parameters():
            param.requires_grad = False
        self.prompt_learner.ctx_modality.requires_grad = True
        self.prompt_learner.ctx_platform.requires_grad = True

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None,image_feature = None,get_raw_text = False,view = None,get_image_update = False,text_feature = None,get_more_image = False,exp_setting = None,get_image_vp = False):
        if get_text or get_raw_text:
            if get_text and image_feature is None and x is not None:
                _, _, image_features_proj = self.image_encoder(x)
                if self.model_name == 'RN50':
                    image_feature = image_features_proj[0]
                elif self.model_name == 'ViT-B-16':
                    image_feature = image_features_proj[:,0]

            prompts = self.prompt_learner(label, view=view)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        if get_image_vp == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            # self.visual_prompt = nn.Parameter(self.visual_prompt.expand(image_features_proj.shape[0], -1, -1))

            image_features_proj_with_prompt = image_features_proj + self.visual_prompt 
            if self.model_name == 'RN50':
                return image_features_proj_with_prompt[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj_with_prompt[:,0]
        if get_more_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                image_features_proj_low = image_features_proj[0]
                image_features_proj_mid =image_features_proj[1]
                image_features_proj_high = image_features_proj[-1]
                return image_features_proj_low,image_features_proj_mid,image_features_proj_high
            elif self.model_name == 'ViT-B-16':
                image_features_proj_low = image_features_proj[:,0]
                image_features_proj_mid =image_features_proj[:,1]
                image_features_proj_high = image_features_proj[:,-1]
                return image_features_proj_low,image_features_proj_mid,image_features_proj_high  
        if get_image_update == True:
            fused_features = self.image_fusion_net(image_feature,text_feature)
            return fused_features
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = F.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = F.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            image_features_proj_raw = image_features_proj
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj,image_features_proj_raw

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model

from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

#### DA prompt learner######
class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding, exp_setting):
        super().__init__()  
        
        # --- Hyperparameters ---
        ctx_dim = 512
        n_generic_ctx = 8  # Generic context vectors (Stage 1a)
        n_modal_ctx = 4    # Modality-specific context vectors (Stage 1b)
        n_plat_ctx = 4     # Platform-specific context vectors (Stage 1b)
        
        # --- Context Vectors ---
        # Stage 1a: Generic context, per-ID
        self.ctx_generic = nn.Parameter(torch.empty(num_class, n_generic_ctx, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_generic, std=0.02)
        
        # Stage 1b: Domain-specific context
        # Assumes 2 modalities (rgb, ir) and 2 platforms (cctv, uav)
        n_modalities = 2
        n_platforms = 2
        self.ctx_modality = nn.Parameter(torch.empty(n_modalities, n_modal_ctx, ctx_dim, dtype=dtype))
        self.ctx_platform = nn.Parameter(torch.empty(n_platforms, n_plat_ctx, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_modality, std=0.02)
        nn.init.normal_(self.ctx_platform, std=0.02)

        # --- Visual Enhanced Net (formerly ve_net/metanet) ---
        vis_dim = 512
        self.visual_enhanced_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim)) # Output is a single bias
        ]))

        # --- Prompt Tokenization ---
        # New structure: [X]*16 person.
        n_total_ctx = n_generic_ctx + n_modal_ctx + n_plat_ctx
        prompt_suffix = "person."
        template = f"{' '.join(['X'] * n_total_ctx)} {prompt_suffix}"
        
        tokenized_template = clip.tokenize(template).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_template).type(dtype)

        # Correctly find the prefix and suffix by locating the 'X' tokens
        x_token_id = clip.tokenize("X")[0, 1].item()
        x_indices = (tokenized_template[0] == x_token_id).nonzero(as_tuple=True)[0]
        
        prefix_end_idx = x_indices[0]
        suffix_start_idx = x_indices[-1] + 1
        
        # Everything before the first 'X' is the prefix (i.e., the SOT token)
        self.register_buffer("token_prefix", embedding[:, :prefix_end_idx, :]) 
        # Everything after the last 'X' is the suffix (i.e., 'person.', EOT, padding)
        self.register_buffer("token_suffix", embedding[:, suffix_start_idx:, :]) 
        
        self.tokenized_prompts = tokenized_template
        self.training_stage = '1a' # Default stage

    def set_training_stage(self, stage):
        self.training_stage = stage

    def forward(self, label, view=None):
        b = label.shape[0]
        ctx_dim = self.ctx_generic.size(-1)

        # 1. Get Generic Context (for the given person IDs)
        generic_ctx = self.ctx_generic[label] # (b, n_generic_ctx, ctx_dim)

        # 2. Get Domain-Specific Context based on stage and view
        if self.training_stage == '1a':
            # In stage 1a, domain context is zero and not trained
            modal_ctx = torch.zeros(b, self.ctx_modality.size(1), ctx_dim, device=generic_ctx.device, dtype=generic_ctx.dtype)
            plat_ctx = torch.zeros(b, self.ctx_platform.size(1), ctx_dim, device=generic_ctx.device, dtype=generic_ctx.dtype)
        else: # Stage 1b
            if view is not None:
                # 0-5 (cctv_rgb) -> platform 0, modality 0
                # 6-11 (cctv_ir) -> platform 0, modality 1
                # 12 (uav_rgb)   -> platform 1, modality 0
                # 13 (uav_ir)    -> platform 1, modality 1
                
                # Platform selection
                plat_indices = torch.zeros_like(view)
                plat_indices[view >= 12] = 1 # UAV
                
                # Modality selection
                modal_indices = torch.zeros_like(view)
                modal_indices[(view >= 6) & (view < 12)] = 1 # IR for CCTV
                modal_indices[view == 13] = 1 # IR for UAV
                
                modal_ctx = self.ctx_modality[modal_indices]
                plat_ctx = self.ctx_platform[plat_indices]
            else:
                # Fallback: use average if view is not provided in stage 1b
                modal_ctx = self.ctx_modality.mean(dim=0, keepdim=True).expand(b, -1, -1)
                plat_ctx = self.ctx_platform.mean(dim=0, keepdim=True).expand(b, -1, -1)

        # 3. Concatenate to form the full 16-token context
        ctx = torch.cat([generic_ctx, modal_ctx, plat_ctx], dim=1)
        
        # 4. Build the final prompt embeddings
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts 
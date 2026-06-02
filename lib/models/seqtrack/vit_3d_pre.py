""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import math
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    # mae ViT-B/16-224 pre-trained model
    'vit_base_patch16_224_mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch16_224_default': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    # mae ViT-L/16-224 pre-trained model
    'vit_large_patch16_224_mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    # mae ViT-H/14-224 pre-trained model
    'vit_huge_patch14_224_mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 新增：注意力模式控制（0:原始拼接注意力, 1:帧内注意力, 2:跨帧注意力）
        self.mode = 0  # 默认为原始模式
        # 存储跨帧注意力所需的模板特征和变换矩阵
        # self.template_feats = None
        # self.M = None  # 模板间几何变换矩阵
        self.template1_feats = None  # 模板1特征 (B, N_t1, C)
        self.template2_feats = None  # 模板2特征 (B, N_t2, C)


    def set_mode(self, mode):
        """设置注意力模式"""
        self.mode = mode

    # def set_template_info(self, template_feats, M=None):
    #     """设置跨帧注意力所需的模板特征和变换矩阵"""
    #     self.template_feats = template_feats
    #     self.M = M

    def set_templates(self, t1_feat, t2_feat):
        """分别传入两个模板的特征，不拼接"""
        self.template1_feats = t1_feat
        self.template2_feats = t2_feat

    def forward(self, x, mask=None):
        B, N, C = x.shape # 16,256,768
        
        if self.mode == 0:  # 原始模式：使用拼接的搜索图+模板特征
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        elif self.mode == 1:  # 帧内注意力模式（仅模板特征内部交互，带掩码）
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            # 应用目标掩码（抑制背景区域）
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)
                attn = attn.masked_fill(mask == 0, -1e10)
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        elif self.mode == 2:  # 简化跨帧模式：分别与两个模板交互后合并
            # 1. 搜索图作为Query (B, N, C)
            q = self.qkv(x)[..., :C].reshape(B, N, self.num_heads, C//self.num_heads).permute(0,2,1,3)  # (B, h, N, d)

            # 2. 与模板1交互
            k1 = self.qkv(self.template1_feats)[..., C:2*C].reshape(
                B, self.template1_feats.shape[1], self.num_heads, C//self.num_heads
            ).permute(0,2,1,3)  # (B, h, N_t1, d)
            v1 = self.qkv(self.template1_feats)[..., 2*C:].reshape(
                B, self.template1_feats.shape[1], self.num_heads, C//self.num_heads
            ).permute(0,2,1,3)  # (B, h, N_t1, d)
            attn1 = (q @ k1.transpose(-2, -1)) * self.scale  # (B, h, N, N_t1)
            attn1 = attn1.softmax(dim=-1)
            x1 = (attn1 @ v1).transpose(1,2).reshape(B, N, C)  # (B, N, C)

            # 3. 与模板2交互（同模板1）
            k2 = self.qkv(self.template2_feats)[..., C:2*C].reshape(
                B, self.template2_feats.shape[1], self.num_heads, C//self.num_heads
            ).permute(0,2,1,3)  # (B, h, N_t2, d)
            v2 = self.qkv(self.template2_feats)[..., 2*C:].reshape(
                B, self.template2_feats.shape[1], self.num_heads, C//self.num_heads
            ).permute(0,2,1,3)  # (B, h, N_t2, d)
            attn2 = (q @ k2.transpose(-2, -1)) * self.scale  # (B, h, N, N_t2)
            attn2 = attn2.softmax(dim=-1)
            x2 = (attn2 @ v2).transpose(1,2).reshape(B, N, C)  # (B, N, C)

            # 4. 合并两个模板的结果（简单平均，或可学习权重）
            x = (x1 + x2) * 0.5  # 平均合并，避免维度问题

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # 新增：几何辅助函数
    # def _generate_grid_coords(self, batch_size, grid_size, device):
    #     """生成归一化的patch网格坐标"""
    #     x = torch.linspace(-1, 1, grid_size, device=device)
    #     y = torch.linspace(-1, 1, grid_size, device=device)
    #     xx, yy = torch.meshgrid(x, y, indexing='xy')
    #     coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (grid_size^2, 2)
    #     return coords.repeat(batch_size, 1, 1)  # (B, grid_size^2, 2)
    
    # def _generate_grid_coords(self, batch_size, num_patches, device):
    #     """
    #     动态生成与patch数量完全匹配的坐标（不再假设是平方数）
    #     num_patches: 实际的patch数量（如模板特征的总patch数）
    #     """
    #     # 生成1D均匀分布的坐标，再转换为2D网格（按行优先排列）
    #     x = torch.linspace(-1, 1, int(num_patches**0.5) + 1, device=device)  # 略多生成一点
    #     y = torch.linspace(-1, 1, int(num_patches**0.5) + 1, device=device)
    #     xx, yy = torch.meshgrid(x, y, indexing='xy')
    #     coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # 生成网格坐标
    #     coords = coords[:num_patches]  # 截取前num_patches个坐标，确保数量匹配
    #     return coords.repeat(batch_size, 1, 1)  # (B, num_patches, 2)

    # def _apply_affine(self, coords, M):
    #     """对坐标应用仿射变换矩阵M"""
    #     coords_hom = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)  # (B, N, 3)
    #     transformed = torch.matmul(M.unsqueeze(1), coords_hom.unsqueeze(-1)).squeeze(-1)  # (B, N, 3)
    #     return transformed[..., :2]  # 取xy坐标

    # def _coord_similarity(self, coords1, coords2):
    #     """计算坐标间余弦相似度"""
    #     coords1_norm = F.normalize(coords1, dim=-1)
    #     coords2_norm = F.normalize(coords2, dim=-1)
    #     return torch.matmul(coords1_norm, coords2_norm.transpose(-2, -1))  # (B, N1, N2)


# class Block(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        # 新增mask参数，用于帧内注意力
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, C, H, W = x.shape
        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class TemplateTransformer(nn.Module):
    """轻量级Transformer预测双模板间的变换矩阵"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, 9)  # 3x3仿射矩阵参数
        )

    def forward(self, temp1_feat, temp2_feat):
        # 交叉注意力融合双模板特征
        cross_feat, _ = self.cross_attn(temp1_feat, temp2_feat, temp2_feat)
        # 全局池化后预测变换矩阵
        global_feat = cross_feat.mean(dim=1)  # (B, dim)
        M_flat = self.mlp(global_feat)
        return M_flat.reshape(-1, 3, 3)  # (B, 3, 3)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, search_size=384, template_size=192,
                 patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 search_number=1, template_number=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embed_dim_list = [embed_dim]
        self.num_search = search_number
        self.num_template = template_number

        # 分别为搜索图和模板创建PatchEmbed（保持原始尺寸）
        self.patch_embed_search = PatchEmbed(
            img_size=search_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_template = PatchEmbed(
            img_size=template_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # 计算patch数量
        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)
        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)

        # 位置嵌入：搜索图 + 模板1 + 模板2
        self.pos_embed_search = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))
        self.pos_embed_template = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 初始化注意力块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        # 新增：双模板变换矩阵预测器
        # self.template_transformer = TemplateTransformer(embed_dim, num_heads)

        self.norm = norm_layer(embed_dim)
        # trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.pos_embed_search, std=.02)
        trunc_normal_(self.pos_embed_template, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, images_list):
        num_template = self.num_template
        # images_list结构：[template1, template2, search]
        template1, template2, search = images_list[:1], images_list[1:2], images_list[num_template:]
        B = template1[0].shape[0]

        # 1. 提取模板特征并应用帧内注意力精修
        # 模板1特征
        z1 = self.patch_embed_template(template1[0])  # (B, N_t, C)
        z1 = z1 + self.pos_embed_template
        z1_mask = self._generate_template_mask(z1.shape[1])  # 生成目标掩码（假设已知目标区域）
        # 模板2特征
        z2 = self.patch_embed_template(template2[0])
        z2 = z2 + self.pos_embed_template
        z2_mask = self._generate_template_mask(z2.shape[1])

        # 前3层注意力块启用帧内模式处理模板
        for i in range(min(3, len(self.blocks))):
            self.blocks[i].attn.set_mode(1)  # 帧内模式
            z1 = self.blocks[i](z1, mask=z1_mask)
            z2 = self.blocks[i](z2, mask=z2_mask)

        # 2. 预测双模板间的变换矩阵M
        # M = self.template_transformer(z1, z2)  # (B, 3, 3)

        # 3. 提取搜索图特征
        x = self.patch_embed_search(search[0])  # (B, N_s, C)
        x = x + self.pos_embed_search

        # 4. 中间层启用跨帧注意力融合（4-10层）
        template_feats = torch.cat([z1, z2], dim=1)  # 合并双模板特征 (B, 2*N_t, C)
        for i in range(3, min(10, len(self.blocks))):
            self.blocks[i].attn.set_mode(2)  # 跨帧模式
            # self.blocks[i].attn.set_template_info(template_feats, M)
            self.blocks[i].attn.set_templates(z1, z2)
            x = self.blocks[i](x)

        x = torch.cat([x, template_feats], dim=1) # 额外合并x z
        # 5. 最后几层用原始模式精修
        for i in range(10, len(self.blocks)):
            self.blocks[i].attn.set_mode(0)  # 原始模式
            x = self.blocks[i](x)

        x = self.norm(x)  # (B, N_s, C)
        return x

    def forward(self, images_list):
        xz = self.forward_features(images_list)
        out=[xz]
        return out

    def _generate_template_mask(self, num_patches):
        """生成模板的目标区域掩码（示例：中心区域为目标）"""
        # 实际使用时应根据模板的真实边界框生成
        mask = torch.zeros(1, num_patches)
        center_idx = num_patches // 2
        mask[0, center_idx-2:center_idx+2] = 1  # 中心4个patch设为目标
        return mask.to(next(self.parameters()).device)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def vit_3d_base_patch16(pretrained=False, pretrain_type='default',
                         search_size=384, template_size=192, **kwargs):
    patch_size = 16
    model = VisionTransformer(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_base_patch16_224_' + pretrain_type
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type,
                        num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def vit_3d_large_patch16(pretrained=False, pretrain_type='default',
                              search_size=384, template_size=192, **kwargs):
    patch_size = 16
    model = VisionTransformer(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_large_patch16_224_' + pretrain_type
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def vit_huge_patch14(pretrained=False, pretrain_type='default',
                             search_size=364, template_size=182, **kwargs):
    patch_size = 14
    model = VisionTransformer(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_huge_patch14_224_' + pretrain_type
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model,
                        pretrain_type, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# def load_pretrained(model, pretrain_type='default', cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True):
def load_pretrained(model, pretrain_type='default', cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=False):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        print("Pretrained model URL is invalid, using random initialization.")
        return
    state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
    if pretrain_type == 'mae':
        state_dict = state_dict['model']

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        print('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            print('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            print('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if pretrain_type == "mae":
        pass
    elif num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']

    # adjust position encoding
    pe = state_dict['pos_embed'][:,1:,:]
    b_pe, hw_pe, c_pe = pe.shape
    side_pe = int(math.sqrt(hw_pe))
    side_num_patches_search = int(math.sqrt(model.num_patches_search))
    side_num_patches_template = int(math.sqrt(model.num_patches_template))
    pe_2D = pe.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0,3,1,2])  #b,c,h,w
    if side_pe != side_num_patches_search:
        pe_s_2D = nn.functional.interpolate(pe_2D, [side_num_patches_search, side_num_patches_search], align_corners=True, mode='bicubic')
        pe_s = torch.flatten(pe_s_2D.permute([0,2,3,1]),1,2)
    else:
        pe_s = pe
    if side_pe != side_num_patches_template:
        pe_t_2D = nn.functional.interpolate(pe_2D, [side_num_patches_template, side_num_patches_template], align_corners=True, mode='bicubic')
        pe_t = torch.flatten(pe_t_2D.permute([0, 2, 3, 1]), 1, 2)
    else:
        pe_t = pe
    pe_xz = torch.cat((pe_s, pe_t), dim=1)
    state_dict['pos_embed'] = pe_xz
    del state_dict['cls_token']
    model.load_state_dict(state_dict, strict=strict)

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

import numpy as np



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



try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# ----------------- helper ops -----------------
def l2_normalize(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True).clamp(min=eps))

def patch_tokens_from_feat(F):
    B, C, H, W = F.shape
    return F.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]

def feat_from_tokens(tokens, H, W):
    B, N, C = tokens.shape
    return tokens.transpose(1, 2).reshape(B, C, H, W)

# ------------- 辅助：把 coords -> geom_bias / diff_mask -------------
def build_geom_bias_and_diffmask_from_coords(coords, B, Ns, Nt, num_heads, device, base_val=4.0):
    """
    coords: list length B, each element list of matched pairs between template1 and template2 in patch-grid coords:
            coords[b] = [(x1,y1,x2,y2), ...] where indices refer to template patch grid (two templates)
            NOTE: we assume templates were concatenated as [T1_patches, T2_patches] when forming template tokens indices.
    Ns: number of search tokens (e.g., 256)
    Nt: number of template tokens (e.g., 512 = 256 + 256)
    Returns:
      geom_bias: torch.Tensor [B, heads, N_total, N_total]
      diff_mask_attn: torch.Tensor [B, 1, N_total, N_total]  (1 means high-difference => suppress)
    """
    N_total = Ns + Nt
    # init
    geom_bias = torch.zeros((B, num_heads, N_total, N_total), dtype=torch.float32, device=device)
    diff_mask = torch.zeros((B, 1, N_total, N_total), dtype=torch.float32, device=device)

    # We'll encourage attention between matched template-token pairs
    # and also mark diff_mask for template-template positions if needed (here init zero; DAF will provide actual diff)
    # Map patch-grid coords to template token indices:
    # Assume template patches layout is square: Wt = sqrt(Nt/num_templates)? For two templates each of size Nt/2
    # For robustness, user should pass patch-grid dims; here we assume each template has Nt_each = Nt//2 patches arranged as sqrt x sqrt.
    Nt_each = Nt // 2
    Wt = int(np.sqrt(Nt_each))
    if Wt * Wt != Nt_each:
        # fallback: assume Nt_each is rectangular; user should replace with correct Ht,Wt
        Wt = int(round(np.sqrt(Nt_each)))
    for b in range(B):
        pairs = coords[b]
        for (x1, y1, x2, y2) in pairs:
            # indexes in each template's patch-grid
            idx_t1 = int(round(y1)) * Wt + int(round(x1))
            idx_t2 = int(round(y2)) * Wt + int(round(x2))

            # clamp to legal range
            idx_t1 = max(0, min(idx_t1, Nt_each - 1))
            idx_t2 = max(0, min(idx_t2, Nt_each - 1))
            # global indices: template tokens are concatenated after search tokens.
            # template ordering: [T1_tokens (0..Nt_each-1), T2_tokens (Nt_each..Nt-1)]
            # Assume pair (x1,y1) in T1 and (x2,y2) in T2, so idx_t2_global = Ns + Nt_each + idx_t2
            idx_t1_global = Ns + idx_t1
            idx_t2_global = Ns + Nt_each + idx_t2

            # set geom bias both directions
            if idx_t1_global < N_total and idx_t2_global < N_total:
                geom_bias[b, :, idx_t1_global, idx_t2_global] = base_val
                geom_bias[b, :, idx_t2_global, idx_t1_global] = base_val
            # optionally set a small bias linking T1/T2 to self (not necessary)
    # diff_mask is expected from DAF; here we keep zeros. Caller should overwrite using DAF outputs:
    return geom_bias, diff_mask



# ----------------- Geometry helpers -----------------
def estimate_homography_ransac(pts_src, pts_dst, thresh=4.0, max_iters=1000):
    if pts_src is None or pts_dst is None:
        return None
    pts_src = np.asarray(pts_src)
    pts_dst = np.asarray(pts_dst)
    if pts_src.shape[0] < 4:
        return None
    if _HAS_CV2:
        try:
            H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransacReprojThreshold=thresh, maxIters=max_iters)
            return H
        except Exception:
            pass
    # fallback simple RANSAC + DLT (robust but basic)
    N = pts_src.shape[0]
    bestH = None; best_in = 0
    for _ in range(min(max_iters, 300)):
        idx = np.random.choice(N, 4, replace=False)
        src4 = pts_src[idx]; dst4 = pts_dst[idx]
        A = []
        for i in range(4):
            x, y = src4[i]; u, v = dst4[i]
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        A = np.asarray(A)
        try:
            _, _, Vt = np.linalg.svd(A)
            h = Vt[-1, :]
            H = h.reshape(3,3)
        except Exception:
            continue
        src_h = np.concatenate([pts_src, np.ones((N,1))], axis=1)
        proj = (H @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        d = np.linalg.norm(proj - pts_dst, axis=1)
        inliers = np.sum(d < thresh)
        if inliers > best_in:
            best_in = inliers
            bestH = H
            if best_in > N * 0.6:
                break
    return bestH

def warp_features_by_homography(x, H, out_size):
    # F: [B,C,Hf,Wf] , H: 3x3 numpy, out_size: (Hout,Wout) in feature grid coords
    B, C, Hf, Wf = x.shape
    Hout, Wout = out_size
    # build pixel coords grid
    ys = np.linspace(0, Hf-1, Hout)
    xs = np.linspace(0, Wf-1, Wout)
    xs_grid, ys_grid = np.meshgrid(xs, ys)
    pts = np.stack([xs_grid.ravel(), ys_grid.ravel(), np.ones(xs_grid.size)], axis=0)  # (3, N)
    # H_inv = np.linalg.inv(H)
    H_inv = np.linalg.pinv(H)
    # H_inv = np.linalg.inv(H+1e-8 * np.eye(H.shape[1]))
    src = H_inv @ pts
    src = src[:2] / src[2:3]
    src_x = torch.from_numpy(src[0].reshape(Hout, Wout)).to(x.device).float()
    src_y = torch.from_numpy(src[1].reshape(Hout, Wout)).to(x.device).float()
    grid_x = (src_x / (Wf - 1)) * 2 - 1
    grid_y = (src_y / (Hf - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)  # Hout x Wout x 2
    grid = grid.unsqueeze(0).repeat(B,1,1,1)
    warped = F.grid_sample(x, grid, align_corners=True)
    return warped

# ----------------- TinyMatcher (geometry-lite) -----------------
class TinyMatcher(nn.Module):
    def __init__(self, topk=128, temp=0.05):
        super().__init__()
        self.topk = topk
        self.temp = temp

    def forward(self, F1, F2):
        # F1, F2: [B,C,H,W]
        B, C, H, W = F1.shape
        t1 = patch_tokens_from_feat(F1)   # [B, N, C]
        t2 = patch_tokens_from_feat(F2)
        t1n = l2_normalize(t1, dim=-1)
        t2n = l2_normalize(t2, dim=-1)
        sim = torch.einsum('bkc,blc->bkl', t1n, t2n) / self.temp  # [B,N,N]
        B_, N, _ = sim.shape
        sim_flat = sim.reshape(B_, -1)
        k = min(self.topk, sim_flat.shape[-1])
        topk_vals, topk_idx = torch.topk(sim_flat, k=k, dim=-1)
        i1 = (topk_idx // N).cpu().numpy()
        i2 = (topk_idx % N).cpu().numpy()
        coords = []
        for b in range(B_):
            c_b = []
            for kk in range(i1.shape[1]):
                idx1 = i1[b, kk]; idx2 = i2[b, kk]
                y1 = idx1 // W; x1 = idx1 % W
                y2 = idx2 // W; x2 = idx2 % W
                c_b.append((float(x1 + 0.5), float(y1 + 0.5), float(x2 + 0.5), float(y2 + 0.5)))
            coords.append(c_b)
        return topk_vals, coords

# ----------------- DAFusion -----------------
class DAFusion(nn.Module):
    def __init__(self, in_ch, proto_k=6, lowrank=32):
        super().__init__()
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_ch, max(8, in_ch//4), 1), nn.ReLU(),
            nn.Conv2d(max(8, in_ch//4), 1, 1)
        )
        self.lowrank_proj = nn.Sequential(
            nn.Conv2d(in_ch, lowrank, 1),
            nn.ReLU(),
            nn.Conv2d(lowrank, in_ch, 1)
        )
        self.pool_q = nn.Linear(in_ch, proto_k)
        self.pool_v = nn.Linear(in_ch, in_ch)
        self.norm = nn.LayerNorm(in_ch)
        self.proto_k = proto_k

    def forward(self, F1, F2_warp):
        # F1,F2_warp: [B,C,H,W]
        B, C, H, W = F1.shape
        # print(F1.shape,F2_warp.shape,'========')
        # diff = torch.abs(F1 - F2_warp).mean(1, keepdim=True)  # [B,1,H,W]
        diff = torch.abs(F1 - F2_warp)  # [B,1,H,W]
        diff_logits = self.diff_conv(diff)  # [B,1,H,W]
        diff_mask = torch.sigmoid(diff_logits)  # 1 -> different/unreliable
        F1r = self.lowrank_proj(F1)
        gate = 1.0 - diff_mask
        Ff = F1r * gate + F2_warp * (1.0 - gate)
        tokens = patch_tokens_from_feat(Ff)  # [B,N,C]
        assign_logits = self.pool_q(tokens)  # [B,N,K]
        assign = F.softmax(assign_logits, dim=1).transpose(1,2)  # [B, K, N] normalized over tokens
        values = self.pool_v(tokens)  # [B,N,C]
        prototypes = torch.einsum('bkn,bnc->bkc', assign, values)  # [B,K,C]
        prototypes = self.norm(prototypes)
        return tokens, diff_mask, prototypes

# # ----------------- ROIMatcher (lightweight) -----------------
# class ROIMatcher(nn.Module):
#     def __init__(self, proto_dim, search_ch, roi_size=16):
#         super().__init__()
#         self.roi_size = roi_size
#         self.to_q = nn.Linear(proto_dim, proto_dim)
#         self.to_k = nn.Conv2d(search_ch, proto_dim, 1)
#         self.to_v = nn.Conv2d(search_ch, proto_dim, 1)
#         self.out_head = nn.Sequential(
#             nn.Linear(proto_dim, max(8, proto_dim//4)),
#             nn.GELU(),
#             nn.Linear(max(8, proto_dim//4), 1)
#         )

#     def forward(self, prototypes, Fs, rois):
#         B, K, C = prototypes.shape
#         _, Cs, Hs, Ws = Fs.shape
#         q = self.to_q(prototypes)
#         responses = []
#         for b in range(B):
#             Fs_b = Fs[b:b+1]
#             res_b = []
#             for k in range(K):
#                 cx, cy = rois[b][k]
#                 x0 = max(0, int(round(cx - self.roi_size//2)))
#                 y0 = max(0, int(round(cy - self.roi_size//2)))
#                 x1 = min(Ws, x0 + self.roi_size)
#                 y1 = min(Hs, y0 + self.roi_size)
#                 if x1 <= x0 or y1 <= y0:
#                     res = torch.zeros((1, self.roi_size, self.roi_size), device=Fs.device)
#                     res_b.append(res); continue
#                 patch = Fs_b[:, :, y0:y1, x0:x1]
#                 k_proj = self.to_k(patch).flatten(2).transpose(1,2)  # [1, Nroi, C]
#                 v_proj = self.to_v(patch).flatten(2).transpose(1,2)
#                 qk = torch.einsum('kc,bnc->kbn', q[b:b+1,k:k+1,:], k_proj)
#                 attn = F.softmax(qk / (C**0.5), dim=-1)
#                 weighted = (attn @ v_proj).squeeze(0).squeeze(0)
#                 score = self.out_head(weighted)
#                 patch_score = torch.zeros((1, self.roi_size, self.roi_size), device=Fs.device)
#                 patch_score[0, self.roi_size//2, self.roi_size//2] = score
#                 res_b.append(patch_score)
#             responses.append(torch.stack(res_b, dim=0))
#         return responses

# # ----------------- TemplateMemory -----------------
# class TemplateMemory:
#     def __init__(self, max_slots=3):
#         self.max_slots = max_slots
#         self.slots = []

#     def add(self, prototypes, avg_feat):
#         if len(self.slots) >= self.max_slots:
#             self.slots.pop(0)
#         self.slots.append({'proto': prototypes.detach().cpu(), 'avg': avg_feat.detach().cpu()})

#     def route(self, search_avg_feat, topk=1):
#         if len(self.slots) == 0:
#             return []
#         sims = []
#         sarr = search_avg_feat.detach().cpu().numpy().reshape(-1)
#         for s in self.slots:
#             a = s['avg'].reshape(-1)
#             cos = (a @ sarr) / (np.linalg.norm(a) * np.linalg.norm(sarr) + 1e-8)
#             sims.append(cos)
#         idx = np.argsort(sims)[-topk:][::-1]
#         selected = [self.slots[i]['proto'] for i in idx]
#         return selected
class ROIGuidedMatcher(nn.Module):
    def __init__(self, dim, num_proto=4):
        super().__init__()
        self.num_proto = num_proto
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, search_tokens, prototypes):
        # search_tokens: [B, 256, C]
        # prototypes:    [B, P, C]   (from DAF)
        Q = self.q_proj(search_tokens)   # [B, 256, C]
        K = self.k_proj(prototypes)      # [B, P, C]
        V = self.v_proj(prototypes)      # [B, P, C]

        attn = torch.softmax(Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5), dim=-1)  # [B, 256, P]
        out = attn @ V    # [B, 256, C]
        search_tokens = search_tokens + self.out_proj(out)  # residual

        return search_tokens  # shape unchanged


class TemplateMemory(nn.Module):
    def __init__(self, dim, mem_size=10):
        super().__init__()
        self.mem_size = mem_size
        self.memory = nn.Parameter(torch.zeros(mem_size, dim))  # learnable, init 0
        nn.init.normal_(self.memory, std=0.02)

        self.routing_proj = nn.Linear(dim, 1)

    def forward(self, template_tokens, search_tokens):
        # template_tokens: [B, 512, C]
        # search_tokens:   [B, 256, C]
        B, Nt, C = template_tokens.shape

        # average search tokens to global query
        query = search_tokens.mean(dim=1)  # [B, C]
        scores = torch.softmax(self.routing_proj(query)[:, None, :], dim=1)  # [B,1,1]

        # weighted memory (broadcast across Nt)
        mem = self.memory.mean(dim=0, keepdim=True).expand(B, Nt, C)  # [B,Nt,C]
        template_tokens = template_tokens + scores.view(B,1,1) * mem

        return template_tokens  # shape unchanged
    
# ----------------- Original Attention / Block / PatchEmbed (adapted) -----------------
# Keep parameter names and shapes unchanged, only add optional geom/diff handling in forward.
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

        # NEW lightweight extra proj for geom bias generation if needed
        self.geom_bias_proj = nn.Linear(dim, num_heads)  # new param (does not break pretrained)

    def forward(self, x, geom_bias=None, diff_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]

        # optionally add geometric bias (precomputed externally and passed in)
        if geom_bias is not None:
            # geom_bias shape expected [B, heads, N, N]
            attn = attn + geom_bias

        # optionally apply diff mask for template-template suppression
        if diff_mask is not None:
            # diff_mask broadcast to [B, 1, N, N] or [B, heads, N, N]
            attn = attn * (1.0 - diff_mask)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_heads = num_heads
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, geom_bias=None, diff_mask=None):
        # pass geom_bias/diff_mask down to Attention (defaults None)
        x = x + self.drop_path(self.attn(self.norm1(x), geom_bias=geom_bias, diff_mask=diff_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# ----------------- Modified VisionTransformer -----------------
class VisionTransformer(nn.Module):
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

        # original patch embed (unchanged)
        self.patch_embed = PatchEmbed(
                img_size=search_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)
        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_search + self.num_patches_template, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # --- NEW lightweight modules (extra params only) ---
        self.tiny_matcher = TinyMatcher(topk=64)
        self.daf = DAFusion(in_ch=embed_dim, proto_k=6, lowrank=max(8, embed_dim//8))
        self.roi_matcher = ROIGuidedMatcher(dim=embed_dim)
        self.template_memory = TemplateMemory(dim=embed_dim)
        # -----------------------------------------------------------------

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        # keep original initialization scheme for other modules
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, images_list):
        """
        images_list: [t1, t2, s]  (two templates then search)
        We will produce:
        search_tokens: [B, Ns, C]
        template_tokens: [B, Nt, C]  (拼接两个模板的 tokens: [T1_tokens, T2_tokens])
        Then iterate over self.blocks calling blk(search_tokens, template_tokens, geom_bias, diff_mask)
        Finally output xz = cat([search_tokens, template_tokens], dim=1) normalized
        """
        num_template = self.num_template  # assume 2
        template_list = images_list[0:num_template]
        search_list = images_list[num_template:]
        num_search = len(search_list)
        device = template_list[0].device


        z_list = []
        for i in range(num_template):
            z = template_list[i]
            z = self.patch_embed(z)
            z = z + self.pos_embed[:, self.num_patches_search:, :]
            z_list.append(z)
        template_tokens = torch.cat(z_list, dim=1)

        x_list = []
        for i in range(num_search):
            x = search_list[i]
            x = self.patch_embed(x)
            x = x + self.pos_embed[:, :self.num_patches_search, :]
            x_list.append(x)
        search_tokens = torch.cat(x_list, dim=1)


        # # --- 1) PatchEmbed for templates and search (保持与你原代码一致) ---
        # # template_tokens: list of [B, Nt_each, C]
        # z_list = []
        # for t in template_list:
        #     z = self.patch_embed(t)  # NOTE: 这里你的 patch_embed 是为 search_size 定制的，但原始代码里对 template 使用了同一 patch_embed -> 保持不变
        #     z_list.append(z)
        # # concat templates along token dim -> template_tokens [B, Nt, C]  (Nt = Nt_each * 2)
        # template_tokens = torch.cat(z_list, dim=1)

        # x_list = []
        # for s in search_list:
        #     x = self.patch_embed(s)
        #     x_list.append(x)
        # search_tokens = torch.cat(x_list, dim=1)  # [B, Ns, C]

        # --- 2) build feature maps for geometry / DAF (we produce F1,F2 and warp F2->F1) ---
        # convert tokens -> feature maps for matching
        Nt_each = template_tokens.shape[1] // 2
        Ht = Wt = int(np.sqrt(Nt_each))
        # template tokens layout: [T1_tokens (0..Nt_each-1), T2_tokens (Nt_each..Nt-1)]
        T1_tokens = template_tokens[:, :Nt_each, :]  # [B, Nt_each, C]
        T2_tokens = template_tokens[:, Nt_each:, :]  # [B, Nt_each, C]
        F1 = T1_tokens.transpose(1,2).reshape(T1_tokens.shape[0], T1_tokens.shape[2], Ht, Wt)
        F2 = T2_tokens.transpose(1,2).reshape(T2_tokens.shape[0], T2_tokens.shape[2], Ht, Wt)

        # TinyMatcher -> coords
        _, coords = self.tiny_matcher(F1, F2)  # coords: list per batch of tuples (x1,y1,x2,y2)
        # estimate H and warp F2 -> F1 (per-batch)
        F2_warp_list = []
        B = F1.shape[0]
        for b in range(B):
            # build pts arrays from coords[b] and estimate homography mapping T2->T1
            pts1 = []
            pts2 = []
            for (x1,y1,x2,y2) in coords[b]:
                pts1.append([x1, y1]); pts2.append([x2, y2])
            H = estimate_homography_ransac(np.array(pts2), np.array(pts1))
            if H is None:
                H = np.eye(3)
            warped = warp_features_by_homography(F2[b:b+1], H, out_size=(Ht, Wt))
            F2_warp_list.append(warped)
        F2_warp = torch.cat(F2_warp_list, dim=0)

        # DAF fusion: produce tokens_fused and diff_mask and prototypes
        tokens_fused, diff_mask_map, prototypes = self.daf(F1, F2_warp)  # tokens [B, Nt_each, C]
        # here tokens_fused corresponds spatially to Nt_each tokens; we need to produce full template_tokens shape [B, Nt, C]
        # Minimal approach: expand tokens_fused to both T1 & T2 positions (or use tokens_fused for T1 and warped-projection for T2)
        # For simplicity, we put fused tokens into both halves (T1 and T2) so template_tokens_fused [B, Nt, C]:
        

        # ROI-Guided Tri-Match: refine search tokens
        search_tokens = self.roi_matcher(search_tokens, prototypes)

        # Memory Routing: refine template tokens
        tokens_fused = self.template_memory(tokens_fused, search_tokens)

        template_tokens_fused = torch.cat([tokens_fused, tokens_fused], dim=1)  # [B, Nt, C]


        # --- 3) build geom_bias and diff_mask_attn (for attention) using coords & diff_mask_map  ---
        Ns = search_tokens.shape[1]; Nt = template_tokens_fused.shape[1]
        geom_bias, diff_mask_attn = build_geom_bias_and_diffmask_from_coords(coords, B, Ns, Nt, num_heads=self.blocks[0].attn.num_heads, device=device)
        # incorporate DAF's diff_map (B,1,Ht,Wt) -> token-space diff by pooling:
        try:
            Nt_total = Nt
            nt_side = int(np.sqrt(Nt//2))  # side length per template
            dm = F.adaptive_avg_pool2d(diff_mask_map, (nt_side, nt_side))  # [B,1,nt_side,nt_side]
            dm_flat = dm.flatten(2)  # [B,1,Nt_each]
            dm_tokens = torch.cat([dm_flat, dm_flat], dim=-1)  # [B,1,Nt]
            diff_mask_attn = (dm_tokens.unsqueeze(-1) + dm_tokens.unsqueeze(-2)) / 2.0  # [B,1,Nt,Nt], embed into full N_total below
            # expand to full N_total shape (including search tokens) by zero padding for search rows/cols
            padded = torch.zeros((B,1,Ns+Nt,Ns+Nt), device=device)
            # place template-template block at indices [Ns:, Ns:]
            padded[:, :, Ns:, Ns:] = diff_mask_attn
            diff_mask_attn = padded  # final [B,1,N_total,N_total]
        except Exception:
            diff_mask_attn = None

        # --- 4) run transformer blocks, each block accepts (search_tokens, template_tokens_fused, geom_bias, diff_mask_attn) ---
        # xz = torch.cat([search_tokens, template_tokens_fused], dim=1)
        xz = torch.cat([search_tokens, template_tokens_fused+template_tokens], dim=1)
        xz = self.pos_drop(xz)
        for blk in self.blocks:
            xz = blk(xz, geom_bias=geom_bias, diff_mask=diff_mask_attn)

        # for blk in self.blocks:   #batch is the first dimension.
        #     if self.use_checkpoint:
        #         xz = checkpoint.checkpoint(blk, xz)
        #     else:
        #         xz = blk(xz)

        # xz = self.norm(xz) # B,N,C

        # final output concatenation and normalization
        
        xz = self.norm(xz)
        return xz

    def forward(self, images_list):
        xz = self.forward_features(images_list)
        return [xz]



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

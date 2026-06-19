"""
SeqTrack Model  (RMP-SeqTrack: Reliable Motion SeqTrack)
"""
import torch
import math
from torch import nn
import torch.nn.functional as F

from lib.utils.misc import NestedTensor

from lib.models.seqtrack.encoder import build_encoder
from .decoder import build_decoder
from .motion_module import MotionModule
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class SEQTRACK(nn.Module):
    """ This is the base class for SeqTrack (with optional RMP motion module) """
    def __init__(self, encoder, decoder, hidden_dim,
                 bins=1000, feature_type='x', num_frames=1, num_template=1,
                 motion_cfg=None):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
            motion_cfg: optional edict with MOTION config; None disables motion.
        """
        super().__init__()
        self.encoder = encoder
        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.side_fx = int(math.sqrt(self.num_patch_x))
        self.side_fz = int(math.sqrt(self.num_patch_z))
        self.hidden_dim = hidden_dim
        self.bottleneck = nn.Linear(encoder.num_channels, hidden_dim) # the bottleneck layer, which aligns the dimmension of encoder and decoder
        self.decoder = decoder
        self.vocab_embed = MLP(hidden_dim, hidden_dim, bins+2, 3)

        self.num_frames = num_frames
        self.num_template = num_template
        self.feature_type = feature_type

        # ---- RMP Motion Module (optional) ----
        self.enable_motion = False
        self.enable_motion_guided_attn = False
        self.motion_module = None
        motion_encoder_enabled = motion_cfg is not None \
            and motion_cfg.get('ENABLE', False) \
            and motion_cfg.get('ENABLE_MOTION_ENCODER', True)
        if motion_encoder_enabled:
            self.enable_motion = True
            self.enable_motion_guided_attn = motion_cfg.get('ENABLE_MOTION_GUIDED_ATTN', True)
            self.motion_module = MotionModule(
                hidden_dim=motion_cfg.get('HIDDEN_DIM', hidden_dim),
                history_length=motion_cfg.get('HISTORY_LENGTH', 5),
                num_layers=motion_cfg.get('NUM_LAYERS', 2),
                num_heads=motion_cfg.get('NUM_HEADS', 8),
                enable_reliability=motion_cfg.get('ENABLE_RELIABILITY', True),
                motion_scale=motion_cfg.get('MOTION_SCALE', 128.0),
            )

        # Different type of visual features for decoder.
        # Since we only use one search image for now, the 'x' is same with 'x_last' here.
        if self.feature_type == 'x':
            num_patches = self.num_patch_x * self.num_frames
        elif self.feature_type == 'xz':
            num_patches = self.num_patch_x * self.num_frames + self.num_patch_z * self.num_template
        elif self.feature_type == 'token':
            num_patches = 1
        else:
            raise ValueError('illegal feature type')

        # position embeding for the decocder
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        pos_embed = get_sinusoid_encoding_table(num_patches, self.pos_embed.shape[-1], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    # ------------------------------------------------------------------
    # Public helpers for the actor
    # ------------------------------------------------------------------
    def compute_motion(self, historical_boxes: torch.Tensor, return_aux: bool = False):
        """
        Run the motion pipeline on a stack of historical boxes.

        Args:
            historical_boxes: [B, N, 4]   (N = history_length, format [x,y,w,h])
            return_aux: return auxiliary tensors for loss / analysis
        Returns:
            dict with 'motion_bias' (and optionally aux tensors)
        """
        if not self.enable_motion or self.motion_module is None:
            dummy = {'motion_bias': None}
            if return_aux:
                dummy['motion_feature'] = None
            return dummy
        return self.motion_module(historical_boxes, return_aux=return_aux)

    def compute_motion_loss(self, aux: dict) -> torch.Tensor:
        """Compute auxiliary motion regularisation loss."""
        if not self.enable_motion or self.motion_module is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.motion_module.compute_motion_loss(aux)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def forward(self, images_list=None, xz=None, seq=None, mode="encoder",
                historical_boxes=None, return_motion_aux=False):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        historical_boxes: [B, N, 4] optional boxes for motion module
        return_motion_aux: if True, return aux motion tensors alongside output
        """
        if mode == "encoder":
            return self.forward_encoder(images_list)
        elif mode == "decoder":
            return self.forward_decoder(xz, seq, historical_boxes, return_motion_aux)
        else:
            raise ValueError

    def forward_encoder(self, images_list):
        # Forward the encoder
        xz = self.encoder(images_list)
        return xz

    def _get_motion_bias(self, historical_boxes, return_aux):
        """Compute motion_bias from historical boxes; returns (motion_bias, aux_dict_or_None)."""
        if not self.enable_motion or not self.enable_motion_guided_attn:
            return None, {}
        if historical_boxes is None:
            return None, {}
        motion_out = self.compute_motion(historical_boxes, return_aux=return_aux)
        return motion_out.get('motion_bias', None), motion_out

    def forward_decoder(self, xz, sequence,
                        historical_boxes=None, return_motion_aux=False):

        xz_mem = xz[-1]
        B, _, _ = xz_mem.shape

        # get different type of visual features for decoder.
        if self.feature_type == 'x': # get features of all search images
            dec_mem = xz_mem[:,0:self.num_patch_x * self.num_frames]
        elif self.feature_type == 'xz': # get all features of search and template images
            dec_mem = xz_mem
        elif self.feature_type == 'token': # get an average feature vector of search and template images.
            dec_mem = xz_mem.mean(1).unsqueeze(1)
        else:
            raise ValueError('illegal feature type')

        # align the dimensions of the encoder and decoder
        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)  # [B, S, D]  S = num_patches
        dec_mem = dec_mem.permute(1,0,2)  # [S, B, D]  ready for MultiheadAttention

        # ltr_collate_stack1 stacks on dim=1, so historical_boxes arrives as
        # [H, B, 4] instead of [B, H, 4].  Transpose back to canonical shape.
        if historical_boxes is not None:
            hb = historical_boxes
            if hb.ndim == 3 and hb.shape[0] != B and hb.shape[1] == B:
                historical_boxes = hb.transpose(0, 1).contiguous()  # [H,B,4] → [B,H,4]

        # Compute motion bias
        motion_bias, motion_aux = self._get_motion_bias(historical_boxes, return_motion_aux)

        out = self.decoder(dec_mem,
                           self.pos_embed.permute(1,0,2).expand(-1,B,-1),
                           sequence,
                           motion_bias=motion_bias)
        out = self.vocab_embed(out) # embeddings --> likelihood of words

        if return_motion_aux and motion_aux:
            return out, motion_aux
        return out

    def inference_decoder(self, xz, sequence, window=None, seq_format='xywh',
                          historical_boxes=None):
        # Forward the decoder
        xz_mem = xz[-1]
        B, _, _ = xz_mem.shape

        # get different type of visual features for decoder.
        if self.feature_type == 'x':
            dec_mem = xz_mem[:,0:self.num_patch_x]
        elif self.feature_type == 'xz':
            dec_mem = xz_mem
        elif self.feature_type == 'token':
            dec_mem = xz_mem.mean(1).unsqueeze(1)
        else:
            raise ValueError('illegal feature type')

        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)  #[B,NL,D]
        dec_mem = dec_mem.permute(1,0,2)  #[NL,B,D]

        # Compute motion bias for inference
        motion_bias, _ = self._get_motion_bias(historical_boxes, return_aux=False)

        out = self.decoder.inference(dec_mem,
                                    self.pos_embed.permute(1,0,2).expand(-1,B,-1),
                                    sequence, self.vocab_embed,
                                    window, seq_format,
                                    motion_bias=motion_bias)

        return out



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_seqtrack(cfg):
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)

    # Extract motion config if present
    motion_cfg = getattr(cfg.MODEL, 'MOTION', None)

    model = SEQTRACK(
        encoder,
        decoder,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        bins = cfg.MODEL.BINS,
        feature_type = cfg.MODEL.FEATURE_TYPE,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        motion_cfg = motion_cfg,
    )

    return model

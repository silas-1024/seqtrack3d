from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch


class SeqTrackActor(BaseActor):
    """ Actor for training the SeqTrack (with optional RMP motion module)"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.BINS = cfg.MODEL.BINS
        self.seq_format = cfg.DATA.SEQ_FORMAT

        # Motion config
        motion_cfg = getattr(cfg.MODEL, 'MOTION', None)
        self.enable_motion = motion_cfg is not None \
            and motion_cfg.get('ENABLE', False) \
            and motion_cfg.get('ENABLE_MOTION_ENCODER', True)
        # Default to 0.0 — only enable motion aux loss after verifying cross-attention works.
        self.motion_loss_weight = motion_cfg.get('MOTION_LOSS_WEIGHT', 0.0) if motion_cfg else 0.0
        self.history_length = motion_cfg.get('HISTORY_LENGTH', 5) if motion_cfg else 5
        self.motion_delta_type = motion_cfg.get('MOTION_DELTA_TYPE', 'raw') if motion_cfg else 'raw'

        # ---- RMP diagnostic counter ----
        self.iteration = 0

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
                   When MOTION.ENABLE=True, also contains 'historical_boxes' [B, N, 4] in xywh [0,1].
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # ---- RMP diagnostic: increment global iteration ----
        self.iteration += 1

        # forward pass
        result = self.forward_pass(data)

        if self.enable_motion:
            outputs, target_seqs, motion_aux = result
        else:
            outputs, target_seqs = result
            motion_aux = None

        # compute losses
        loss, status = self.compute_losses(outputs, target_seqs, motion_aux)

        return loss, status

    def forward_pass(self, data):
        n, b, _, _, _ = data['search_images'].shape   # n,b,c,h,w
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (n*b, c, h, w)
        search_list = search_img.split(b,dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b,dim=0)
        feature_xz = self.net(images_list=template_list+search_list, mode='encoder') # forward the encoder

        bins = self.BINS # coorinate token
        start = bins + 1 # START token
        end = bins # End token
        len_embedding = bins + 2 # number of embeddings, including the coordinate tokens and the special tokens

        # box of search region
        targets = data['search_anno'].permute(1,0,2).reshape(-1, data['search_anno'].shape[2])   # x0y0wh
        targets = box_xywh_to_xyxy(targets)   # x0y0wh --> x0y0x1y1
        targets = torch.max(targets, torch.tensor([0.]).to(targets)) # Truncate out-of-range values
        targets = torch.min(targets, torch.tensor([1.]).to(targets))

        # different formats of sequence, for ablation study
        if self.seq_format != 'corner':
            targets = box_xyxy_to_cxcywh(targets)

        # ---- Extract REAL historical boxes (pre-computed by sampler from video annotation) ----
        historical_boxes = None
        ego_compensated_prev_boxes = None
        if self.enable_motion and 'historical_boxes' in data:
            # historical_boxes is [B, N, 4] in [x,y,w,h] format, normalized [0,1]
            historical_boxes = data['historical_boxes']
            ego_compensated_prev_boxes = data.get('ego_compensated_prev_boxes', None)

        box = (targets * (bins - 1)).int() # discretize the coordinates

        if self.seq_format == 'whxy':
            box = box[:, [2, 3, 0, 1]]

        batch = box.shape[0]
        # input sequence
        input_start = torch.ones([batch, 1]).to(box) * start
        input_seqs = torch.cat([input_start, box], dim=1)
        input_seqs = input_seqs.reshape(b,n,input_seqs.shape[-1])
        input_seqs = input_seqs.flatten(1)

        # target sequence
        target_end = torch.ones([batch, 1]).to(box) * end
        target_seqs = torch.cat([box, target_end], dim=1)
        target_seqs = target_seqs.reshape(b, n, target_seqs.shape[-1])
        target_seqs = target_seqs.flatten()
        target_seqs = target_seqs.type(dtype=torch.int64)

        if self.enable_motion:
            outputs, motion_aux = self.net(
                xz=feature_xz, seq=input_seqs, mode="decoder",
                historical_boxes=historical_boxes,
                ego_compensated_prev_boxes=ego_compensated_prev_boxes,
                return_motion_aux=True
            )
            if 'motion_estimation_stats' in data:
                motion_aux['motion_estimation_stats'] = data['motion_estimation_stats']
            outputs = outputs[-1].reshape(-1, len_embedding)
            return outputs, target_seqs, motion_aux
        else:
            outputs = self.net(xz=feature_xz, seq=input_seqs, mode="decoder")
            outputs = outputs[-1].reshape(-1, len_embedding)
            return outputs, target_seqs

    def compute_losses(self, outputs, targets_seq, motion_aux=None, return_status=True):
        # Get loss
        ce_loss = self.objective['ce'](outputs, targets_seq)
        # weighted sum
        loss = self.loss_weight['ce'] * ce_loss

        # Motion auxiliary loss (default weight=0 — enable after verifying cross-attn works)
        motion_loss_val = 0.0
        reliability = motion_bias = motion_feat = deltas = None
        raw_deltas = residual_deltas = estimation_stats = None
        if self.enable_motion and motion_aux is not None and motion_aux.get('motion_feature') is not None:
            motion_loss = self.net.module.compute_motion_loss(motion_aux) \
                if hasattr(self.net, 'module') else self.net.compute_motion_loss(motion_aux)
            loss = loss + self.motion_loss_weight * motion_loss
            motion_loss_val = motion_loss.item()

            # ---- Monitoring metrics ----
            reliability = motion_aux.get('reliability')        # [B, N-1]
            motion_bias = motion_aux.get('motion_bias')        # [B, D]
            motion_feat = motion_aux.get('motion_feature')     # [B, D]
            deltas      = motion_aux.get('deltas')             # [B, N-1, 4]
            raw_deltas = motion_aux.get('raw_deltas')
            residual_deltas = motion_aux.get('residual_deltas')
            estimation_stats = motion_aux.get('motion_estimation_stats')

        outputs = outputs.softmax(-1)
        outputs = outputs[:, :self.BINS]
        value, extra_seq = outputs.topk(dim=-1, k=1)
        boxes_pred = extra_seq.squeeze(-1).reshape(-1,5)[:, 0:-1]
        boxes_target = targets_seq.reshape(-1,5)[:,0:-1]
        boxes_pred = box_cxcywh_to_xyxy(boxes_pred)
        boxes_target = box_cxcywh_to_xyxy(boxes_target)
        iou = box_iou(boxes_pred, boxes_target)[0].mean()

        if return_status:
            # status for log
            status = {
                "Loss/total": loss.item(),
                "IoU": iou.item(),
            }
            if self.enable_motion:
                status["Loss/motion"] = motion_loss_val
                gate_std = None
                # ---- Monitor reliability distribution ----
                if reliability is not None:
                    status["Motion/rel_mean"] = reliability.mean().item()
                    status["Motion/rel_std"]  = reliability.std().item()
                    status["Motion/rel_min"]  = reliability.min().item()
                    status["Motion/rel_max"]  = reliability.max().item()
                # ---- Monitor gate (sigmoid of motion_bias) ----
                if motion_bias is not None:
                    status["Motion/bias_norm"] = motion_bias.norm(dim=-1).mean().item()
                    gate_std = motion_bias.sigmoid().std().item()
                    status["Motion/gate_std"]  = gate_std
                if motion_feat is not None:
                    status["Motion/feat_norm"] = motion_feat.norm(dim=-1).mean().item()
                if deltas is not None:
                    status["Motion/delta_norm"] = deltas.norm(dim=-1).mean().item()
                if raw_deltas is not None:
                    status["Motion/raw_delta_mean"] = raw_deltas.mean().item()
                    status["Motion/raw_delta_var"] = raw_deltas.var(unbiased=False).item()
                if residual_deltas is not None:
                    status["Motion/residual_delta_mean"] = residual_deltas.mean().item()
                    status["Motion/residual_delta_var"] = residual_deltas.var(unbiased=False).item()

                if estimation_stats is not None:
                    # Columns: pair_valid, cache_hit, fallback, affine_valid,
                    # inlier_ratio, reprojection_error.
                    if estimation_stats.ndim == 3 and estimation_stats.shape[-1] == 6:
                        stats = estimation_stats
                        if stats.shape[0] == self.history_length - 1:
                            stats = stats.transpose(0, 1).contiguous()
                        valid_pairs = stats[..., 0] > 0.5
                        if valid_pairs.any():
                            valid_stats = stats[valid_pairs]
                            status["Motion/affine_cache_hit_rate"] = valid_stats[:, 1].mean().item()
                            status["Motion/fallback_identity_ratio"] = valid_stats[:, 2].mean().item()
                            status["Motion/affine_valid_rate"] = valid_stats[:, 3].mean().item()
                            successful = (valid_stats[:, 1] > 0.5) & (valid_stats[:, 3] > 0.5)
                            if successful.any():
                                success_stats = valid_stats[successful]
                                status["Motion/mean_inlier_ratio"] = success_stats[:, 4].mean().item()
                                status["Motion/mean_reprojection_error"] = success_stats[:, 5].mean().item()
                            else:
                                status["Motion/mean_inlier_ratio"] = 0.0
                                status["Motion/mean_reprojection_error"] = 0.0

                # ---- RMP diagnostics: print every 10 iters for first 3000 ----
                if self.iteration <= 3000 and self.iteration % 10 == 1:
                    gstd_str = f"{gate_std:.4f}" if gate_std is not None else "N/A"
                    dnorm_str = f"{deltas.norm(dim=-1).mean().item():.4f}" if deltas is not None else "N/A"
                    print(f"[RMP iter {self.iteration:5d}] "
                          f"gate_std={gstd_str}  delta_norm={dnorm_str}")
                # ---- Print raw deltas for first 3 batches (scale verification) ----
                if self.iteration <= 3 and deltas is not None:
                    print(f"[RMP DELTA iter {self.iteration}] "
                          f"type={self.motion_delta_type}, selected deltas[0] (scaled):\n{deltas[0]}")
                    if self.motion_delta_type == 'residual':
                        print(f"[RMP RAW DELTA iter {self.iteration}] raw deltas[0] (scaled):\n"
                              f"{raw_deltas[0]}")

                # ---- RMP guard: gate.std → 0 means V-gating is dead ----
                GATE_STD_MIN = 0.02       # only panic if truly collapsed
                WARMUP_ITERS = 3000       # enough time for Motion Encoder to warm up
                if gate_std is not None and self.iteration > WARMUP_ITERS and gate_std < GATE_STD_MIN:
                    raise RuntimeError(
                        f"\n[RMP PANIC iter {self.iteration}] gate_std={gate_std:.5f} < {GATE_STD_MIN}. "
                        f"Motion-guided V-gating has collapsed — all channels gated identically. "
                        f"Training aborted.\n"
                    )
            return loss, status
        else:
            return loss

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.objective['ce'].to(device)

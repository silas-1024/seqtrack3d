from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.seqtrack_utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.models.seqtrack import build_seqtrack
from lib.test.tracker.seqtrack_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.global_motion import transform_xywh_box
from lib.train.data.affine_cache import AffineCache
import numpy as np


class SEQTRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SEQTRACK, self).__init__(params)
        network = build_seqtrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.seq_format = self.cfg.DATA.SEQ_FORMAT
        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.bins = self.cfg.MODEL.BINS
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.hanning = torch.tensor(np.hanning(self.bins)).unsqueeze(0).cuda()
            self.hanning = self.hanning
        else:
            self.hanning = None
        self.start = self.bins + 1 # start token
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.debug = params.debug
        self.frame_id = 0

        # ---- RMP Motion Module: maintain rolling history of past boxes ----
        self.enable_motion = getattr(self.cfg.MODEL, 'MOTION', None) is not None \
                             and self.cfg.MODEL.MOTION.get('ENABLE', False) \
                             and self.cfg.MODEL.MOTION.get('ENABLE_MOTION_ENCODER', True)
        self.history_length = self.cfg.MODEL.MOTION.get('HISTORY_LENGTH', 5) if self.enable_motion else 0
        self.motion_delta_type = self.cfg.MODEL.MOTION.get(
            'MOTION_DELTA_TYPE', 'raw').lower() if self.enable_motion else 'raw'
        self.history_boxes = []  # list of [x,y,w,h] in absolute pixels, oldest first
        self.compensated_prev_boxes = []  # T_{i-1->i}(b_{i-1}), aligned to history pairs
        self.image_wh = None     # (W, H) of the video frames
        self.sequence_name = None
        self.affine_cache = AffineCache(
            self.cfg.MODEL.MOTION.get('AFFINE_CACHE_ROOT', '') if self.enable_motion else '',
            dataset_name=dataset_name,
            enabled=self.enable_motion
            and self.cfg.MODEL.MOTION.get('AFFINE_CACHE_ENABLE', False),
            fallback=self.cfg.MODEL.MOTION.get(
                'AFFINE_CACHE_FALLBACK', 'identity') if self.enable_motion else 'identity')
        self.motion_stats = {
            'pairs': 0.0, 'hits': 0.0, 'fallback': 0.0, 'success': 0.0,
            'inlier_sum': 0.0, 'reprojection_sum': 0.0,
        }
        self.use_krsu = bool(getattr(self.cfg.TEST, 'USE_KRSU', False))
        self.krsu_mode = getattr(self.cfg.TEST, 'KRSU_MODE', 'kalman_lite')
        self.krsu_center_only = bool(getattr(self.cfg.TEST, 'KRSU_CENTER_ONLY', True))
        self.krsu_use_logit_conf = bool(getattr(self.cfg.TEST, 'KRSU_USE_LOGIT_CONF', True))
        self.krsu_use_affine_valid = bool(getattr(self.cfg.TEST, 'KRSU_USE_AFFINE_VALID', True))
        self.krsu_q_scale = float(getattr(self.cfg.TEST, 'KRSU_Q_SCALE', 1.0))
        self.krsu_r_scale = float(getattr(self.cfg.TEST, 'KRSU_R_SCALE', 400.0))
        self.krsu_bootstrap_q = float(getattr(self.cfg.TEST, 'KRSU_BOOTSTRAP_Q', 400.0))
        self.krsu_target_q_floor_scale = float(
            getattr(self.cfg.TEST, 'KRSU_TARGET_Q_FLOOR_SCALE', 0.5))
        self.krsu_min_history_for_prior = int(
            getattr(self.cfg.TEST, 'KRSU_MIN_HISTORY_FOR_PRIOR', 2))
        self.krsu_affine_invalid_q_penalty = float(
            getattr(self.cfg.TEST, 'KRSU_AFFINE_INVALID_Q_PENALTY', 4.0))
        self.krsu_max_update_pixels = float(getattr(self.cfg.TEST, 'KRSU_MAX_UPDATE_PIXELS', 20.0))
        self.krsu_dynamic_clip = bool(getattr(self.cfg.TEST, 'KRSU_DYNAMIC_CLIP', False))
        self.krsu_min_r = float(getattr(self.cfg.TEST, 'KRSU_MIN_R', 1e-4))
        self.krsu_min_q = float(getattr(self.cfg.TEST, 'KRSU_MIN_Q', 1e-4))
        self.krsu_min_p = float(getattr(self.cfg.TEST, 'KRSU_MIN_P', 1e-4))
        self.krsu_P = None

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)
        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)
        print(
            "[SEQTRACK init] "
            f"USE_KRSU={self.use_krsu} "
            "VGATE_MODE=sigmoid_value_gate "
            f"motion_enabled={self.enable_motion} "
            f"motion_guided_vgate={self.cfg.MODEL.MOTION.get('ENABLE_MOTION_GUIDED_ATTN', False) if self.enable_motion else False} "
            "encoder_spatial_gate=False "
            "visual_calibration=False "
            "paraux=False "
            "residual_tanh=False")



    def initialize(self, image, info: dict):

        # get the initial templates
        z_patch_arr, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)

        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        # get the initial sequence i.e., [start]
        batch = template.shape[0]
        self.init_seq = (torch.ones([batch, 1]).to(template) * self.start).type(dtype=torch.int64)

        self.state = info['init_bbox']
        self.frame_id = 0
        self.krsu_P = None
        self.image_wh = (image.shape[1], image.shape[0])  # (W, H)
        self.sequence_name = info.get('seq_name', '')

        # Initialise motion history with the first ground-truth box
        if self.enable_motion:
            self.history_boxes = [list(info['init_bbox'])]
            self.compensated_prev_boxes = []

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        self.image_wh = (W, H)
        current_affine = None
        current_lookup = None
        previous_state = list(self.state)
        if self.enable_motion and self.motion_delta_type == 'residual':
            current_lookup = self.affine_cache.get_affine(
                self.sequence_name, self.frame_id - 1, self.frame_id)
            current_affine = current_lookup.affine
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        images_list = self.template_list + [search]

        # run the encoder
        with torch.no_grad():
            xz = self.network.forward_encoder(images_list)

        # ---- Build historical_boxes for MotionModule ----
        historical_boxes = None
        ego_compensated_prev_boxes = None
        if self.enable_motion and len(self.history_boxes) > 0:
            # Maintain exactly history_length past boxes (causal: [t-N, ..., t-1])
            # If we have fewer than history_length, pad with the oldest available box
            hist = self.history_boxes[-self.history_length:]  # most recent N
            if len(hist) < self.history_length:
                pad = [hist[0]] * (self.history_length - len(hist))
                hist = pad + hist
            hist_t = torch.tensor(hist, dtype=torch.float32, device=xz[0].device)  # [N, 4] abs pixels
            scale = torch.tensor([W, H, W, H], dtype=torch.float32, device=xz[0].device)
            historical_boxes = (hist_t / scale).clamp(0.0, 1.0)  # [N, 4] in [0,1]
            historical_boxes = historical_boxes.unsqueeze(0)      # [1, N, 4] add batch dim

            if self.motion_delta_type == 'residual':
                pair_boxes = self.compensated_prev_boxes[-(self.history_length - 1):]
                pad_count = self.history_length - len(self.history_boxes)
                pair_boxes = [hist[0]] * pad_count + pair_boxes
                comp_t = torch.tensor(
                    pair_boxes, dtype=torch.float32, device=xz[0].device)
                ego_compensated_prev_boxes = (
                    comp_t / scale).clamp(0.0, 1.0).unsqueeze(0)

        # run the decoder
        with torch.no_grad():
            out_dict = self.network.inference_decoder(xz=xz,
                                                      sequence=self.init_seq,
                                                      window=self.hanning,
                                                      seq_format=self.seq_format,
                                                      historical_boxes=historical_boxes,
                                                      ego_compensated_prev_boxes=ego_compensated_prev_boxes)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)

        # if use other formats of sequence
        if self.seq_format == 'corner':
            pred_boxes = box_xyxy_to_cxcywh(pred_boxes)
        if self.seq_format == 'whxy':
            pred_boxes = pred_boxes[:, [2, 3, 0, 1]]

        pred_boxes = pred_boxes / (self.bins-1)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        # get the final box result
        visual_box = self.map_box_back(pred_box, resize_factor)
        if self.use_krsu:
            final_box = self._apply_krsu_update(
                visual_box=visual_box,
                previous_state=previous_state,
                current_lookup=current_lookup,
                out_dict=out_dict,
                image_size=(W, H))
        else:
            final_box = visual_box
        self.state = clip_box(final_box, H, W, margin=1)

        # ---- Update motion history ----
        if self.enable_motion:
            if self.motion_delta_type == 'residual':
                compensated_prev = transform_xywh_box(
                    previous_state, current_affine, image_size=(W, H)).tolist()
                self.compensated_prev_boxes.append(compensated_prev)
                if len(self.compensated_prev_boxes) > self.history_length - 1:
                    self.compensated_prev_boxes.pop(0)

                self.motion_stats['pairs'] += 1.0
                self.motion_stats['hits'] += float(current_lookup.cache_hit)
                self.motion_stats['fallback'] += float(current_lookup.fallback_identity)
                self.motion_stats['success'] += float(
                    current_lookup.cache_hit and current_lookup.valid)
                if current_lookup.cache_hit and current_lookup.valid:
                    self.motion_stats['inlier_sum'] += current_lookup.inlier_ratio
                    self.motion_stats['reprojection_sum'] += current_lookup.reproj_error
                if self.frame_id <= 3 or self.frame_id % 100 == 0:
                    pair_count = self.motion_stats['pairs']
                    success_count = self.motion_stats['success']
                    mean_inlier = self.motion_stats['inlier_sum'] / max(success_count, 1.0)
                    mean_reproj = self.motion_stats['reprojection_sum'] / max(success_count, 1.0)
                    print(
                        f"[E2 affine frame {self.frame_id}] "
                        f"cache_hit_rate={self.motion_stats['hits'] / pair_count:.4f} "
                        f"fallback_identity_ratio={self.motion_stats['fallback'] / pair_count:.4f} "
                        f"success_rate={success_count / pair_count:.4f} "
                        f"mean_inlier_ratio={mean_inlier:.4f} "
                        f"mean_reprojection_error={mean_reproj:.4f}")
            self.history_boxes.append(list(self.state))
            if len(self.history_boxes) > self.history_length:
                self.history_boxes.pop(0)

        # update the template
        conf_score = out_dict['confidence'].sum().item() * 10  # the confidence score
        if self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, _ = sample_target(image, self.state, self.params.template_factor,
                                               output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        return {"target_bbox": self.state,
                "best_score": conf_score}

    @staticmethod
    def _box_center(box):
        return np.array([float(box[0]) + 0.5 * float(box[2]),
                         float(box[1]) + 0.5 * float(box[3])],
                        dtype=np.float32)

    def _center_confidence(self, out_dict):
        if not self.krsu_use_logit_conf or 'confidence' not in out_dict:
            return 0.5
        conf = out_dict['confidence'].detach().float().view(-1)
        if conf.numel() < 2:
            return 0.5
        if self.seq_format == 'whxy' and conf.numel() >= 4:
            center_conf = conf[[2, 3]]
        else:
            center_conf = conf[[0, 1]]
        return float(center_conf.mean().clamp(0.0, 1.0).item())

    def _residual_motion_prior(self, previous_state, current_lookup, image_size):
        W, H = image_size
        if current_lookup is not None and self.krsu_use_affine_valid:
            prev_comp = transform_xywh_box(
                previous_state, current_lookup.affine, image_size=(W, H)).tolist()
            affine_invalid = bool(
                (not current_lookup.valid) or current_lookup.fallback_identity)
        else:
            prev_comp = list(previous_state)
            affine_invalid = self.krsu_use_affine_valid

        prev_comp_center = self._box_center(prev_comp)
        residuals = []
        pair_count = min(len(self.history_boxes) - 1, len(self.compensated_prev_boxes))
        if self.enable_motion and self.motion_delta_type == 'residual' and pair_count > 0:
            hist_tail = self.history_boxes[-pair_count:]
            comp_tail = self.compensated_prev_boxes[-pair_count:]
            for hist_box, comp_box in zip(hist_tail, comp_tail):
                residuals.append(self._box_center(hist_box) - self._box_center(comp_box))

        if residuals:
            residuals = np.asarray(residuals, dtype=np.float32)
            weights = np.linspace(1.0, 2.0, num=len(residuals), dtype=np.float32)
            v_res = np.average(residuals, axis=0, weights=weights)
            if len(residuals) > 1:
                q = float(np.var(residuals[:, 0]) + np.var(residuals[:, 1]))
            else:
                q = self.krsu_min_q
        else:
            v_res = np.zeros(2, dtype=np.float32)
            q = self.krsu_min_q

        if affine_invalid:
            q += max(1.0, q) * self.krsu_affine_invalid_q_penalty

        prior_center = prev_comp_center + v_res
        return (prior_center, v_res.astype(np.float32), float(q), affine_invalid,
                prev_comp_center, residuals, pair_count)

    def _stabilize_covariance(self, P):
        P = 0.5 * (P + P.T)
        diag_idx = np.diag_indices_from(P)
        P[diag_idx] = np.maximum(P[diag_idx], self.krsu_min_p)
        return P.astype(np.float32)

    def _apply_krsu_update(self, visual_box, previous_state, current_lookup,
                           out_dict, image_size):
        if self.krsu_mode != 'kalman_lite' or not self.krsu_center_only:
            raise ValueError(
                'Only TEST.KRSU_MODE=kalman_lite with KRSU_CENTER_ONLY=True is supported')

        z = self._box_center(visual_box).astype(np.float32)
        prior_center, v_res, q_raw, affine_invalid, prev_comp_center, residuals, pair_count = self._residual_motion_prior(
            previous_state, current_lookup, image_size)
        x_prior = np.array([prior_center[0], prior_center[1], v_res[0], v_res[1]],
                           dtype=np.float32)

        target_q_floor = (
            self.krsu_target_q_floor_scale *
            max(float(visual_box[2]), float(visual_box[3]))) ** 2
        if pair_count < self.krsu_min_history_for_prior:
            q_base = max(q_raw, self.krsu_bootstrap_q, target_q_floor)
            prior_ready = False
        else:
            q_base = max(q_raw, target_q_floor)
            prior_ready = True
        q = max(q_base * self.krsu_q_scale, self.krsu_min_q)
        conf = self._center_confidence(out_dict)
        r_conf = 1.0 - conf + 1e-6
        r = max(r_conf * self.krsu_r_scale, self.krsu_min_r)

        F_mat = np.array([[1.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        H_mat = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        Q = np.eye(4, dtype=np.float32) * q
        R = np.eye(2, dtype=np.float32) * r
        if self.krsu_P is None:
            P_prior = Q.copy()
        else:
            P_prior = F_mat.dot(self.krsu_P).dot(F_mat.T) + Q
        P_prior = self._stabilize_covariance(P_prior)

        innovation = z - H_mat.dot(x_prior)
        S = H_mat.dot(P_prior).dot(H_mat.T) + R
        K = P_prior.dot(H_mat.T).dot(np.linalg.inv(S))
        x_post = x_prior + K.dot(innovation)
        I = np.eye(4, dtype=np.float32)
        P_post = (I - K.dot(H_mat)).dot(P_prior)
        self.krsu_P = self._stabilize_covariance(P_post)

        update = x_post[:2] - z
        update_norm = float(np.linalg.norm(update))
        max_update = self.krsu_max_update_pixels
        if self.krsu_dynamic_clip:
            max_update = min(max_update, max(4.0, 2.0 * max(float(visual_box[2]), float(visual_box[3]))))
        clipped_update = update
        if update_norm > max_update > 0:
            clipped_update = update * (max_update / update_norm)
        clipped_norm = float(np.linalg.norm(clipped_update))
        final_center = z + clipped_update

        if self.frame_id <= 3 or self.frame_id % 100 == 0:
            qr_ratio = q / max(r, self.krsu_min_r)
            if isinstance(residuals, np.ndarray) and residuals.size:
                residual_dbg = residuals[-min(3, len(residuals)):].tolist()
            else:
                residual_dbg = []
            print(
                f"[E6a-KRSU frame {self.frame_id}] "
                f"z=({z[0]:.2f},{z[1]:.2f}) "
                f"prev_comp=({prev_comp_center[0]:.2f},{prev_comp_center[1]:.2f}) "
                f"prior=({prior_center[0]:.2f},{prior_center[1]:.2f}) "
                f"post=({x_post[0]:.2f},{x_post[1]:.2f}) "
                f"final=({final_center[0]:.2f},{final_center[1]:.2f}) "
                f"v_res=({v_res[0]:.2f},{v_res[1]:.2f}) "
                f"q_raw_px2={q_raw:.6f} q_base_px2={q_base:.6f} Q={q:.6f} "
                f"r_conf={r_conf:.6f} R_px2={r:.6f} Q_R_ratio={qr_ratio:.6f} "
                f"K_diag=({K[0,0]:.4f},{K[1,1]:.4f}) "
                f"innovation_norm={np.linalg.norm(innovation):.2f} "
                f"update_norm={update_norm:.2f}->{clipped_norm:.2f} "
                f"max_update={max_update:.2f} dynamic_clip={self.krsu_dynamic_clip} "
                f"coord_conf={conf:.4f} affine_invalid={affine_invalid} "
                f"pair_count={pair_count} prior_ready={prior_ready}")
            print(
                f"[E6a-KRSU residual frame {self.frame_id}] "
                "residual_delta=center(box_i)-center(affine_compensated_box_i-1_to_i) "
                f"recent={residual_dbg} "
                f"K={np.array2string(K, precision=4, suppress_small=True)}")

        return [float(final_center[0] - 0.5 * visual_box[2]),
                float(final_center[1] - 0.5 * visual_box[3]),
                float(visual_box[2]),
                float(visual_box[3])]

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return SEQTRACK

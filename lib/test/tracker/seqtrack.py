from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.seqtrack_utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.models.seqtrack import build_seqtrack
from lib.test.tracker.seqtrack_utils import Preprocessor
from lib.utils.box_ops import clip_box
import numpy as np
from lib.test.tracker.motion_prior import MotionPrior
import math


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

        # ---- Motion Prior: MLP-based search-crop center SOFT guidance ----
        # When enabled, a tiny learnable MLP predicts the next-frame center
        # OFFSET (Δx, Δy) from historical observed centers.  The predicted
        # center is SOFTLY blended with the fallback center to gently guide
        # the search crop — the model's encoder/decoder/box-head remain unchanged.
        # No post-hoc correction is applied.
        use_motion_search = getattr(self.cfg.TEST, 'USE_MOTION_SEARCH_CENTER', False)
        motion_model = getattr(self.cfg.TEST, 'MOTION_MODEL', 'constant_velocity')
        motion_alpha = getattr(self.cfg.TEST, 'MOTION_ALPHA', 0.1)
        motion_clip = getattr(self.cfg.TEST, 'MOTION_CLIP', 100.0)
        motion_warmup = getattr(self.cfg.TEST, 'MOTION_WARMUP_FRAMES', 2)
        motion_history_len = getattr(self.cfg.TEST, 'MOTION_HISTORY_LEN', 4)
        motion_hidden_dim = getattr(self.cfg.TEST, 'MOTION_HIDDEN_DIM', 32)
        motion_conf_thresh = getattr(self.cfg.TEST, 'MOTION_CONF_THRESHOLD', 0.5)
        self.motion_prior = MotionPrior(
            use=use_motion_search,
            model=motion_model,
            alpha=motion_alpha,
            clip=motion_clip,
            warmup_frames=motion_warmup,
            history_len=motion_history_len,
            hidden_dim=motion_hidden_dim,
            conf_threshold=motion_conf_thresh,
        )
        # Move MLP (if any) to GPU
        self.motion_prior.to('cuda')
        self.motion_prior.eval()
        # Track previous frame's confidence for motion prior attenuation
        self._prev_conf_score = None
        if use_motion_search:
            print(f"[MotionSearch] Enabled: model={motion_model}, "
                  f"alpha={motion_alpha}, clip={motion_clip:.0f}px, "
                  f"warmup={motion_warmup}frames, "
                  f"history_len={motion_history_len}, hidden_dim={motion_hidden_dim}, "
                  f"conf_thresh={motion_conf_thresh}")
        else:
            print("[MotionSearch] Disabled (baseline mode)")
        # ----------------------------------------------------

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

        # Initialize motion prior with the first-frame center
        # search_crop_center stores the actual center used for cropping;
        # needed by map_box_back() to correctly map crop coords → image coords.
        init_cx = self.state[0] + 0.5 * self.state[2]
        init_cy = self.state[1] + 0.5 * self.state[3]
        self.search_crop_center = (init_cx, init_cy)  # used by map_box_back
        self._motion_info = {}
        if self.motion_prior.use:
            self.motion_prior.reset()
            self.motion_prior.update(init_cx, init_cy)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        # ===== Motion Prior: compute search crop center via SOFT blending =====
        #
        # Three centers (all in original image coordinates):
        #   observed_center  : model's raw output center from PREVIOUS frame
        #   predicted_center : motion model forecast for THIS frame
        #   search_center    : soft_blend(observed, predicted) → used for crop
        #
        # History update rule:
        #   - At start of frame t: update history with frame t-1's OBSERVED center
        #   - Then predict frame t's center from history
        #   - Frame t's OBSERVED center is NOT added until start of frame t+1
        #   - This avoids "update then predict in same frame" confusion
        # ==============================================================

        # observed_center: model's output center from the PREVIOUS frame
        observed_cx = self.state[0] + 0.5 * self.state[2]
        observed_cy = self.state[1] + 0.5 * self.state[3]

        # Feed PREVIOUS frame's observed center into motion history
        # (NOT motion-corrected center — avoids compounding error)
        if self.motion_prior.use:
            self.motion_prior.update(observed_cx, observed_cy)

        # Get the search crop center via SOFT blending:
        #   search_center = (1 - eff_alpha) * observed + eff_alpha * predicted
        #   - If disabled: eff_alpha = 0 → search_center = observed (baseline)
        #   - If enabled + warmup passed: eff_alpha = alpha * conf_weight * dist_weight
        #   - If enabled but warmup not passed: eff_alpha = 0 (fallback)
        #   - Uses PREVIOUS frame's confidence for attenuation
        search_cx, search_cy, eff_alpha, self._motion_info = \
            self.motion_prior.get_search_center(
                fallback_cx=observed_cx,
                fallback_cy=observed_cy,
                conf_score=self._prev_conf_score,
                img_W=W, img_H=H,
            )

        # Store for map_box_back() — MUST reflect the ACTUAL center used for cropping.
        # This ensures geometric consistency: crop coordinate system == map_back reference.
        self.search_crop_center = (search_cx, search_cy)

        # Construct a temporary bounding box centered at search_crop_center
        # (keep the same width/height as the previous frame's state)
        search_bb = [
            search_cx - 0.5 * self.state[2],
            search_cy - 0.5 * self.state[3],
            self.state[2],
            self.state[3],
        ]
        # ======================================================

        x_patch_arr, resize_factor = sample_target(image, search_bb, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        images_list = self.template_list + [search]

        # run the encoder
        with torch.no_grad():
            xz = self.network.forward_encoder(images_list)

        # run the decoder
        with torch.no_grad():
            out_dict = self.network.inference_decoder(xz=xz,
                                                      sequence=self.init_seq,
                                                      window=self.hanning,
                                                      seq_format=self.seq_format)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)

        # if use other formats of sequence
        if self.seq_format == 'corner':
            pred_boxes = box_xyxy_to_cxcywh(pred_boxes)
        if self.seq_format == 'whxy':
            pred_boxes = pred_boxes[:, [2, 3, 0, 1]]

        pred_boxes = pred_boxes / (self.bins-1)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        # get the final box result — map_box_back uses self.search_crop_center
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=1)

        # update the template
        conf_score = out_dict['confidence'].sum().item() * 10  # the confidence score
        # Store for next frame's motion prior attenuation
        self._prev_conf_score = conf_score

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
                "best_score": conf_score,
                "motion_prior": self._motion_info}

    def map_box_back(self, pred_box: list, resize_factor: float):
        """
        Map predicted box from crop coordinates back to original image coordinates.

        Geometry guarantee:
          - self.search_crop_center = the ACTUAL center used by sample_target()
            for cropping (i.e., the center of search_bb passed to sample_target).
          - This center is the coordinate system origin for mapping back.
          - No hand-reconstructed center is used; always read from
            self.search_crop_center which is set atomically in track().
          - This ensures the crop coordinate system and the back-mapping
            coordinate system are strictly identical.
        """
        cx_prev, cy_prev = self.search_crop_center
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.search_crop_center
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return SEQTRACK

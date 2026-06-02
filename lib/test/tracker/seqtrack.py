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

        # ---- Motion Prior: search-crop center prediction ----
        # When enabled, the motion model predicts where the target will be
        # in the next frame, and shifts the search crop BEFORE feature extraction.
        # The model's encoder/decoder/box-head remain unchanged.
        # No post-hoc correction is applied to the output box.
        use_motion_search = getattr(self.cfg.TEST, 'USE_MOTION_SEARCH_CENTER', False)
        motion_model = getattr(self.cfg.TEST, 'MOTION_MODEL', 'constant_velocity')
        motion_clip = getattr(self.cfg.TEST, 'MOTION_CLIP', 100.0)
        motion_warmup = getattr(self.cfg.TEST, 'MOTION_WARMUP_FRAMES', 2)
        self.motion_prior = MotionPrior(
            use=use_motion_search,
            model=motion_model,
            clip=motion_clip,
            warmup_frames=motion_warmup,
        )
        if use_motion_search:
            print(f"[MotionSearch] Enabled: model={motion_model}, "
                  f"clip={motion_clip:.0f}px, warmup={motion_warmup}frames")
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

        # ===== Motion Prior: compute search crop center =====
        # observed_center: model's output center from the previous frame
        observed_cx = self.state[0] + 0.5 * self.state[2]
        observed_cy = self.state[1] + 0.5 * self.state[3]

        # Feed observed center into motion history (NOT motion-corrected center)
        if self.motion_prior.use:
            self.motion_prior.update(observed_cx, observed_cy)

        # Get the search crop center from motion prior
        #   - If disabled: uses observed_center (baseline behavior)
        #   - If enabled + warmup passed: uses predicted center (clipped)
        #   - If enabled but warmup not passed: uses observed_center
        search_cx, search_cy, motion_weight, self._motion_info = \
            self.motion_prior.get_search_center(
                fallback_cx=observed_cx,
                fallback_cy=observed_cy,
                img_W=W, img_H=H,
            )

        # Store for map_box_back() — must reflect the ACTUAL center used for cropping
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
        # Use the ACTUAL search crop center (may differ from self.state center
        # when motion prior is enabled), not the previous frame's box center.
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

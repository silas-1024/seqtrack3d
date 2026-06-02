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

        # ---- Motion Prior ----
        # Read motion prior settings from cfg (with safe defaults)
        use_motion_prior = getattr(self.cfg.TEST, 'MOTION_PRIOR_USE', False)
        lambda_motion = getattr(self.cfg.TEST, 'MOTION_PRIOR_LAMBDA', 0.3)
        sigma_motion = getattr(self.cfg.TEST, 'MOTION_PRIOR_SIGMA', 20.0)
        self.motion_prior = MotionPrior(
            use_motion_prior=use_motion_prior,
            lambda_motion=lambda_motion,
            sigma=sigma_motion,
        )
        if use_motion_prior:
            print(f"[MotionPrior] Enabled: lambda={lambda_motion}, sigma={sigma_motion}")
        else:
            print("[MotionPrior] Disabled (baseline mode)")
        # -----------------------

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
        if self.motion_prior.use_motion_prior:
            self.motion_prior.reset()
            init_cx = self.state[0] + 0.5 * self.state[2]
            init_cy = self.state[1] + 0.5 * self.state[3]
            self.motion_prior.update(init_cx, init_cy)
            self._motion_info = {}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
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

        # ===== Motion Prior: soft center correction =====
        # pred_box is in original-image crop coordinates:
        #   cx = half_side maps to previous center in original image
        #   half_side = 0.5 * search_size / resize_factor
        self._motion_info = {
            'predicted_center': None,
            'prior_score': 0.0,
            'final_score': 0.0,
            'use_motion_prior': False,
        }
        if self.motion_prior.use_motion_prior:
            # 1. Compute the model-predicted center in original image coords
            cx_prev_img = self.state[0] + 0.5 * self.state[2]
            cy_prev_img = self.state[1] + 0.5 * self.state[3]
            half_side = 0.5 * self.params.search_size / resize_factor
            pred_cx_img = pred_box[0] + (cx_prev_img - half_side)
            pred_cy_img = pred_box[1] + (cy_prev_img - half_side)

            # 2. Predict next center from motion model
            c_hat = self.motion_prior.predict()
            # Alternative: c_hat = self.motion_prior.predict_with_acceleration()

            # 3. Apply Gaussian-weighted soft blend (in original image coords)
            corrected_cx, corrected_cy, prior_weight, motion_info = \
                self.motion_prior.apply(pred_cx_img, pred_cy_img, c_hat)
            self._motion_info = motion_info

            if prior_weight > 0.0:
                # Convert corrected center back to crop coordinates
                pred_box[0] = corrected_cx - (cx_prev_img - half_side)
                pred_box[1] = corrected_cy - (cy_prev_img - half_side)

            # 4. Update motion history (after this frame, for next prediction)
            # Note: we feed the ORIGINAL model prediction (not corrected)
            # to avoid compounding errors from the prior itself.
            # If you prefer using the corrected center, change here.
            self.motion_prior.update(pred_cx_img, pred_cy_img)
        # ====================================================

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=1)

        # update the template
        conf_score = out_dict['confidence'].sum().item() * 10  # the confidence score

        # ---- Append motion prior info to conf_score for logging ----
        if self.motion_prior.use_motion_prior:
            self._motion_info['final_score'] = conf_score + self._motion_info.get('prior_score', 0.0)
        # ------------------------------------------------------------

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

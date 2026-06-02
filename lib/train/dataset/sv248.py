import os
import random
from collections import OrderedDict

import numpy as np
import pandas
import torch

from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader


class SV248(BaseVideoDataset):
    """SV248 dataset loader.

    Expected layout:
    root/
      <seq_name>/
        sequences/000001.tiff
        groundTruth.rect or Groundtruth.rect
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None,
                 multi_modal_vision=False, multi_modal_language=False):
        root = env_settings().sv248_dir if root is None else root
        super().__init__('SV248', root, image_loader)

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.class_list = [f for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))]
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}
        self.seq_per_class = self._build_class_list()
        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None and vid_ids is not None:
            raise ValueError('Cannot set both split_name and vid_ids.')

        return [seq_name for seq_name in sorted(os.listdir(self.root))
                if os.path.isdir(os.path.join(self.root, seq_name))]

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        if self.multi_modal_language:
            return 'sv248_lang'
        return 'sv248'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        candidates = [
            os.path.join(seq_path, "groundTruth.rect"),
            os.path.join(seq_path, "Groundtruth.rect"),
            os.path.join(seq_path, "groundtruth.rect"),
        ]
        bb_anno_file = None
        for cand in candidates:
            if os.path.exists(cand):
                bb_anno_file = cand
                break
        if bb_anno_file is None:
            raise FileNotFoundError(
                "No annotation file found under {}. Tried: {}".format(seq_path, ", ".join(candidates))
            )

        gt = pandas.read_csv(
            bb_anno_file,
            delimiter=',',
            header=None,
            dtype=np.float32,
            na_filter=False,
        ).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name, 'sequences')

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(os.path.join(self.root, self.sequence_list[seq_id]))

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.byte() & valid.byte()
        output = {'bbox': bbox, 'valid': valid, 'visible': visible}
        if self.multi_modal_language:
            output['nlp'] = self._read_nlp(os.path.join(self.root, self.sequence_list[seq_id]))
        return output

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:06}.tiff'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        frame = self.image_loader(self._get_frame_path(seq_path, frame_id))
        if self.multi_modal_vision:
            frame = np.concatenate((frame, frame), axis=-1)
        return frame

    def _get_class(self, seq_path):
        return os.path.basename(os.path.dirname(seq_path))

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        return self._get_class(seq_path)

    def _read_nlp(self, seq_path):
        nlp_file = os.path.join(seq_path, "nlp.txt")
        nlp = pandas.read_csv(nlp_file, dtype=str, header=None).values
        return nlp[0][0]

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                anno_frames[key] = [value for _ in frame_ids]
            else:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({
            'object_class_name': obj_class,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None,
        })

        return frame_list, anno_frames, object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return anno_frames

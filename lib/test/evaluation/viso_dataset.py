import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

import os


class viso_dataset(BaseDataset):
    """
    Satellite Video Dataset
    """

    def __init__(self):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = os.path.join(self.env_settings.viso_path)

        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/{}.txt'.format(self.base_path, sequence_name, 'groundtruth_skip')
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'img')
        frame_list = [frame for frame in os.listdir(frames_path)]
        frame_list.sort(key=lambda f: str(f))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        ground_truth_rect = np.array(ground_truth_rect)

        return Sequence(sequence_name, frames_list, 'viso_dataset', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = os.listdir(os.path.join(self.base_path))
        exclude_strings = ["027_57_23_232", "027_61_23_232","027_81_23_232","027_62_23_232","027_83_23_232"]
        sequence_list = [
            item for item in os.listdir(os.path.join(self.base_path))
            if not any(s in item for s in exclude_strings)
        ]
        return sequence_list

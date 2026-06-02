import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

import os


class sv248s_test_dataset(BaseDataset):
    """
    Satellite Video Dataset
    """

    def __init__(self):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = os.path.join(self.env_settings.sv248s_test_path)

        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/{}.rect'.format(self.base_path, sequence_name, 'groundTruth')

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'sequences')
        frame_list = [frame for frame in os.listdir(frames_path)]
        frame_list.sort(key=lambda f: str(f))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        ground_truth_rect = np.array(ground_truth_rect)

        return Sequence(sequence_name, frames_list, 'sv248s_test_dataset', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = os.listdir(os.path.join(self.base_path))
        return sequence_list

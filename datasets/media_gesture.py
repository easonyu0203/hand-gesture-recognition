from torch.utils.data import Dataset
import torch
import os
import numpy as np


class MediaGestureDataset(Dataset):
    """
    This dataset use media holistic hand landmark as feature with a target gesture
    in data_dir_path, for each kind of gesture should have a [gesture_name].npy file,
    in .npy file should be shape [num_record, 21(hand landmarks), 3(x,y,z)]
    """
    feature_cnt: int
    class_cnt: int
    label_idx_2_name: dict[int, str]
    label_name_2_idx: dict[str, int]

    def __init__(self, data_dir_path="./data", transform=None):
        """
        This dataset use media holistic hand landmark as feature with a target gesture
        in data_dir_path, for each kind of gesture should have a [gesture_name].npy file,
        in .npy file should be shape [num_record, 21(hand landmarks), 3(x,y,z)]
        """
        self.transform = transform
        feature_dict = dict()
        label_dict = dict()
        self.data_size_dict = dict()
        self.label_name_2_idx = dict()
        self.label_idx_2_name = dict()
        self.data_size = 0
        # Get a list of all files in the directory
        files = os.listdir(data_dir_path)

        # Filter out any non-files (directories, symlinks, etc.)
        files = [f for f in files if os.path.isfile(os.path.join(data_dir_path, f))]

        # load data form files
        for idx, filename in enumerate(files):
            filepath = os.path.join(data_dir_path, filename)
            base_name = os.path.splitext(filename)[0]
            feature = np.load(filepath)
            feature_dict[base_name] = feature
            label_dict[base_name] = torch.ones((len(feature), 1)) * idx
            self.label_name_2_idx[base_name] = idx
            self.label_idx_2_name[idx] = base_name
            self.data_size_dict[base_name] = len(feature_dict[base_name])

        self.features = torch.cat(list(torch.from_numpy(f).type(torch.float) for f in feature_dict.values()), 0)
        self.labels = torch.cat(list(l for l in label_dict.values()), 0)

        # calculate data datasize
        self.data_size = sum(f.shape[0] for f in feature_dict.values())
        self.class_cnt = len(self.label_name_2_idx)
        self.feature_cnt = self.features.shape[1] * 2

    def target_to_one_hot(self, target):
        """
        transform target(idx) to on hot vector
        :param target: idx
        :return: one hot encode(idx)
        """
        return torch.zeros(self.class_cnt, dtype=torch.float).scatter_(0, target.type(torch.int64), 1)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        """
        :param idx: index
        :return: (tensor[21,3], on hot vector: tensor[class_cnt])
        """
        # target
        target = self.labels[idx]
        target = self.target_to_one_hot(target)
        # feature
        feature = self.features[idx]
        if self.transform:
            feature = self.transform(feature)
        return feature, target

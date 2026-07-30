import os
import glob
import torch
import pickle
from torch.utils.data import Dataset

#Used for creating a dataset for both U-net pipelines

class USGDataset(Dataset):
    def __init__(self, processed_folder, param, split='train'):
        split_dir = os.path.join(processed_folder, split)
        self.file_paths = glob.glob(os.path.join(split_dir, '*.pt'))
        self.param = param

        stats_path = os.path.join(processed_folder, 'norm_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        self.mean = stats['mean']
        self.std = stats['std']

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Normalizes data using mean and std. Creates an input frame
        :param idx: data loader idx
        :return: Frame with zeroed columns, normalized ground truth
        """
        Y = torch.load(self.file_paths[idx])

        Y = (Y - self.mean) / (self.std + 1e-8)

        X = torch.zeros_like(Y)
        X[:, :, ::self.param] = Y[:, :, ::self.param]

        return X, Y
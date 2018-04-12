import numpy as np
import torch
from torch.utils.data import DataLoader


# create concatenated dataset -- based on https://github.com/pytorch/tnt/blob/master/torchnet/dataset/concatdataset.py
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, shuffle=False):
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets must not be empty'
        self.cum_sizes = np.cumsum([len(x) for x in self.datasets])
        self.index_mapping = None if not shuffle else np.random.permutation(self.cum_sizes[-1])
        super(ConcatDataset, self).__init__()

    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, idx):
        idx = idx if self.index_mapping is None else self.index_mapping[idx]
        dataset_index = self.cum_sizes.searchsorted(idx, 'right')

        if dataset_index == 0:
            dataset_idx = idx
        else:
            dataset_idx = idx - self.cum_sizes[dataset_index - 1]

        return self.datasets[dataset_index][dataset_idx]

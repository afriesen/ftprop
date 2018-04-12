import numpy as np
import torch
from torch.utils.data import DataLoader


# create validation dataset -- from https://gist.github.com/t-vi/9f6118ff84867e89f3348707c7a1271f
class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length, index_mapping=None):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert index_mapping is None or len(parent_ds) == len(index_mapping), 'indices must map to parent dataset'
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        self.index_mapping = index_mapping
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        idx = i + self.offset
        if self.index_mapping is not None:
            idx = self.index_mapping[idx]
        return self.parent_ds[idx]


def validation_split(dataset, val_share=0.1, shuffle_data=False):
    """
       Split a (training and validation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
          shuffle_data: either a boolean indicating whether the data should be randomly permuted before splitting, or a
                        random permutation specifying how the data indices should be remapped
       returns: input dataset split into test_ds, val_ds

       """
    val_offset = round(len(dataset) * (1 - val_share))

    index_mapping = None
    if isinstance(shuffle_data, bool):
        if shuffle_data:
            index_mapping = np.random.permutation(len(dataset))
    else:
        assert len(shuffle_data) == len(dataset), 'index_mapping and dataset must have the same length'
        index_mapping = shuffle_data

    ds1 = PartialDataset(dataset, 0, val_offset, index_mapping=index_mapping)
    ds2 = PartialDataset(dataset, val_offset, len(dataset) - val_offset, index_mapping=index_mapping)

    return ds1, ds2

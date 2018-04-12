import torch.utils.data as data


class DatasetIndexingWrapper(data.Dataset):
    """
    Wrap a dataset so that the returned dataset entries contain the index of the item along with the item and its label.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, target, index

    def __len__(self):
        return len(self.dataset)

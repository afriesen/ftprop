import os
import logging
from functools import partial
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from util.partialdataset import validation_split
from util.concatdataset import ConcatDataset
from util.datasetindexingwrapper import DatasetIndexingWrapper


def create_datasets(ds_name, batch_size, test_batch_size, do_aug, no_val_set, data_root,
                    use_cuda, seed, num_workers, dbg_ds_size, allow_download):

    ds_name = ds_name.lower()

    # ----- set up specified dataset -----
    if ds_name == 'mnist':
        mean_std = ((0.1307,), (0.3081,))
        ds = datasets.MNIST
        create_ds_func = partial(create_mnist_cifar_datasets, ds=ds, download=allow_download, val_pct=1.0/6.0,
                                 data_dir='data/' + ds_name)
        num_classes = 10
    elif ds_name == 'cifar10':
        mean_std = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        ds = datasets.CIFAR10
        create_ds_func = partial(create_mnist_cifar_datasets, ds=ds, download=allow_download, val_pct=0.2,
                                 data_dir='data/' + ds_name)
        num_classes = 10
    elif ds_name == 'cifar100':
        # mean_std = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        mean_std = ((129.30416561 / 255, 124.0699627 / 255, 112.43405006 / 255),
                    (68.1702429 / 255, 65.39180804 / 255, 70.41837019 / 255))
        ds = datasets.CIFAR10
        create_ds_func = partial(create_mnist_cifar_datasets, ds=ds, download=allow_download, val_pct=0.2,
                                 data_dir='data/' + ds_name)
        num_classes = 100
    elif ds_name == 'svhn':
        mean_std = ((0.4309, 0.4302, 0.4463), (0.1965, 0.1983, 0.1994))
        create_ds_func = partial(create_svhn_datasets, download=allow_download, val_pct=0.1, data_dir='data/' + ds_name)
        num_classes = 10
    elif ds_name == 'imagenet':
        mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        root_dir = '/media/ssdpro/data/imagenet/ilsvrc14_256_ims' if not data_root else data_root
        create_ds_func = partial(create_imagenet_datasets, data_root_dir=root_dir)
        num_classes = 1000
    else:
        raise NotImplementedError("'{}' dataset is not supported".format(ds_name))

    train_loader, val_loader, test_loader = \
        create_ds_func(batch_size, use_cuda, seed, mean_std=mean_std, do_aug=do_aug,
                       create_val=not no_val_set, num_workers=num_workers, test_batch_size=test_batch_size,
                       dbg_ds_size=dbg_ds_size)

    return train_loader, val_loader, test_loader, num_classes


def create_mnist_cifar_datasets(batch_size, use_cuda, seed, ds=None, mean_std=None, val_pct=0.1, data_dir='',
                                download=False, test_batch_size=None, do_aug=False, create_val=True,
                                num_workers=2, dbg_ds_size=0):
    kwargs = {}
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        kwargs = {'num_workers': num_workers, 'pin_memory': True}

    if mean_std is None:
        train_set = ds(data_dir, train=True, download=download, transform=transforms.ToTensor())
        mu = train_set.train_data.mean(axis=(0, 1, 2)) / 255
        sig = train_set.train_data.std(axis=(0, 1, 2)) / 255
        mean_std = (mu, sig)
        print('train_data shape:', train_set.train_data.size(), 'mean:', mean_std[0], 'std:', mean_std[1])
        del train_set

    train_transforms = [transforms.ToTensor(),
                        transforms.Normalize(mean_std[0], mean_std[1])]
    test_transforms = deepcopy(train_transforms)
    if do_aug:
        train_transforms = [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip()] + train_transforms

    train_ds = ds(data_dir, train=True, download=download, transform=transforms.Compose(train_transforms))

    if create_val:
        total_len = len(train_ds)
        val_ds = ds(data_dir, train=True, download=False, transform=transforms.Compose(test_transforms))
        train_ds, _ = validation_split(train_ds, val_share=val_pct)
        _, val_ds = validation_split(val_ds, val_share=val_pct)
        assert len(train_ds) + len(val_ds) == total_len
    else:
        val_ds = None
    test_ds = ds(data_dir, train=False, download=download, transform=transforms.Compose(test_transforms))

    if test_batch_size is None or test_batch_size == 0:
        test_batch_size = batch_size

    if dbg_ds_size > 0:
        train_ds, _ = validation_split(train_ds, val_share=(1.0 - float(dbg_ds_size) / len(train_ds)))
        logging.debug('DEBUG: setting train dataset size = {}'.format(len(train_ds)))

    train_ds = DatasetIndexingWrapper(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, **kwargs)  # TODO: REMOVE THIS
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False, **kwargs) if create_val else None
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def create_svhn_datasets(batch_size, use_cuda, seed, mean_std=None, val_pct=0.1, data_dir='', download=False,
                         test_batch_size=None, do_aug=False, create_val=True, num_workers=2, dbg_ds_size=0):
    kwargs = {}
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        kwargs = {'num_workers': num_workers, 'pin_memory': True}

    if mean_std is None:
        logging.info('computing mean and std of SVHN dataset -- pre-specify these to improve performance')
        train_data = datasets.SVHN(data_dir, split='train', download=download, transform=transforms.ToTensor())
        extra_data = datasets.SVHN(data_dir, split='extra', download=download, transform=transforms.ToTensor())
        train_set = ConcatDataset([train_data, extra_data], shuffle=False)
        mean, std = compute_mean_std_per_channel(train_set)
        mean_std = (mean, std)
        print('train_data shape: {}x{}'.format(len(train_set), train_set[0][0].size()), 'mean:', mean, 'std:', std)
        del train_set, extra_data

    base_transforms = [transforms.Resize((40, 40)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean_std[0], mean_std[1])]

    # if do_aug:
    #     train_transforms = [transforms.RandomCrop(32, padding=4),
    #                         transforms.RandomHorizontalFlip()] + deepcopy(base_transforms)
    # assert not do_aug, 'data augmentation not supported for SVHN'
    train_transforms = transforms.Compose(deepcopy(base_transforms))
    test_transforms = transforms.Compose(base_transforms)

    train_ds = ConcatDataset([datasets.SVHN(data_dir, split='train', download=download, transform=train_transforms),
                              datasets.SVHN(data_dir, split='extra', download=download, transform=train_transforms)])

    if create_val:
        val_ds = ConcatDataset([datasets.SVHN(data_dir, split='train', download=False, transform=test_transforms),
                                datasets.SVHN(data_dir, split='extra', download=False, transform=test_transforms)])
        train_ds, _ = validation_split(train_ds, val_share=val_pct, shuffle_data=True)
        _, val_ds   = validation_split(val_ds,   val_share=val_pct, shuffle_data=train_ds.index_mapping)
    else:
        val_ds = None

    test_ds = datasets.SVHN(data_dir, split='test', download=download, transform=test_transforms)

    if test_batch_size is None or test_batch_size == 0:
        test_batch_size = batch_size

    if dbg_ds_size > 0:
        train_ds, _ = validation_split(train_ds, val_share=(1.0 - float(dbg_ds_size) / len(train_ds)))
        logging.debug('DEBUG: setting train dataset size = {}'.format(len(train_ds)))

    train_ds = DatasetIndexingWrapper(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False, **kwargs) if create_val else None
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def create_imagenet_datasets(batch_size, use_cuda, seed, mean_std=None, data_root_dir='', test_batch_size=None,
                             create_val=True, do_aug=True, num_workers=2, dbg_ds_size=0):
    kwargs = {}
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        kwargs = {'num_workers': num_workers, 'pin_memory': True}

    assert mean_std is not None, 'cannot compute mean_std on imagenet, must specify it beforehand'

    # note: this assumes that ImageNet images have already been resized to have short edge = 256 when loaded

    base_transforms = [transforms.ToTensor(),
                       transforms.Normalize(mean_std[0], mean_std[1])]

    # train transforms = random crop to 224, horizontal flip, normalize mean and stddev
    train_transforms = [] if not do_aug else [transforms.RandomCrop(224, 0),
                                              transforms.RandomHorizontalFlip()]
    train_transforms = train_transforms + deepcopy(base_transforms)

    # test transforms = center crop, normalize mean and stddev
    test_transforms = [transforms.CenterCrop(224)] + base_transforms

    train_dir = os.path.join(data_root_dir, 'train')
    val_dir = os.path.join(data_root_dir, 'val')
    train_ds = datasets.ImageFolder(train_dir, transform=transforms.Compose(train_transforms))

    if create_val:
        val_pct = 0.1
        val_ds = datasets.ImageFolder(train_dir, transform=transforms.Compose(test_transforms))
        train_ds, _ = validation_split(train_ds, val_share=val_pct, shuffle_data=True)
        _, val_ds   = validation_split(val_ds,   val_share=val_pct, shuffle_data=train_ds.index_mapping)
    else:
        val_ds = None

    test_ds = datasets.ImageFolder(val_dir, transform=transforms.Compose(test_transforms))
    # test_ds = datasets.ImageFolder(data_root_dir+'/test', transform=transforms.Compose(test_transforms))

    if dbg_ds_size > 0:
        train_ds, = validation_split(train_ds, val_share=(1.0 - float(dbg_ds_size) / len(train_ds)))
        logging.warning('DEBUG: setting train dataset size = {}'.format(len(train_ds)))

    train_ds = DatasetIndexingWrapper(train_ds)

    if test_batch_size is None or test_batch_size == 0:
        test_batch_size = batch_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False, **kwargs) if create_val else None
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def compute_mean_std_per_channel(dataset):
    assert len(dataset) > 0
    assert dataset[0][0].dim() == 3, 'dataset has {} dimensions, but needs 3'.format(dataset[0][0].dim())

    imsize = dataset[0][0].size()
    sum1, sum2, count = torch.FloatTensor(imsize[0]).zero_(), torch.FloatTensor(imsize[0]).zero_(), 0

    # lazy two-pass version
    for i in range(len(dataset)):
        img = dataset[i][0]
        count += img.numel() / img.size(0)
        sum1 += img.sum(dim=2).sum(dim=1)
    mean = sum1 / count

    mus = mean.unsqueeze(1).unsqueeze(2)
    for i in range(len(dataset)):
        img = dataset[i][0]
        sum2 += ((img - mus) * (img - mus)).sum(dim=2).sum(dim=1)
    std = torch.sqrt(sum2 / (count - 1))

    return mean, std

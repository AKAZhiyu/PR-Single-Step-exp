import os
import torch


from .cifar10s import load_cifar10s
from .cifar100s import load_cifar100s

from .semisup import get_semisup_dataloaders


SEMISUP_DATASETS = ['cifar10s', 'cifar100s']
DATASETS = ['cifar10', 'svhn', 'cifar100', 'tiny-imagenet', 'imagenet100'] + SEMISUP_DATASETS

_LOAD_DATASET_FN = {
    'cifar10s': load_cifar10s,
    'cifar100s': load_cifar100s
}




def load_data(data_dir, batch_size=256, batch_size_test=256, num_workers=8, use_augmentation=False, shuffle_train=True, 
              aux_data_filename=None, unsup_fraction=None, validation=False, aug_transform=None):
    """
    Returns train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        batch_size (int): batch size for training.
        batch_size_test (int): batch size for validation.
        num_workers (int): number of workers for loading the data.
        use_augmentation (bool): whether to use augmentations for training set.
        shuffle_train (bool): whether to shuffle training set.
        aux_data_filename (str): path to unlabelled data.
        unsup_fraction (float): fraction of unlabelled data per batch.
        validation (bool): if True, also returns a validation dataloader for unspervised cifar10 (as in Gowal et al, 2020).
    """
    dataset = os.path.basename(os.path.normpath(data_dir)) + 's'
    load_dataset_fn = _LOAD_DATASET_FN[dataset]

    
    train_dataset, test_dataset, val_dataset = load_dataset_fn(data_dir=data_dir, use_augmentation=use_augmentation, 
                                                                aux_data_filename=aux_data_filename, validation=validation, aug_transform=aug_transform)
    
    train_dataloader, test_dataloader, val_dataloader = get_semisup_dataloaders(
        train_dataset, test_dataset, val_dataset, batch_size, batch_size_test, num_workers, unsup_fraction)
        
    return train_dataset, test_dataset, train_dataloader, test_dataloader
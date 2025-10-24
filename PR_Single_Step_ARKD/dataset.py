import os
import numpy as np
import torch
# from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import glob
from torch.utils.data import Dataset

EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = '/media/data1/dongjunh/Robust_distill/Ours/data/tiny-imagenet-200/wnids.txt'
VAL_ANNOTATION_FILE = '/media/data1/dongjunh/Robust_distill/Ours/data/tiny-imagenet-200/val/val_annotations.txt'


class Triplet_CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Triplet_CIFAR10, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)
        

        # print(self.targets)
        self.labels_set = set(np.array(self.targets))
        self.label_to_indices = {label: np.where(np.array(self.targets) == label)[0]
                                    for label in self.labels_set}
        # print(self.labels_set)              
        # print(self.label_to_indices)        # 每类标签对应index     

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        ############## Negative sampling ##############
        negative_label = np.random.choice(list(self.labels_set - set([target])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        
        negative_img, negative_target = self.data[int(negative_index)], self.targets[int(negative_index)]
        negative_img = Image.fromarray(negative_img)

        ############## Negative sampling ##############



        if self.transform is not None:
            img = self.transform(img)
            negative_img = self.transform(negative_img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            negative_target = self.target_transform(negative_target)
        
        return img, negative_img, target, negative_target
        # return img, target, index
    
    @property
    def num_classes(self):
        return 10
    

class Triplet_CIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Triplet_CIFAR100, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)
        

        # print(self.targets)
        self.labels_set = set(np.array(self.targets))
        self.label_to_indices = {label: np.where(np.array(self.targets) == label)[0]
                                    for label in self.labels_set}
        # print(self.labels_set)              
        # print(self.label_to_indices)        # 每类标签对应index     

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        ############## Negative sampling ##############
        negative_label = np.random.choice(list(self.labels_set - set([target])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        
        negative_img, negative_target = self.data[int(negative_index)], self.targets[int(negative_index)]
        negative_img = Image.fromarray(negative_img)

        ############## Negative sampling ##############



        if self.transform is not None:
            img = self.transform(img)
            negative_img = self.transform(negative_img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            negative_target = self.target_transform(negative_target)
        
        return img, negative_img, target, negative_target
        # return img, target, index
    
    @property
    def num_classes(self):
        return 100
    
class Triplet_SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(Triplet_SVHN, self).__init__(root, split='train', transform=transform,
                                     target_transform=target_transform, download=download)
        

        # print(self.targets)
        self.labels_set = set(np.array(self.targets))
        self.label_to_indices = {label: np.where(np.array(self.targets) == label)[0]
                                    for label in self.labels_set}
        # print(self.labels_set)              
        # print(self.label_to_indices)        # 每类标签对应index     

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        ############## Negative sampling ##############
        negative_label = np.random.choice(list(self.labels_set - set([target])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        
        negative_img, negative_target = self.data[int(negative_index)], self.targets[int(negative_index)]
        negative_img = Image.fromarray(negative_img)

        ############## Negative sampling ##############



        if self.transform is not None:
            img = self.transform(img)
            negative_img = self.transform(negative_img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            negative_target = self.target_transform(negative_target)
        
        return img, negative_img, target, negative_target
        # return img, target, index
    
    @property
    def num_classes(self):
        return 10
    

class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img
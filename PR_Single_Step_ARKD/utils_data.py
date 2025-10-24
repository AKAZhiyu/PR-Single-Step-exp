import os
import math
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import functional
from torchvision.transforms import InterpolationMode as Interpolation
from PIL import Image

class IDBH(torch.nn.Module):
    def __init__(self, version):
        super().__init__()
        if version == 'IDBH_weak':
            layers = [
                transforms.RandomHorizontalFlip(),
                CropShift(0, 11),
                ColorShape('color'),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5)
            ]
        elif version == 'IDBH_strong':
            layers = [
                transforms.RandomHorizontalFlip(),
                CropShift(0, 11),
                ColorShape('color'),
                transforms.ToTensor(),
                transforms.RandomErasing(p=1)
            ]
        elif version == 'svhn':
            layers = [
                transforms.RandomHorizontalFlip(),
                CropShift(0, 9),
                ColorShape('shape'),
                transforms.ToTensor(),
                transforms.RandomErasing(p=1, scale=(0.02, 0.5))
            ]
        else:
            raise Exception("IDBH: invalid version string")
        
        self.layers = transforms.Compose(layers)
            
    def forward(self, img):
        return self.layers(img)
    
        
class ColorShape(torch.nn.Module):
    ColorBiased = [
        (0.125, 'color', 0.1, 1.9),
        (0.125, 'brightness', 0.5, 1.9),
        (0.125, 'contrast', 0.5, 1.9),
        (0.125, 'sharpness', 0.1, 1.9),
        (0.125, 'autocontrast'),
        (0.125, 'equalize'),
        (0.125, 'shear', 0.05, 0.15),
        (0.125, 'rotate', 1, 11)  
    ]
    ShapeBiased = [
        (0.08, 'color', 0.1, 1.9),
        (0.08, 'brightness', 0.5, 1.9),
        (0.04, 'contrast', 0.5, 1.9),
        (0.08, 'sharpness', 0.1, 1.9),
        (0.04, 'autocontrast'),
        (0.08, 'equalize'),
        (0.3, 'shear', 0.05, 0.35),
        (0.3, 'rotate', 1, 31)
    ]
    
    def __init__(self, version='color'):
        super().__init__()

        assert version in ['color', 'shape']
        space = self.ColorBiased if version == 'color' else self.ShapeBiased
        
        self.space = {}
        p_accu = 0.0
        for trans in space:
            p = trans[0]
            self.space[(p_accu, p_accu+p)] = trans[1:]
            p_accu += p
            
    def transform(self, img, trans):
        if len(trans) == 1:
            trans = trans[0]
        else:
            lower, upper = trans[1:]
            trans = trans[0]
            if trans == 'rotate':
                strength = torch.randint(lower, upper, (1,)).item()
            else:
                strength = torch.rand(1) * (upper-lower) + lower

        if trans == 'color':
            img = functional.adjust_saturation(img, strength)
        elif trans == 'brightness':
            img = functional.adjust_brightness(img, strength)
        elif trans == 'contrast':
            img = functional.adjust_contrast(img, strength)
        elif trans == 'sharpness':
            img = functional.adjust_sharpness(img, strength)
        elif trans == 'shear':
            if torch.randint(2, (1,)):
                # random sign
                strength *= -1
            strength = math.degrees(strength)
            strength = [strength, 0.0] if torch.randint(2, (1,)) else [0.0, strength]
            img = functional.affine(img,
                           angle=0.0,
                           translate=[0, 0],
                           scale=1.0,
                           shear=strength,
                           interpolation=Interpolation.NEAREST,
                           fill=0)
        elif trans == 'rotate':
            if torch.randint(2, (1,)):
                strength *= -1
            img = functional.rotate(img, angle=strength, interpolation=Interpolation.NEAREST, fill=0)
        elif trans == 'autocontrast':
            img = functional.autocontrast(img)
        elif trans == 'equalize':
            img = functional.equalize(img)

        return img
            
    def forward(self, img):
        roll = torch.rand(1)
        for (lower, upper), trans in self.space.items():
            if roll <= upper and roll >= lower:
                return self.transform(img, trans)
        
        return img

class CropShift(torch.nn.Module):
    def __init__(self, low, high=None):
        super().__init__()
        high = low if high is None else high
        self.low, self.high = int(low), int(high)
        
    def sample_top(self, x, y):
        x = torch.randint(0, x+1, (1,)).item()
        y = torch.randint(0, y+1, (1,)).item()
        return x, y
            
    def forward(self, img):
        if self.low == self.high:
            strength = self.low
        else:
            strength = torch.randint(self.low, self.high, (1,)).item()
        
        # w, h = F.get_image_size(img)
        w, h = img.size
        crop_x = torch.randint(0, strength+1, (1,)).item()
        crop_y = strength - crop_x
        crop_w, crop_h = w - crop_x, h - crop_y

        top_x, top_y = self.sample_top(crop_x, crop_y)
        
        img = functional.crop(img, top_y, top_x, crop_h, crop_w)
        img = functional.pad(img, padding=[crop_x, crop_y], fill=0)
        
        top_x, top_y = self.sample_top(crop_x, crop_y)
        
        return functional.crop(img, top_y, top_x, h, w)
    

def load_data(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == 'CIFAR-10':
        num_classes = 10
        if args.augmentation != 'vanilla':
            transform_train = IDBH(args.augmentation)
        trainset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'CIFAR-100':
        num_classes = 100
        if args.augmentation != 'vanilla':
            transform_train = IDBH(args.augmentation)
        trainset = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader, num_classes



class Get_Dataset_C10(torchvision.datasets.CIFAR10):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        
        img, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(img)
        image_clean = self.transform[0](image)
        image_auto1 = self.transform[1](image)
        return image_clean, image_auto1, target

class Get_Dataset_C100(torchvision.datasets.CIFAR100):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        
        img, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(img)
        image_clean = self.transform[0](image)
        image_auto1 = self.transform[1](image)
        return image_clean, image_auto1, target


## load Dual augmentation data
def load_dual_aug_data(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.augmentation == 'IDBH_V_W':         # Vanilla + Weak
        transform_auto = IDBH(version='IDBH_weak')
    elif args.augmentation == 'IDBH_V_S':
        transform_auto = IDBH(version='IDBH_strong')
    elif args.augmentation == 'IDBH_W_W':
        transform_train = IDBH(version='IDBH_weak')
        transform_auto = IDBH(version='IDBH_weak')
    elif args.augmentation == 'IDBH_W_S':
        transform_train = IDBH(version='IDBH_weak')
        transform_auto = IDBH(version='IDBH_strong')
    elif args.augmentation == 'IDBH_S_S':
        transform_train = IDBH(version='IDBH_strong')
        transform_auto = IDBH(version='IDBH_strong')

    if args.dataset == 'cifar10':
        num_classes = 10
        trainset = Get_Dataset_C10(root='../datasets/cifar10', train=True, download=True, transform=[transform_train, transform_auto])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'cifar100':
        num_classes = 100
        trainset = Get_Dataset_C100(root='../datasets/cifar100', train=True, download=True, transform=[transform_train, transform_auto])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader, num_classes
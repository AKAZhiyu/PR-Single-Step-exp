from __future__ import print_function
from email.policy import default
import os
import argparse
from tabnanny import verbose
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable


from aux.preact_resnetwithswish import preact_resnetwithswish
from aux.wideresnetwithswish import wideresnetwithswish
from aux.preact_resnet import preact_resnet
from models import *


from imagenet100 import load_imagenet100
# from mart import mart_loss
import numpy as np
import time
from tqdm import tqdm

from utils import str2bool, checkpoint_load
from autoattack import AutoAttack

parser = argparse.ArgumentParser(description='Robustness validation')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--exp_base', type=str, default='./ckpts/ckpts_base_From_WRN28_Pang22/')
parser.add_argument('--exp_name', default='Origin',
                    help='directory of model for saving checkpoint')

parser.add_argument('--gpu', default='0', type=str,
                    help='GPU choice')
parser.add_argument('--epsilon', type=float, default=8.0,
                    help='perturbation')
parser.add_argument('--network',type=str, default='resnet18',
                    help='resnet18|preact_res18|resnet34|resnet50|WRN|WRN28_Swish|MNV2|Res_Split_BN|ViTB|DeiTS')

parser.add_argument('--attack_type', default='AA', type=str,
                    help='Adversarial type selection: AA, FGSM PGD, CW, MIM, FGSM, PGD_L2, AA_L2')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10|cifar100|SVHN|imagenet100|TinyImagenet')
parser.add_argument('--attack_step', type=int, default=20,
                    help='PGD-20, PGD-20, CW-30')


"""
New T

CUDA_VISIBLE_DEVICES=X python test_robust.py --dataset cifar10 --network resnet18 --exp_name XXXX
CUDA_VISIBLE_DEVICES=X python test_robust.py --dataset cifar10 --network Res_Split_BN --exp_base ./ckpts/ckpts_base_From_WRN28_Pang22_Dual_Branch/ --exp_name XXXX

CUDA_VISIBLE_DEVICES=X python test_robust.py --dataset cifar10 --network preact_res18 --exp_name XXXX
CUDA_VISIBLE_DEVICES=X python test_robust.py --dataset cifar10 --network WRN28_Swish --exp_name XXXX



"""


args = parser.parse_args()

print(str(vars(args)))


args.epsilon = args.epsilon/255.0

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=2.0/255.0):
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def _pgd_whitebox_L2(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=15.0/255.0):

    batch_size = len(X)
    delta = 0.001 * torch.randn(X.shape).cuda().detach()
    delta = Variable(delta.data, requires_grad=True)

    # Setup optimizers
    optimizer_delta = optim.SGD([delta], lr=step_size)

    for _ in range(num_steps):
        adv = X + delta

        # optimize
        optimizer_delta.zero_grad()
        with torch.enable_grad():
            loss = (-1) * nn.CrossEntropyLoss()(model(adv), y)
        loss.backward()
        # renorming gradient
        grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
        optimizer_delta.step()

        # projection
        delta.data.add_(X)
        delta.data.clamp_(0, 1).sub_(X)
        delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
    X_pgd = Variable(X + delta, requires_grad=False)
    
    return X_pgd

def _fgsm_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon):
    X_pgd = Variable(X.data, requires_grad=True)

    # random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    # X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
    loss.backward()
    eta = epsilon * X_pgd.grad.data.sign()
    X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def _cw_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=30,
                  step_size=0.003):
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            loss = cwloss(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def _mim_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=2.0/255.0,
                  decay = 1.0):
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    momentum = torch.zeros_like(X_pgd).detach().cuda()

    for _ in range(num_steps):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        # loss.backward()
        # momentum generation #
        grad = torch.autograd.grad(loss, X_pgd, retain_graph=False, create_graph=False)[0]
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        grad = grad + momentum * decay
        momentum = grad
        # momentum generation #

        eta = step_size * grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        
    return X_pgd

def cwloss(output, target, confidence=50,num_classes=10):
    # compute the probability of the label class versus the maximum other
    if args.dataset == 'cifar10' or args.dataset == 'SVHN':
        num_classes=10
    elif args.dataset == 'cifar100' or args.dataset == 'imagenet100':
        num_classes=100
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# exp_dir = args.exp_name
exp_dir = os.path.join(args.exp_base, args.dataset, args.exp_name)
ckpt_path = os.path.join(exp_dir, "best_rob_acc_epoch.pt")



torch.manual_seed(args.seed)
device = torch.device("cuda")
kwargs = {'num_workers': 10, 'pin_memory': True}
torch.backends.cudnn.benchmark = True

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'cifar10':
    num_classes = 10
    testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)
elif args.dataset == 'cifar100':
    num_classes = 100
    testset = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)
elif args.dataset == 'SVHN':
    num_classes = 10
    testset = torchvision.datasets.SVHN(root='../datasets/SVHN', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)
elif args.dataset == 'imagenet100':
    num_classes = 100
    trainset, testset = load_imagenet100(data_dir='/media/dataX/dongjunh/ImageNet-CLS')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == 'TinyImagenet':
    num_classes = 200
    testset, test_loader= New_ImageNet_get_loaders_64_testloader(dir_="/media/data1/dongjunh/Robustness/dataset/tiny-imagenet-200", batch_size=args.test_batch_size)

if args.network == 'resnet18':

    ################# Normally Load_ckpt #################
    if args.dataset == 'imagenet100':
        model = torchvision.models.resnet18(num_classes=100).to(device)
    else:
        model = resnet18(num_classes=num_classes).to(device)
    checkpoint_load(torch.load(ckpt_path, map_location='cuda:0'), model)
    ################# Normally Load_ckpt #################
    ################# HAT Load_ckpt #################
    # model = ResNet18(num_classes=num_classes)
    # model = torch.nn.Sequential(model)
    # model = torch.nn.DataParallel(model).to(device)
    # model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    ################# HAT Load_ckpt #################
elif args.network == 'preact_res18':
    model = PreActResNet18(num_classes=num_classes).to(device)
    # model = preact_resnet(name='preact-resnet18').to(device)
    model.load_state_dict(torch.load(ckpt_path))
    # checkpoint_load(torch.load(ckpt_path, map_location='cuda:0'), model)
elif args.network == 'resnet34':
    model = torchvision.models.resnet34(num_classes=100).to(device)
    checkpoint_load(torch.load(ckpt_path), model)
elif args.network == 'resnet50':
    model = torchvision.models.resnet50(num_classes=100).to(device)
    checkpoint_load(torch.load(ckpt_path), model)
elif args.network == 'WRN':
    model = WideResNet(depth=34, widen_factor=10, num_classes=num_classes).to(device)
    checkpoint_load(torch.load(ckpt_path), model)
    # model.load_state_dict(torch.load(ckpt_path))
elif args.network == 'WRN28_Swish':
    model = wideresnetwithswish("wrn-28-10-swish", dataset=args.dataset, num_classes=num_classes).to(device)
    checkpoint_load(torch.load(ckpt_path), model)
elif args.network == 'MNV2':
    if args.dataset == 'imagenet100':
        model = torchvision.models.mobilenet_v2(num_classes=num_classes).to(device)
    else:
        model = mobilenet_v2(num_classes=num_classes).to(device)
    checkpoint_load(torch.load(ckpt_path), model)
elif args.network == 'Res_Split_BN':
    if args.dataset == 'TinyImagenet':
        model = Tiny_ResNet18(num_classes=num_classes).to(device)
    else:
        model = ResNet18_SplitBN(num_classes=num_classes).to(device)
    checkpoint_load(torch.load(ckpt_path), model)
elif args.network == 'ViTB':
    model = vit_base_patch16_224(img_size=32,num_classes =10,patch_size=4).cuda()
    checkpoint_load(torch.load(ckpt_path), model)
elif args.network == 'DeiTS':
    model = deit_small_patch16_224(img_size=32,num_classes =10, patch_size=4).cuda()
    checkpoint_load(torch.load(ckpt_path), model)
    
if args.attack_type == 'AA':
    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard', verbose=False)
elif args.attack_type == 'AA_L2':
    adversary = AutoAttack(model, norm='L2', eps=args.epsilon, version='standard', verbose=False)


def main():
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in tqdm(test_loader, ncols = 80):
        data, target = data.to(device), target.to(device)
        
        X, y = Variable(data, requires_grad=True), Variable(target)
        if args.attack_type == 'AA' or args.attack_type == 'AA_L2':
            x_adv = adversary.run_standard_evaluation(X, y, bs=args.test_batch_size)
        elif args.attack_type == 'PGD':
            x_adv = _pgd_whitebox(model, X, y, num_steps=args.attack_step)
        elif args.attack_type == 'PGD_L2':
            x_adv = _pgd_whitebox_L2(model, X, y, num_steps=args.attack_step)
        elif args.attack_type == 'CW':
            x_adv = _cw_whitebox(model, X, y)
        elif args.attack_type == 'FGSM':
            x_adv = _fgsm_whitebox(model, X, y)
        elif args.attack_type == 'MIM':
            x_adv = _mim_whitebox(model, X, y)
            

        
        out = model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        err_adv = (model(x_adv).data.max(1)[1] != y.data).float().sum()

        robust_err_total += err_adv
        natural_err_total += err

        # print(str(1 - natural_err_total.item()))
    print("Evaluation on {} Attack".format(args.attack_type))
    print('natural_acc: ' + str(1 - natural_err_total.item() / len(test_loader.dataset)))
    print('robust_acc: ' + str(1- robust_err_total.item() / len(test_loader.dataset)))

if __name__ == '__main__':
    attack_test_list = ['PGD', 'CW', 'AA']
    for attack_item in attack_test_list:
        args.attack_type = attack_item
        main()
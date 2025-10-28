import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ARKD_loss import *
from models import *
from aux.wideresnetwithswish import wideresnetwithswish
import torchvision
from torchvision import datasets, transforms
from utils import str2bool,str2float, eval_adv_test_whitebox, Cutout, dkd_loss, clip_bound_compute
from tqdm import tqdm

from dataset import Triplet_CIFAR10

import time
from torch.autograd import profiler

# we fix the random seed to 0, this method can keep the results consistent in the same conputer.

"""
Default: All the predictions are aligned with teacher's clean predictions.
200 epochs + cyclic + alpha=1.0

RSLAD:
CUDA_VISIBLE_DEVICES=1 python robust_distill_base.py --dataset cifar10 --teacher_model WRN28_Swish_Pang22 --student_model Res18 --advloss_type KL --exp_name C10_R18_KL

AdaAD
CUDA_VISIBLE_DEVICES=1 python robust_distill_base.py --dataset cifar10 --teacher_model WRN28_Swish_Pang22 --student_model Res18 --advloss_type AdaKL --exp_name C10_R18_AdaKL

N_FGSM
CUDA_VISIBLE_DEVICES=1 python robust_distill_base.py --dataset cifar10 --teacher_model WRN28_Swish_Pang22 --student_model Res18 --advloss_type N_FGSM --exp_name C10_R18_N_FGSM

NuAT
CUDA_VISIBLE_DEVICES=1 python robust_distill_base.py --dataset cifar10 --teacher_model WRN28_Swish_Pang22 --student_model Res18 --advloss_type NuAT --exp_name C10_R18_NuAT

# Teacher_ckpts links:
https://drive.google.com/file/d/16ChNkterCp17BXv-xxqpfedb4u2_CjjS   as Teacher_C10_Pang2022Robustness_WRN28_10.pt 
https://drive.google.com/file/d/1VDDM_j5M4b6sZpt1Nnhkr8FER3kjE33M   as Teacher_C100_Pang2022Robustness_WRN28_10.pt 

"""

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)

parser = argparse.ArgumentParser(description='Robustness validation')
parser.add_argument('--exp_basedir', type=str, default='../ckpts/ckpts_base_From_WRN28_Pang22/')
parser.add_argument('--exp_name', type=str, default='temp_exp')
parser.add_argument('--teacher_model', type=str, default='WRN28_Swish_Pang22',
                    help='Res18|MNV2|WRN28_Swish_Wang23|WRN28_Swish_Pang22|WRN34|WRN70')
parser.add_argument('--student_model', type=str, default='Res18',
                    help='Res18|MNV2')
parser.add_argument('--wd', type=str2float, default='5e-4', help='weight decay')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='training epoch')
parser.add_argument('--scheduler', type=str, default='cyclic', help='cyclic')
parser.add_argument('--alpha', type=str2float, default=1.0, 
                    help='weight for adv logit alignment')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10|cifar100')

parser.add_argument('--epsilon', type=str2float, default="8.0/255.0")
parser.add_argument('--perturb_steps', type=int, default=10)
parser.add_argument('--advloss_type', type=str, default='AdaKL',
                    help='CE|KL|SE|AdaKL|AdaSE for adversary generation, Ada means computing gradient of the teacher branch')

parser.add_argument('--temp', type=str2float, default=1.0,
                    help='temperature')

# New component
parser.add_argument('--augmentation', type=str, default='vanilla',
                    help='vanilla')
parser.add_argument('--clip_loss_thr', type=float, default=0.0,
                    help='Clip the metric loss < 1.0, e.g., 0.1, 0.2, 0.3, 0.4')




args = parser.parse_args()

args.exp_name = os.path.join(args.exp_basedir, args.dataset, args.exp_name)

os.makedirs(args.exp_name, exist_ok=True)
results_log_csv_name = os.path.join(args.exp_name, 'results.csv')
log_path = os.path.join(args.exp_name, 'log.txt')

log(log_path, str(vars(args)))



torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# import torch.backends.cudnn as cudnn


epochs = args.epochs
batch_size = args.batch_size # 128

epsilon = args.epsilon # 8/255.0
step_size = epsilon / 4.0
perturb_steps = args.perturb_steps
adv_config = {
    "epsilon": epsilon,
    "step_size": step_size,
    "perturb_steps": perturb_steps
}



#######################################  Dataset Start  #######################################
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'cifar10':
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
# elif args.dataset == 'cifar10':
elif args.dataset == 'cifar100':
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

args.num_classes = num_classes
#######################################  Dataset End  #######################################



#######################################  T&S models Start  #######################################
# Student
if args.student_model == 'Res18':
    student = resnet18(num_classes=num_classes)

elif args.student_model == 'MNV2':
    student = mobilenet_v2(num_classes=num_classes)

student = torch.nn.DataParallel(student)
student = student.cuda()
student.train()

optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
# init_scheduler
if args.scheduler == 'cyclic':
    num_samples = len(trainset)
    update_steps = int(np.floor(num_samples/batch_size) + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.25,
                                                            steps_per_epoch=update_steps, epochs=int(args.epochs))

# Teacher
if args.teacher_model == 'WRN34':
    teacher = wideresnet(num_classes=num_classes)
    if args.dataset == 'cifar10':
        teacher.load_state_dict(torch.load('../Teacher_ckpts/Teacher_C10_WRN34.pt'))
    elif args.dataset == 'cifar100':
        teacher.load_state_dict(torch.load('../Teacher_ckpts/Teacher_C100_WRN34.pt'))
elif args.teacher_model == 'WRN28_Swish_Pang22':
    if args.dataset == 'cifar10':
        teacher = wideresnetwithswish("wrn-28-10-swish", dataset='cifar10', num_classes=num_classes)
        teacher.load_state_dict(torch.load('../Teacher_ckpts/Teacher_C10_Pang2022Robustness_WRN28_10.pt'))
    elif args.dataset == 'cifar100':
        teacher = wideresnetwithswish("wrn-28-10-swish", dataset='cifar100', num_classes=num_classes)
        teacher.load_state_dict(torch.load('../Teacher_ckpts/Teacher_C100_Pang2022Robustness_WRN28_10.pt'))
elif args.teacher_model == 'WRN28_Swish_Wang23':
    if args.dataset == 'cifar10':
        teacher = wideresnetwithswish("wrn-28-10-swish", dataset='cifar10', num_classes=num_classes)
        teacher.load_state_dict(torch.load('../Teacher_ckpts/Teacher_C10_Wang2023Better_WRN-28-10.pt'))
    elif args.dataset == 'cifar100':
        teacher = wideresnetwithswish("wrn-28-10-swish", dataset='cifar100', num_classes=num_classes)
        teacher.load_state_dict(torch.load('../Teacher_ckpts/Teacher_C100_Wang2023Better_WRN-28-10.pt'))
elif args.teacher_model == 'Res18':
    teacher = resnet18(num_classes=num_classes)
    if args.dataset == 'cifar10':
        teacher.load_state_dict(torch.load('../Teacher_ckpts/Teacher_C10_R18.pt'))
    if args.dataset == 'cifar100':
        teacher.load_state_dict(torch.load('../Teacher_ckpts/Teacher_C100_R18.pt'))
elif args.teacher_model == 'MNV2':
    teacher = mobilenet_v2(num_classes=num_classes)
    if args.dataset == 'cifar10':
        teacher.load_state_dict(torch.load('../Teacher_ckpts/Teacher_C10_MNV2.pt'))
    if args.dataset == 'cifar100':
        teacher.load_state_dict(torch.load('../Teacher_ckpts/Teacher_C100_MNV2.pt'))

teacher = torch.nn.DataParallel(teacher)
teacher = teacher.cuda()
teacher.eval()

#######################################  T&S models End  #######################################




with open(results_log_csv_name, 'w') as f:
    f.write('epoch, test_clean_acc, test_PGD_acc\n')
best_acc_nat = 0.0
best_acc_nat_epoch = 0
best_acc_adv = 0.0
best_acc_adv_epoch = 0

for epoch in range(1,epochs+1):
    for step, total_input in enumerate(tqdm(trainloader, ncols=80)):

        train_batch_data, train_batch_labels = total_input[0], total_input[1] # (train_batch_data,train_batch_labels)


        student.train()
        train_batch_data = train_batch_data.cuda()
        train_batch_labels = train_batch_labels.cuda()
        
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(train_batch_data)

        
        ####################################### Adversary Generation (Start)#######################################
        if args.advloss_type == 'AdaKL' or args.advloss_type == 'AdaSE':
            x_adv = adaad_adv_generation(student, teacher, train_batch_data, train_batch_labels,
                                        optimizer, adv_config, advloss_type=args.advloss_type)
        elif args.advloss_type == 'KL' or args.advloss_type == 'SE' or args.advloss_type == 'CE':
            x_adv = rslad_adv_generation(student, teacher_logits, train_batch_data, train_batch_labels,
                                        optimizer, adv_config, advloss_type=args.advloss_type)
        elif args.advloss_type == 'N_FGSM':
            x_adv = N_FGSM_adv_generation(student, train_batch_data, train_batch_labels, optimizer, adv_config)
        elif args.advloss_type == 'NuAT':
            x_adv = NuAT_adv_generation(student, train_batch_data, train_batch_labels, optimizer, adv_config)
            
        ####################################### Adversary Generation (Start)#######################################

        student.train()

        
        loss = ARKD_outer_minimization(teacher, teacher_logits, student, train_batch_data, x_adv, train_batch_labels, args)

        loss.backward()
        optimizer.step()
        if args.scheduler == 'cyclic':
            scheduler.step()
        

        if step % 100 == 0:
            print('loss',loss.item())
            log(log_path,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(train_batch_data), len(trainloader.dataset),
                       100. * step / len(trainloader), loss.item()))
            
    ############################ Robust Test ############################


    test_accs_adv = []
    test_accs_nat = []
    student.eval()

    natural_acc_total, robust_acc_total = eval_adv_test_whitebox(student, testloader)
    test_acc_nat = natural_acc_total.item()
    test_acc_adv = robust_acc_total.item()

    log(log_path,'natural acc {}'.format(test_acc_nat))
    log(log_path,'robust acc {}'.format(test_acc_adv))

    with open(results_log_csv_name, 'a') as f:
        f.write('%5d, %.5f, %.5f,\n'
                '' % (epoch, test_acc_nat, test_acc_adv))

    # Last epoch save
    if epoch == epochs: 
        torch.save(student.state_dict(), os.path.join(args.exp_name, "last_epoch.pt"))
    # Best nat_acc epoch save
    if test_acc_nat > best_acc_nat:
        best_acc_nat = test_acc_nat
        best_acc_nat_epoch = epoch
        torch.save(student.state_dict(), os.path.join(args.exp_name, "best_nat_acc_epoch.pt"))
    # Best adv_acc epoch save
    if test_acc_adv > best_acc_adv:
        best_acc_adv = test_acc_adv
        best_acc_adv_epoch = epoch
        torch.save(student.state_dict(), os.path.join(args.exp_name, "best_rob_acc_epoch.pt"))

        
    if args.scheduler == 'step' and epoch in [215,260,285]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    ############################ Robust Test ############################


print("best_Test_Clean_acc: %.3f" % best_acc_nat)
print("best_Test_Clean_acc_epoch: %d" % best_acc_nat_epoch)
print("best_Test_PGD10_acc: %.3f" % best_acc_adv)
print("best_Test_PGD10_acc_epoch: %d" % best_acc_adv_epoch)

# best ACC
with open(results_log_csv_name, 'a') as f:
    f.write('%s,%03d,%0.3f,%s,%03d,%0.3f,\n' % ('best clean acc (test)',
                                                best_acc_nat_epoch,
                                                best_acc_nat,
                                                'best PGD20 acc (test)',
                                                best_acc_adv_epoch,
                                                best_acc_adv))
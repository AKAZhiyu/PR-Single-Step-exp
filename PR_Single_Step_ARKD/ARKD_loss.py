import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from utils import clip_bound_compute

############################################ Base ############################################

def rslad_adv_generation(model, teacher_logits, x_natural, y,
                        optimizer, adv_config, advloss_type='KL'):
    
    epsilon = adv_config['epsilon']
    step_size = adv_config['step_size']
    perturb_steps = adv_config['perturb_steps']

    criterion_kl = nn.KLDivLoss(reduction='none')
    criterion_ce = nn.CrossEntropyLoss()
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            if advloss_type == 'KL':
                loss = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                        F.softmax(teacher_logits, dim=1))
                loss = torch.sum(loss)
            elif advloss_type == 'CE':
                loss = criterion_ce(model(x_adv), y)
            elif advloss_type == 'SE':
                loss = torch.sum( (F.softmax(model(x_adv), dim=1) - F.softmax(teacher_logits, dim=1)) ** 2)

        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    model.train()
    # zero gradient
    optimizer.zero_grad()

    return x_adv


def adaad_adv_generation(model, teacher, x_natural, y,
                        optimizer, adv_config, advloss_type='AdaKL'):
    
    epsilon = adv_config['epsilon']
    step_size = adv_config['step_size']
    perturb_steps = adv_config['perturb_steps']

    criterion_kl = nn.KLDivLoss(reduction='none')
    model.eval()
    teacher.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            if advloss_type == 'AdaKL':
                loss = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                        F.softmax(teacher(x_adv), dim=1))
                loss = torch.sum(loss)
            elif advloss_type == 'AdaSE':
                loss = torch.sum( (F.softmax(model(x_adv), dim=1) - F.softmax(teacher(x_adv), dim=1)) ** 2)

        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    model.train()
    # zero gradient
    optimizer.zero_grad()

    return x_adv


def ARKD_outer_minimization(teacher, T_nat_logits, student, x_nat, x_adv, y, args):
    criterion_KL = nn.KLDivLoss(reduction='batchmean')

    clip_bound_value = clip_bound_compute(args.clip_loss_thr, args.num_classes, args.advloss_type)
                                          
    with torch.no_grad():
        T_adv_logits = teacher(x_adv)
    
    S_nat_logits = student(x_nat)
    S_adv_logits = student(x_adv)

    pred_teacher_adv = F.softmax(T_adv_logits.detach() / args.temp, dim=1)
    pred_teacher_nat = F.softmax(T_nat_logits.detach() / args.temp, dim=1)
    pred_student_adv = F.softmax(S_adv_logits / args.temp, dim=1)
    pred_student_nat = F.softmax(S_nat_logits / args.temp, dim=1)
    
    if args.advloss_type == 'SE' or args.advloss_type == 'AdaSE':
        # Adv loss
        loss_adv = torch.sum((pred_student_adv - pred_teacher_nat) ** 2, dim=-1).mean()
        # Nat loss
        loss_nat = torch.sum((pred_student_nat - pred_teacher_nat) ** 2, dim=-1).mean()
    else:
        # Adv loss
        loss_adv = criterion_KL(torch.log(pred_student_adv), pred_teacher_nat)
        # Nat loss
        loss_nat = criterion_KL(torch.log(pred_student_nat), pred_teacher_nat)

    if args.clip_loss_thr > 0.0:
        loss_adv = F.relu(loss_adv - clip_bound_value)
        loss_nat = F.relu(loss_nat - clip_bound_value)

    loss = args.alpha * loss_adv + (1 - args.alpha) * loss_nat

    return loss

############################################ Base ############################################



############################################ Adv_Regularizer ############################################
def _l2_norm_batch(v):
    return (v**2).sum(dim=(1, 2, 3))**0.5

def _get_input_grad_for_align(model, X, y, eps):
    model_was_training = model.training
    model.eval()

    eta = (torch.rand_like(X) * 2 - 1) * eps
    x_eta = (X + eta).detach()
    x_eta.requires_grad = True

    x_natural_copy = X.detach()
    x_natural_copy.requires_grad = True
    
    with torch.enable_grad():
        logits_nat = model(x_natural_copy)
        loss_nat = F.cross_entropy(logits_nat, y)
        
        logits_eta = model(x_eta)
        loss_eta = F.cross_entropy(logits_eta, y)

    grad1 = torch.autograd.grad(loss_nat, x_natural_copy, create_graph=True)[0]
    grad2 = torch.autograd.grad(loss_eta, x_eta, create_graph=True)[0]

    if model_was_training:
        model.train()
        
    return grad1, grad2



def grad_align_loss(model, x_natural, y, adv_config):
    grad1, grad2 = _get_input_grad_for_align(model, x_natural, y, adv_config['epsilon'])
    B = grad1.shape[0]

    grad1_flat = grad1.view(B, -1)
    grad2_flat = grad2.view(B, -1)
    cos = F.cosine_similarity(grad1_flat, grad2_flat, dim=1)
    return (1.0 - cos.mean())



############################################ Adv_Regularizer ############################################




############################################ single_step ############################################
def N_FGSM_adv_generation(model, x_natural, y, optimizer, adv_config):
    
    epsilon = adv_config['epsilon']
    # alpha = adv_config['step_size']
    # Use the recommended hyperparameter for N-FGSM in the paper    
    alpha = epsilon  
    k = 2 * epsilon

    criterion_ce = nn.CrossEntropyLoss()
    model.eval()
    
    eta = k * (torch.rand_like(x_natural).cuda().detach() * 2 - 1)
    x_aug = x_natural.detach() + eta

    x_aug.requires_grad_()
    with torch.enable_grad():
        loss = criterion_ce(model(x_aug), y)

    grad = torch.autograd.grad(loss, [x_aug])[0]

    x_adv = x_aug.detach() + alpha * torch.sign(grad.detach())

    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    model.train()
    
    optimizer.zero_grad()

    return x_adv


def NuAT_adv_generation(model, x_natural, y, optimizer, adv_config):

    epsilon = adv_config['epsilon'] 
    
    default_noise_alpha = 4.0 / 255.0
    noise_alpha = adv_config.get('nuat_noise_alpha', default_noise_alpha) 

    attack_step_size = adv_config.get('nuat_step_size', epsilon) 
    
    lambda_val = adv_config.get('nuat_lambda', 4.0)
    
    criterion_ce = nn.CrossEntropyLoss()
    model.eval()
    
    x_natural_detached = x_natural.detach()
    batch_size = x_natural_detached.shape[0]

    with torch.no_grad():
        logits_natural = model(x_natural_detached)
    
    delta_init = (torch.bernoulli(torch.ones_like(x_natural_detached) * 0.5) * 2.0 - 1.0) * noise_alpha
    x_pert = (x_natural_detached + delta_init).clamp(0.0, 1.0)
    x_pert.requires_grad_(True)

    with torch.enable_grad():
        logits_pert = model(x_pert)
        
        loss_ce = criterion_ce(logits_pert, y)
        
        loss_nuc = 0.0
        if lambda_val > 0:
            diff_matrix = logits_pert - logits_natural.detach()
            loss_nuc_raw = torch.norm(diff_matrix, p='nuc')
            loss_nuc = loss_nuc_raw / batch_size

        loss = loss_ce + lambda_val * loss_nuc

    grad = torch.autograd.grad(loss, [x_pert])[0]

    x_adv_intermediate = x_pert.detach() + attack_step_size * torch.sign(grad.detach())
    
    delta = x_adv_intermediate - x_natural_detached
    delta = torch.clamp(delta, -epsilon, epsilon)
    
    x_adv = x_natural_detached + delta
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = x_adv.detach()
    
    model.train()
    optimizer.zero_grad()

    return x_adv

def NuAT_adaad_adv_generation(model, teacher, x_natural, optimizer, adv_config):

    epsilon = adv_config['epsilon']
    default_noise_alpha = 4.0 / 255.0
    noise_alpha = adv_config.get('nuat_noise_alpha', default_noise_alpha)
    attack_step_size = adv_config.get('nuat_step_size', epsilon)
    lambda_val = adv_config.get('nuat_lambda', 4.0)

    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    teacher.eval()

    x_natural_detached = x_natural.detach()
    batch_size = x_natural_detached.shape[0]
    
    delta_init = (torch.bernoulli(torch.ones_like(x_natural_detached) * 0.5) * 2.0 - 1.0) * noise_alpha
    x_pert = (x_natural_detached + delta_init).clamp(0.0, 1.0)
    x_pert.requires_grad_(True)
    with torch.no_grad():
        logits_natural = model(x_natural_detached)
    with torch.enable_grad():
        logits_pert = model(x_pert)
        teacher_logits_pert = teacher(x_pert)
        
        loss_kl = criterion_kl(F.log_softmax(logits_pert, dim=1),
                               F.softmax(teacher_logits_pert, dim=1))
        loss_nuc = 0.0
        if lambda_val > 0:
            diff_matrix = logits_pert - logits_natural.detach()
            loss_nuc_raw = torch.norm(diff_matrix, p='nuc')
            loss_nuc = loss_nuc_raw / batch_size

        loss = loss_kl + lambda_val * loss_nuc

    grad = torch.autograd.grad(loss, [x_pert])[0]

    x_adv_intermediate = x_pert.detach() + attack_step_size * torch.sign(grad.detach())
    
    delta = x_adv_intermediate - x_natural_detached
    delta = torch.clamp(delta, -epsilon, epsilon)
    
    x_adv = (x_natural_detached + delta).clamp(0.0, 1.0)
    x_adv = x_adv.detach()
    
    model.train() 
    optimizer.zero_grad()
    
    return x_adv

############################################ single_step ############################################



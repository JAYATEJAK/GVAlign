import os
import torch
import random
import numpy as np
import torch.nn as nn
cudnn_deterministic = True
import torch.nn.functional as F

def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)

def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_lamda_given(x, y, alpha, lamda_list):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    # if alpha > 0.:
    #     lam = np.random.beta(alpha, alpha)
    # else:
    #     lam = 1.
    # breakpoint()
    # lam =lamda_list[y]
    lam_values = []

# Iterate over the elements of y
    for y_scalar in y:
        # Convert y_scalar to a Python integer
        y_index = int(y_scalar.item())
        
        # Access the corresponding element from lamda_list and append it to lam_values
        lam1 = lamda_list[y_index]
        lam_values.append(lam1)

    lam = torch.tensor(lam_values)

    lam = lam.reshape(-1, 1).cuda()
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    # mixed_x = lam * x + (1 - lam) * x[index,:]
    mixed_x = x * lam[:, None, None] + x[index, :] * (1 - lam[:, None, None])
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def add_noise_based_on_scale(x, y, alpha, lamda_list):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    # if alpha > 0.:
    #     lam = np.random.beta(alpha, alpha)
    # else:
    #     lam = 1.
    # breakpoint()
    # lam =lamda_list[y]
    lam_values = []

# Iterate over the elements of y
    for y_scalar in y:
        # Convert y_scalar to a Python integer
        y_index = int(y_scalar.item())
        
        # Access the corresponding element from lamda_list and append it to lam_values
        lam1 = lamda_list[y_index]
        lam_values.append(lam1)

    lam = torch.tensor(lam_values)

    lam = lam.reshape(-1, 1).cuda()
    # breakpoint()
    # batch_size = x.size()[0]
    noise = torch.randn(x.shape).cuda() * (1 - lam[:, None, None])
    # index = torch.randperm(batch_size).cuda()
    # mixed_x = lam * x + (1 - lam) * x[index,:]
    # mixed_x = x * lam[:, None, None] + x[index, :] * (1 - lam[:, None, None])
    # y_a, y_b = y, y[index]
    # breakpoint()
    return x + noise


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.2, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target, s):
        # print('hi')
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        # print(F.cross_entropy(self.s*output, target, weight=self.weight))
        return F.cross_entropy(s*output, target, weight=self.weight)

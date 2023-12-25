import torch.nn as nn
from torch.nn import Module, Parameter
import torch
from utils import mixup_data, mixup_data_lamda_given, add_noise_based_on_scale
import random
import torch.nn.functional as F
import numpy as np
import random
import sys, os
__all__ = ['resnet32']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        self.fc_int = nn.Linear(64 * block.expansion, num_classes) ##########  intermediate classifier nodes
        
        
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'
        self.global_variance = Parameter(torch.zeros(1, 64)) #### single global variance
        # print('checking this network')
        # print(c)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, stochatic=False, cs_stoc=None, manifold_mixup=None, layer_mix=None, target=None, lamda_norm_list = None, mixup_alpha = 1.0):
        # breakpoint()
        if manifold_mixup:
            print(c)
            if layer_mix == None:
                # layer_mix = random.randint(0,3)
                layer_mix = 0
            out = x
            # noise_mix = random.randint(0,1)
            # if noise_mix ==1:
            # out  = add_noise_based_on_scale(out, target, mixup_alpha, lamda_norm_list)
            out = torch.cat([out, out])
            # out1, y_a, y_b, lam = mixup_data(x, target, mixup_alpha)
            # out = torch.cat([x, out1])
            
            
            if layer_mix == 0:
                #out = lam * out + (1 - lam) * out[index,:]
                # out1, y_a, y_b, lam = mixup_data(out[x.shape[0]:], target, mixup_alpha)
                out1, y_a, y_b, lam = mixup_data_lamda_given(out[x.shape[0]:], target, mixup_alpha, lamda_norm_list)
                # out  = add_noise_based_on_scale(out[:x.shape[0]], target, mixup_alpha, lamda_norm_list)
                
                out = torch.cat([out[:x.shape[0]], out1])
                # out = torch.cat([out, out1])
            #print (out)       
            
            out = F.relu(self.bn1(self.conv1(out)))
            
            out = self.layer1(out)
    
            if layer_mix == 1:
                #out = lam * out + (1 - lam) * out[index,:]
                # out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
                # out1, y_a, y_b, lam = mixup_data(out[x.shape[0]:], target, mixup_alpha)
                # out1, y_a, y_b, lam = mixup_data_lamda_given(out[x.shape[0]:], target, mixup_alpha, lamda_norm_list)
                # out  = add_noise_based_on_scale(out[:x.shape[0]], target, mixup_alpha, lamda_norm_list)
                out = torch.cat([out[:x.shape[0]], out1])
                # out = torch.cat([out, out1])
            
            #print (out)

            out = self.layer2(out)
    
            if layer_mix == 2:
                #out = lam * out + (1 - lam) * out[index,:]
                # out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
                out1, y_a, y_b, lam = mixup_data(out[x.shape[0]:], target, mixup_alpha)
                # out1, y_a, y_b, lam = mixup_data_lamda_given(out[x.shape[0]:], target, mixup_alpha, lamda_norm_list)
                out  = add_noise_based_on_scale(out[:x.shape[0]], target, mixup_alpha, lamda_norm_list)
                # out = torch.cat([out[:x.shape[0]], out1])
                out = torch.cat([out, out1])
           #print (out)

            out = self.layer3(out)
            
            if layer_mix == 3:
                #out = lam * out + (1 - lam) * out[index,:]
                # out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
                # out1, y_a, y_b, lam = mixup_data(out[x.shape[0]:], target, mixup_alpha)
                # out1, y_a, y_b, lam = mixup_data_lamda_given(out[x.shape[0]:], target, mixup_alpha, lamda_norm_list)
                # out  = add_noise_based_on_scale(out[:x.shape[0]], target, mixup_alpha, lamda_norm_list)
                out = torch.cat([out[:x.shape[0]], out1])
                # out = torch.cat([out, out1])
            #print (out)

            # out = self.layer4(out)
            
            # if layer_mix == 4:
            #     #out = lam * out + (1 - lam) * out[index,:]
            #     out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            #print (out)
            # out = F.avg_pool2d(out, 4)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            
            if layer_mix == 4:
                #out = lam * out + (1 - lam) * out[index,:]
                # out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
                # out1, y_a, y_b, lam = mixup_data(out[x.shape[0]:], target, mixup_alpha)
                # out1, y_a, y_b, lam = mixup_data_lamda_given(out[x.shape[0]:], target, mixup_alpha, lamda_norm_list)
                # out  = add_noise_based_on_scale(out[:x.shape[0]], target, mixup_alpha, lamda_norm_list)
                out = torch.cat([out[:x.shape[0]], out1])
                # out = torch.cat([out, out1])
            
            # lam = torch.tensor(lam).cuda()
            # lam = lam.repeat(y_a.size())

            return out, y_a, y_b, lam

        else:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


def resnet32(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    # change n=3 for ResNet-20, and n=9 for ResNet-56
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model

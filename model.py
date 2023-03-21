from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.modules.batchnorm import BatchNorm2d
import copy
from utils_1 import *


'''
Quantization Network
'''
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class BasicBlock_Q(nn.Module):
    expansion = 1
    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act1 = Activate(self.a_bit)

        self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                '''
                For CIFAR10 ResNet paper uses option A.
                '''
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    #Conv2d_Q_(self.w_bit, in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    #SwitchBatchNorm2d(self.w_bit, self.expansion * planes)
                    ## Full-precision
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        self.act2 = Activate(self.a_bit)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # x used here
        out = self.act2(out)
        return out

# ResNet code modified from original of [https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py]
# Modified version for our experiment.
class ResNet20_Q(nn.Module):
    def __init__(self, a_bit, w_bit, block, num_blocks, num_classes=10, expand=1): 
        super().__init__()
        self.in_planes = 16 # Resnet

        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act = Activate(self.a_bit)

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False),
            SwitchBatchNorm2d(self.w_bit, 16),
            Activate(self.a_bit),
            
            *self._make_layer(block, 16, num_blocks[0], stride=1),
            *self._make_layer(block, 32, num_blocks[1], stride=2),
            *self._make_layer(block, 64, num_blocks[2], stride=2),
        )

        # mask_prune(self.layers)
        self.fc = nn.Linear(64, num_classes) 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # Full precision
            # option is 'A': Use F.pad
            # option is 'B': Use Conv+BN
            layers.append(block(self.a_bit, self.w_bit, self.in_planes, planes, stride, option='B'))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 




class Wide_BasicBlock_Q(nn.Module):
    expansion = 1
    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1):
        super().__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act1 = Activate(self.a_bit)
        self.act2 = Activate(self.a_bit)


        self.dropout = nn.Dropout(0.3) # p = 0.3

        
        
        if in_planes == planes:
            self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, padding=(1,1), stride=stride, bias=False)

        elif in_planes != planes:
            self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, padding=(1,1), stride=stride, bias=False)

        self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)
        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=3, padding=(1,1), stride=1, bias=False) 
        self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(self.expansion * in_planes),
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                )

    def forward(self, x):
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.dropout(out)
        out = self.act2(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)  # x used here
        return out


class Wide_ResNet_Q(nn.Module):
    def __init__(self, a_bit, w_bit, block, num_blocks, scale, num_classes=10): 
        super().__init__()

        self.in_planes = 16
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act = Activate(self.a_bit)
        nStages = [16, 16*scale, 32*scale, 64*scale]
        self.bn1 = SwitchBatchNorm2d(self.w_bit, nStages[3])
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, nStages[0], kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(),
            
            *self._make_layer(block, nStages[1], num_blocks[0], stride=1), 
            *self._make_layer(block, nStages[2], num_blocks[1], stride=2),
            *self._make_layer(block, nStages[3], num_blocks[2], stride=2),
        )

        # mask_prune(self.layers)
        self.fc = nn.Linear(nStages[3], num_classes) 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # Full precision
            # option is 'A': Use F.pad
            # option is 'B': Use Conv+BN
            layers.append(block(self.a_bit, self.w_bit, self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = self.act(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 
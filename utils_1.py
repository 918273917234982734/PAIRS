from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.modules.batchnorm import BatchNorm2d

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def SDK (image_col, image_row, filter_col, filter_row, in_channel, out_channel, \
                    array_row, array_col) :
    
    row_vector = filter_row * filter_col * in_channel
    col_vector = out_channel
    
    used_row = math.ceil(row_vector/array_row)
    used_col = math.ceil(col_vector/array_col)
    
    new_array_row = array_row * used_row
    new_array_col = array_col * used_col

    cycle = []
    w = [] 
    w.append(filter_row*filter_col)
    cycle.append(used_row*used_col*(image_row-filter_row+1)*(image_col-filter_col+1))
    
    i=0
    while True :
        i += 1
        pw_row = filter_row + i - 1 
        pw_col = filter_col + i - 1
        pw = pw_row * pw_col
        if pw*in_channel <= new_array_row and i * i * out_channel <= new_array_col :
            parallel_window_row = math.ceil((image_row - (filter_row + i) + 1)/i) + 1
            parallel_window_col = math.ceil((image_col - (filter_col + i) + 1)/i) + 1
            
            if parallel_window_row * parallel_window_row * used_row * used_col <= cycle[0] :
                del cycle[0]
                del w[0]
                cycle.append(parallel_window_row * parallel_window_col * used_row * used_col)
                w.append(pw)
        else :
            break
        
    return  cycle, math.sqrt(w[0]), math.sqrt(w[0])



class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification.
    Refer to https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/loss_ops.py
    """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()
        return cross_entropy_loss
        
class Activate(nn.Module):
    def __init__(self, a_bit, quantize=True):
        super(Activate, self).__init__()
        self.abit = a_bit
        # Since ReLU is not differentible at x=0, changed to GELU
        #self.acti = nn.ReLU(inplace=True)
        self.acti = nn.GELU()
        self.quantize = quantize
        if self.quantize:
            self.quan = activation_quantize_fn(self.abit)

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if self.quantize:
            x = self.quan(x)
        return x

class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        self.abit = a_bit
        assert self.abit <= 8 or self.abit == 32

    def forward(self, x):
        if self.abit == 32:
            activation_q = x
        else:
            activation_q = qfn.apply(x, self.abit)
        return activation_q


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**k - 1)
        # out = torch.round(input * n) / n
        out = torch.round(input*n)
        out = torch.where(out==0, out, out/n)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        self.wbit = w_bit
        # assert (self.wbit <= 8) or (self.wbit == 32) ################## 수정?

    def forward(self, x):
        if self.wbit == 32:
            #E = torch.mean(torch.abs(x)).detach()
            E = torch.nanmean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            # Commented because this has potential 'Divide by zero' issue, which results in NaN.
            #weight = weight / torch.max(torch.abs(weight))
            weight = torch.where(weight == 0, weight, weight / torch.max(torch.abs(weight)))
            weight_q = weight * E
        else:
            #E = torch.mean(torch.abs(x)).detach()
            E = torch.nanmean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            # Commented because this has potential 'Divide by zero' issue, which results in NaN.
            #weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
            weight = torch.where(weight == 0, weight+0.5, weight / 2 / torch.max(torch.abs(weight)) + 0.5)
            weight_q = 2 * qfn.apply(weight, self.wbit) - 1
            weight_q = weight_q * E
            
        return weight_q


class SwitchBatchNorm2d(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, w_bit, num_features):
        super(SwitchBatchNorm2d, self).__init__()
        self.w_bit = w_bit
        self.bn_dict = nn.ModuleDict()
        # for i in self.bit_list:
        #     self.bn_dict[str(i)] = nn.BatchNorm2d(num_features)
        self.bn_dict[str(w_bit)] = nn.BatchNorm2d(num_features, eps=1e-4)

        self.abit = self.w_bit
        self.wbit = self.w_bit
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')

    def forward(self, x):
        x = self.bn_dict[str(self.abit)](x)
        return x

class SwitchBatchNorm2d_(SwitchBatchNorm2d) : 
    def __init__(self, w_bit, num_features) :
        super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, w_bit=w_bit)
        self.w_bit = w_bit      
        # return SwitchBatchNorm2d_
    


def batchnorm2d_fn(w_bit):
    class SwitchBatchNorm2d_(SwitchBatchNorm2d):
        def __init__(self, num_features, w_bit=w_bit):
            super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, w_bit=w_bit)

    return SwitchBatchNorm2d_


class Conv2d_Q(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_Q, self).__init__(*args, **kwargs)


class Conv2d_Q_(Conv2d_Q): ## original
    def __init__(self, w_bit, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                    bias=False):
        super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)

    def forward(self, input):
        weight_q = self.quantize_fn(self.weight) 
        return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)



# def conv2d_quantize_fn(w_bit):
#     class Conv2d_Q_(Conv2d_Q):
#         def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
#                      bias=True):
#             super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
#                                             bias)
#             self.bit_list = w_bit
#             # self.bit_list = bit_list
#             # self.w_bit = self.bit_list[-1]
#             self.quantize_fn = weight_quantize_fn(self.bit_list)

#         def forward(self, input, order=None):
#             weight_q = self.quantize_fn(self.weight)
#             return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

#     return Conv2d_Q_


class Linear_Q(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear_Q, self).__init__(*args, **kwargs)

class Linear_Q_(Linear_Q): ## 만든거
    def __init__(self, w_bit, in_features, out_features, bias=True):
        super(Linear_Q_, self).__init__(in_features, out_features, bias=bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)

    def forward(self, input, order=None):
        weight_q = self.quantize_fn(self.weight)
        return F.linear(input, weight_q, self.bias)

# def linear_quantize_fn(w_bit):
#     class Linear_Q_(Linear_Q):
#         def __init__(self, in_features, out_features, bias=True):
#             super(Linear_Q_, self).__init__(in_features, out_features, bias)
#             self.w_bit = w_bit
#             self.quantize_fn = weight_quantize_fn(self.w_bit)

#         def forward(self, input):
#             weight_q = self.quantize_fn(self.weight)
#             return F.linear(input, weight_q, self.bias)

#     return Linear_Q_



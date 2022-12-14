from __future__ import print_function
import os
import argparse
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
from torchvision import datasets, transforms
import utils
from utils import *
from tqdm import tqdm
import pickle
from function import vwsdk, counting

##### train, test, retrain function #######################################################
def train(args, model, device, pattern_set, train_loader, test_loader, optimizer, Z, Y, U, V):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = admm_loss(args, device, model, Z, Y, U, V, output, target)
        loss.backward()
        optimizer.step()


def ttest(args, model, device, test_loader, pattern_set, Z, Y, U, V):
    model.eval()
    test_loss = 0
    loss_c = 0
    loss_z = 0
    loss_y = 0
    correct = 0
    with torch.no_grad():   # No Training
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.cross_entropy(output, target).item()
            loss_c += admm_lossc(args, device, model, Z, Y, U, V, output, target).item()
            loss_z += admm_lossz(args, device, model, Z, Y, U, V, output, target).item()
            loss_y += admm_lossy(args, device, model, Z, Y, U, V, output, target).item()
           
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    loss_c /= len(test_loader.dataset)
    loss_z /= len(test_loader.dataset)
    loss_y /= len(test_loader.dataset)
    prec = 100. * correct / len(test_loader.dataset)
    print('Accuracy : {:.2f}, Cross Entropy: {:f}, Z loss: {:f}, Y loss: {:f}'.format(prec, loss_c, loss_z, loss_y))
    #print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset), prec))

    return prec


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    prec = 100. * correct / len(test_loader.dataset)
    print('Accuracy: {:.2f}, Cross Entropy: {:f}'.format(prec, test_loss))

    return prec


def retrain(args, model, mask, device, train_loader, test_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = regularized_nll_loss(args, model, F.log_softmax(output, dim=1), target)
        loss = regularized_nll_loss(args, model, output, target)
        loss.backward()
        optimizer.prune_step(mask)


##### Settings #########################################################################
parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--model',      default='resnet18',         help='select model')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--dataset',    default='cifar10',          help='select dataset')
parser.add_argument('--batchsize',  default=256, type=int,      help='set batch size')
parser.add_argument('--lr',         default=6e-5, type=float,   help='set learning rate')
parser.add_argument('--re_lr',      default=1e-4, type=float,   help='set fine learning rate')
parser.add_argument('--alpha',      default=5e-4, type=float,   help='set l2 regularization alpha')
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='adam epsilon')
parser.add_argument('--rho',        default=6e-1, type=float,   help='set rho')
#parser.add_argument('--rho',        default=1000, type=float,   help='set rho')
parser.add_argument('--connect_perc',  default=1, type=float, help='connectivity pruning ratio')
parser.add_argument('--epoch',      default=40, type=int,      help='set epochs')
parser.add_argument('--re_epoch',   default=10, type=int,       help='set retrain epochs')
parser.add_argument('--num_sets',   default='8', type=int,      help='# of pattern sets')
parser.add_argument('--exp',        default='test', type=str,   help='test or not')
parser.add_argument('--l2',         default=False, action='store_true', help='apply l3 regularization')
parser.add_argument('--scratch',    default=False, action='store_true', help='start from pretrain/scratch')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--GPU', type=int, default=2) 
parser.add_argument('--method', default = 'patdnn', type = str, help='patdnn, ours, ours_fixed')
parser.add_argument('--withoc', type=int, default=1) 
parser.add_argument('--mask', type=int, default=5)
parser.add_argument('--gradual', type=int, default=0) 
parser.add_argument('--ar', type=int, default=512)
parser.add_argument('--ac', type=int, default=512)
args = parser.parse_args()
print(args)
comment = "check11_patdnn"

candidate = args.mask

GPU_NUM = args.GPU # GPU
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print ('Current cuda device ', torch.cuda.current_device()) # check

if args.exp == 'test':
    args.exp = f'{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
args.save = f'logs/{args.dataset}/{args.model}/{args.exp}_lr{str(args.lr)}_rho{str(args.rho)}_{comment}'

args.workers = 4

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
##########################################################################################################

print('Preparing pre-trained model...')
if args.dataset == 'imagenet':
    pre_model = torchvision.models.vgg16(pretrained=True)
elif args.dataset == 'cifar10':
    pre_model = torchvision.models.resnet18(pretrained=True)
    # pre_model = utils.__dict__[args.model]()
    # pre_model.load_state_dict(torch.load('./cifar10_pretrain/vgg16_bn.pt'), strict=True)
print("pre-trained model:\n", pre_model)


##### Find Pattern Set #####
if args.method == 'patdnn' :
    print('\nFinding Pattern Set...')
    # if os.path.isfile('pattern_set_'+ args.dataset + '.npy') is False:
    #     pattern_set = pattern_setter(pre_model, candidate)
    #     np.save('pattern_set_'+ args.dataset + '.npy', pattern_set)
    # else:
    #     pattern_set = np.load('pattern_set_'+ args.dataset + '.npy')
    pattern_set = pattern_setter(pre_model, candidate)

    pattern_set = pattern_set[:args.num_sets, :]
    print(pattern_set)
    print('pattern_set loaded')

# elif args.method == 'ours' :






##### Load Dataset ####
print('\nPreparing Dataset...')
train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)
print('Dataset Loaded')


##### Load Model #####
# model = utils.__dict__[args.model]()
model = pre_model

# if pre-trained... load pre-trained weight
if not args.scratch:
    state_dict = pre_model.state_dict()
    torch.save(state_dict, 'tmp_pretrained.pt')

    model.load_state_dict(torch.load('tmp_pretrained.pt'), strict=True)
model.cuda()
pre_model.cuda()


# History collector
history_score = np.zeros((200, 2))
his_idx = 0



print('patdnn')
print('lr:', args.lr, 'rho:', args.rho)
print('\nTraining...') ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
best_prec = 0
Z, Y, U, V = initialize_Z_Y_U_V(model)

#WarmUp
#optimizer = PruneAdam(model.named_parameters(), lr=1e-6, eps=args.adam_epsilon)
optimizer = PruneAdam(model.named_parameters(), lr=1e-5, eps=args.adam_epsilon)
for epoch in range(5):
    train(args, model, device, pattern_set, train_loader, test_loader, optimizer, Z, Y, U, V)
       
    X = update_X(model)
    Z = update_Z(X, U, pattern_set, args.withoc)
    Y = update_Y(X, V, args)

    prec = ttest(args, model, device, test_loader, pattern_set, Z, Y, U, V)
    history_score[his_idx][0] = epoch
    history_score[his_idx][1] = prec
    his_idx += 1
 
# Optimizer
optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)

for epoch in range(args.epoch):
    if epoch in [args.epoch//4, args.epoch//2, args.epoch//4*3]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    
    train(args, model, device, pattern_set, train_loader, test_loader, optimizer, Z, Y, U, V)
       
    X = update_X(model)
    Z = update_Z(X, U, pattern_set, args.withoc)
    Y = update_Y(X, V, args)
    U = update_U(U, X, Z)
    V = update_V(V, X, Y)

    prec = ttest(args, model, device, test_loader, pattern_set, Z, Y, U, V)
    history_score[his_idx][0] = epoch
    history_score[his_idx][1] = prec
    his_idx += 1
    
create_exp_dir(args.save)
torch.save(model.state_dict(), os.path.join(args.save, 'cifar10_before.pth.tar'))


# Real Pruning ! ! !
print("\nApply Pruning with connectivity & pattern set...")
mask = apply_prune(args, model, device, pattern_set, args.withoc)
print_prune(model)

# for name, param in model.named_parameters():
#     if name.split('.')[-1] == "weight" and len(param.shape)==4:
#         param.data.mul_(mask[name])

torch.save(model.state_dict(), os.path.join(args.save, 'cifar10_after.pth.tar'))

images = [32, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 4, 4, 4]


idx = -1
total_skipped_rows = 0
layer6464_skipped_rows = 0
layer128128_skipped_rows = 0
layer256256_skipped_rows = 0
layer512512_skipped_rows = 0
for name, param in model.named_parameters():
    if name.split('.')[-1] == "weight" and len(param.shape)==4 and 'downsample' not in name:
        if name[:5] == 'conv1' :
            continue
        rows = 0
        if idx != -1 :
            oc, ic, kr, kh = mask[name].shape
            _, _, _, _, pw_r, pw_h = vwsdk(images[idx], images[idx], kr, kh, ic, oc, args.ar, args.ac)
            print(f'idx : {idx}, pwr : {pw_r}, pwh : {pw_h}')
            print(mask[name].shape)
        
            # pwr.append(pw_r)
            # pwh.append(pw_h)
            if pw_r > kr :
                mode_ = True
                rows = counting(mask[name], mask[name].shape, pw_r, pw_h, mode_)
                print(f'layer name = {name}, # offed rows = {rows}')
                if 0 <= idx and idx < 4 :
                    layer6464_skipped_rows += rows
                elif 5 <= idx and idx < 8 :
                    layer128128_skipped_rows += rows
                elif 9 <= idx and idx < 12 :
                    layer256256_skipped_rows += rows
                elif 13 <= idx and idx < 16 :
                    layer512512_skipped_rows += rows

                if idx != 4 or idx != 8 or idx != 12 or idx != 16 :
                    total_skipped_rows += rows


            param.data.mul_(mask[name])
        idx += 1


print("="*100)
print(f'epochs = {epoch+1}, # layer6464 offed rows = {layer6464_skipped_rows}')
print(f'epochs = {epoch+1}, # layer128128 offed rows = {layer128128_skipped_rows}')
print(f'epochs = {epoch+1}, # layer256256 offed rows = {layer256256_skipped_rows}')
print(f'epochs = {epoch+1}, # layer512512 offed rows = {layer512512_skipped_rows}')      
print(f'epochs = {epoch+1}, # total offed rows = {total_skipped_rows}')
print("="*100)


print("\ntesting...")
test(args, model, device, test_loader)


# Optimizer for Retrain
optimizer = PruneAdam(model.named_parameters(), lr=args.re_lr, eps=args.adam_epsilon)


# Fine-tuning...
print("\nfine-tuning...")
best_prec = 0
for epoch in range(args.re_epoch):
    if epoch in [args.re_epoch//4, args.re_epoch//2, args.re_epoch//4*3]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    retrain(args, model, mask, device, train_loader, test_loader, optimizer)

    prec = test(args, model, device, test_loader)
    history_score[his_idx][0] = epoch
    history_score[his_idx][1] = prec
    his_idx += 1
    
    if prec > best_prec:
        best_prec = prec
        torch.save(model.state_dict(), os.path.join(args.save, 'cifar10_pruned.pth.tar'))

np.savetxt(os.path.join(args.save, 'train_record.txt'), history_score, fmt='%10.5f', delimiter=',')

print('patdnn lr:', args.lr, 'rho:', args.rho)

############################################

# my mistake 1 - making mask.pickle
"""
with open('mask.pickle', 'wb') as fw:
    pickle.dump(mask, fw)
with open('mask.pickle', 'rb') as fr:
    mask = pickle.load(fr)
    print("mask loaded")
"""

# my mistake 2
"""
for module in model.named_modules():
    if isinstance(module[1], nn.Conv2d):
        print("module:", module[0])
        prune.custom_from_mask(module, 'weight', mask=mask[module[0] +'.weight'])
"""
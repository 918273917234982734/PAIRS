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
from function import *
from utils_1 import *
import logging
import random
from model import *

import torch.backends.cudnn as cudnn


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
        loss = regularized_nll_loss(args, model, output, target)
        loss.backward()
        optimizer.step()
        
        # optimizer.prune_step(mask)



parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--model',      default='WRN16-4_Q',          help = 'select model : ResNet20_Q, WRN16-4_Q')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--dataset',    default='cifar100',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=1e-2, type=float,   help='set learning rate') # 6e-5
parser.add_argument('--re_lr',      default=1e-3, type=float,   help='set fine learning rate')
parser.add_argument('--alpha',      default=5e-4, type=float,   help='set l2 regularization alpha')
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='adam epsilon')
parser.add_argument('--rho',        default=1, type=float,   help='set rho') # original 6e-1?
#parser.add_argument('--rho',        default=1000, type=float,   help='set rho')
# parser.add_argument('--rho',        default=5e-3, type=float,   help='set rho')
parser.add_argument('--connect_perc',  default=1, type=float, help='connectivity pruning ratio')
parser.add_argument('--epoch',      default=1, type=int,      help='set epochs') # 60
parser.add_argument('--re_epoch',   default=1, type=int,       help='set retrain epochs') # 100
parser.add_argument('--num_sets',   default='4', type=int,      help='# of pattern sets')
parser.add_argument('--exp',        default='test', type=str,   help='test or not')
parser.add_argument('--l2',         default=False, action='store_true', help='apply l3 regularization')
parser.add_argument('--scratch',    default=False, action='store_true', help='start from pretrain/scratch')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--method', default = 'random', type = str, help='patdnn, ours, random, pconv')
parser.add_argument('--withoc', type=int, default=1) 
parser.add_argument('--GPU', type=int, default=2) 
parser.add_argument('--mask', type=int, default=5)
parser.add_argument('--ar', type=int, default=512)
parser.add_argument('--ac', type=int, default=512)
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=2)
parser.add_argument('--seed', type=int, default=1992)
args = parser.parse_args()
print(args)

# candidate = 9 - args.mask + 1
candidate = args.mask
print(f'Actual used array columns = {args.ac}')

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)

GPU_NUM = args.GPU # GPU
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print ('Current cuda device ', device)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if args.withoc == 0:
    pat_dim = 'kernel-wise'
elif args.withoc == 1:
    pat_dim = 'array-wise'
else :
    pat_dim = 'block-wise'

directory_save = './log/%s/%s/%s/%s/%s'%(args.model, args.dataset, args.seed, pat_dim, args.method)
if not os.path.isdir(directory_save):
    os.makedirs(directory_save)

file_name = directory_save + '/%dx%d_mask%d.log'%(args.ar, args.ac, args.mask)

file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



if args.exp == 'test':
    args.exp = f'{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'

args.workers = 2

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


print('Preparing pre-trained model...')
if args.model == 'ResNet20_Q':
    pre_model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=10).cuda()
elif args.model == 'WRN16-4_Q':
    pre_model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=4, num_classes=100).cuda()



pt_save = './pt_save/%s/%s'%(args.model, args.dataset)
if args.method == 'patdnn' :
    print('pretrained model uploaded...')
    pre_model.load_state_dict(torch.load(pt_save+'/trained.pt'), strict=False)
    # print("pre-trained model:\n", pre_model)

elif args.method == 'pruned_sparse' :
    print('pretrained model uploaded...')
    pre_model.load_state_dict(torch.load('./pre_trained_1119_sparse.pt'), strict=False)
    # print("pre-trained model:\n", pre_model)
else :
    print('new model uploaded...')


##### Load Dataset ####
print('\nPreparing Dataset...')
train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)
print('Dataset Loaded')

mask_dic = './pattern_set'
if not os.path.isdir(directory_save):
    os.makedirs(directory_save)


##### Find Pattern Set #####
print('\nFinding Pattern Set...')
if args.method == 'patdnn' :
    if os.path.isfile(directory_save + '/mask_' + str(args.mask) + '.npy') is False:
        pattern_set = pattern_setter(pre_model, candidate)
        np.save(directory_save + '/mask_' + str(args.mask) + '.npy', pattern_set)
    else:
        pattern_set = np.load(directory_save + '/mask_' + str(args.mask) + '.npy')

    pattern_set = pattern_set[:args.num_sets, :]

elif args.method == 'pruned_sparse' :
    if os.path.isfile('pattern_set_'+ args.dataset + '_mask_' + str(args.mask) + '_0729_v.npy') is False:
        pattern_set = pattern_setter_v(pre_model, candidate)
        np.save('pattern_set_'+ args.dataset + '_mask_' + str(args.mask) + '_0729_v.npy', pattern_set)
    else:
        pattern_set = np.load('pattern_set_'+ args.dataset + '_mask_' + str(args.mask) + '_0729_v.npy')

    pattern_set = pattern_set[:args.num_sets, :]


elif args.method == 'random' :
    candi_list = []
    ran_list = [0,1,2,3,4,5,6,7,8]
    for i in range(args.num_sets) :
        one_list = [1,1,1,1,1,1,1,1,1]
        sample = random.sample(ran_list, args.mask)
        for sam in sample :
            one_list[sam] = 0
        candi_list.append(one_list)
    
    pattern_set = np.array(candi_list)

elif args.method == 'original' :
    pattern_set = np.array([[1,1,1,1,1,1,1,1,1]])

elif args.method == 'pconv' :
    pattern_set = np.array([[0,1,0,1,1,1,0,0,0], [0,1,0,1,1,0,0,1,0], [0,1,0,0,1,1,0,1,0], [0,0,0,1,1,1,0,1,0]])



if args.method == 'ours' :
    if args.mask == 5 : # 4
        pattern_set = np.array([[0,0,0,0,1,1,0,1,1], [0,0,0,1,1,0,1,1,0], [1,1,0,1,1,0,0,0,0], [0,1,1,0,1,1,0,0,0]])
    elif args.mask == 3 : # 4
        pattern_set = np.array([[0,0,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,0,0], [1,1,0,1,1,0,1,1,0], [0,1,1,0,1,1,0,1,1]])
    elif args.mask == 1: # 4
        pattern_set = np.array([[0,1,1,1,1,1,1,1,1], [1,1,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,1,1], [1,1,1,1,1,1,1,1,0]])
    elif args.mask == 2 : # 14
        pattern_set = np.array([
                                [0,0,1,1,1,1,1,1,1], [1,0,0,1,1,1,1,1,1], [0,1,1,0,1,1,1,1,1], [1,1,0,1,1,0,1,1,1],
                                [1,1,1,0,1,1,0,1,1], [1,1,1,1,1,1,0,0,1], [1,1,1,1,1,1,1,0,0], [1,1,1,1,1,0,1,1,0],
                                [0,1,0,1,1,1,1,1,1], [1,1,0,1,1,1,1,1,0], [1,1,1,1,1,1,0,1,0], [0,1,1,1,1,1,0,1,1],
                                [0,1,1,1,1,1,1,1,0], [1,1,0,1,1,1,0,1,1]
                                ])
    elif args.mask == 4 :# 16
        pattern_set = np.array([
                                [0,0,0,0,1,1,1,1,1], [0,0,0,1,1,0,1,1,1], [0,0,0,1,1,1,0,1,1], [0,0,0,1,1,1,1,1,0],
                                [1,1,1,1,1,0,0,0,0], [1,1,1,0,1,1,0,0,0], [0,1,1,1,1,1,0,0,0], [1,1,0,1,1,1,0,0,0],
                                [0,1,1,0,1,1,0,0,1], [0,0,1,0,1,1,0,1,1], [0,1,0,0,1,1,0,1,1], [0,1,1,0,1,1,0,1,0],
                                [1,1,0,1,1,0,1,0,0], [1,0,0,1,1,0,1,1,0], [0,1,0,1,1,0,1,1,0], [1,1,0,1,1,0,0,1,0]
                                ])

pattern_set_1 = pattern_set

print(pattern_set)
print('Patternset is loaded...')
                    

if args.model == 'ResNet20_Q' :
    if args.dataset == 'cifar10' :
        model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=10).cuda()
    elif args.dataset == 'cifar100' :
        model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=100).cuda()

elif args.model == 'WRN16-4_Q' :
    if args.dataset == 'cifar10' :
        model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=4, num_classes=10).cuda()
    elif args.dataset == 'cifar100' :
        model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=4, num_classes=100).cuda()

print('Model is loaded...')

print('\nTraining...') ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
best_prec = 0
Z, Y, U, V = initialize_Z_Y_U_V(model)

#WarmUp                                            
optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("-"*50)
if args.model == 'ResNet20_Q':
    images = [32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8]
else:
    images = [32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8]

pwr = []
pwh = []

idxx = -1
for name, param in model.named_parameters(): # name : weight, bias... # params : value list
    if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
        if name[:5] == 'conv1' :
            continue
        idxx += 1 
        output_c, input_c, kkr, kkh = param.shape
        _, pw_r, pw_h = SDK(images[idxx], images[idxx], kkr, kkh, input_c, output_c, args.ar, args.ac)
        pwr.append(pw_r)
        pwh.append(pw_h)


print('*'*100)
logger.info(f"Pattern set : {pattern_set}")
if args.method == 'ours' :
    logger.info(f"Pattern set_1 : {pattern_set_1}")


print("-"*50)
print('WarmUp for some epochs...')
print("-"*50)


for epoch in range(5):
    train(args, model, device, 0, train_loader, test_loader, optimizer, Z, Y, U, V)
       
    X = update_X(model)
    for ppww in range(len(pwr)) :
        if args.method == 'ours' :
            if ppww > 3 :
                Z=update_Z(X, U, pattern_set_1, args.withoc)
            else :
                Z=update_Z(X, U, pattern_set, args.withoc)
        else :
            Z=update_Z(X, U, pattern_set, args.withoc)
    Y = update_Y(X, V, args)

    prec = ttest(args, model, device, test_loader, pattern_set, Z, Y, U, V)



# Optimizer
optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
print("-"*50)
print('Lego training!!!')
print("-"*50)

for epoch in range(args.epoch+1):
    print(f"Current epochs : {epoch}")
    # if epoch in [args.epoch//2, (args.epoch * 3)//4]: ############################################################################
    # if epoch in [args.epoch//4, args.epoch//2, args.epoch//4*3]:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1
    
    train(args, model, device, 0, train_loader, test_loader, optimizer, Z, Y, U, V)
       
    X = update_X(model)
    for ppww in range(len(pwr)) :
        if args.method == 'ours' :
            if ppww > 3 :
                Z=update_Z(X, U, pattern_set_1, args.withoc)
            else :
                Z=update_Z(X, U, pattern_set, args.withoc)
        else :
            Z=update_Z(X, U, pattern_set, args.withoc)

    Y = update_Y(X, V, args)
    U = update_U(U, X, Z)
    V = update_V(V, X, Y)

    prec = ttest(args, model, device, test_loader, 0, Z, Y, U, V)
    logger.info("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.epoch, prec))


# Real Pruning ! ! !
print("\nApply Pruning with connectivity & pattern set...")
mask = apply_prune_pat(args, model, device, pattern_set, pattern_set_1, pwr, args.withoc, args.method)
print_prune(model)

# Fine-tuning...
print("Retraining for fine tuning...")
logger.info("="*100)
logger.info("Retraining for fine tuning...")
best_prec = -1

# optimizer 2
# optimizer_1 = optim.SGD(model.parameters(), lr=args.re_lr, momentum=0.9, weight_decay= 5e-4)
optimizer_1 = torch.optim.Adam(model.parameters(), lr=args.re_lr, weight_decay=5e-4)
T = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T, 1e-4)

for epoch in range(args.re_epoch):
    print("Epoch: {} with re_lr: {}".format(epoch+1, args.re_lr))

    retrain(args, model, mask, device, train_loader, test_loader, optimizer_1)
    scheduler.step()

    print("\ntesting...")
    prec = test(args, model, device, test_loader)
    logger.info("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.re_epoch, prec))
        
    if prec > best_prec :
        best_prec = prec
        print('Best accuracy is updated! at epoch %d/%d: %.4f '%(epoch, args.re_epoch, best_prec))



idx = -1
total_skipped_rows = 0
layer1_skipped_rows = 0
layer2_skipped_rows = 0
layer3_skipped_rows = 0
for name, param in model.named_parameters():
    if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
        print(name)
        print(param.shape)
        if name[:5] == 'conv1' :
            continue
        rows = 0
        idx += 1
        if idx != -1 :
            oc, ic, kr, kh = mask[name].shape
            _, pw_r, pw_h = SDK(images[idx], images[idx], kr, kh, ic, oc, args.ar, args.ac)
            print(f'idx : {idx}, pwr : {pw_r}, pwh : {pw_h}')
        
            rows = counting(mask[name], mask[name].shape, pw_r, pw_h)
            print(f'layer name = {name}, # Skipped rows = {rows}')
            if args.model == 'ResNet20_Q':
                n = 5
                if 1 <= idx and idx <= n :
                    layer1_skipped_rows += rows
                elif n+1 <= idx and idx <= 2*n:
                    layer2_skipped_rows += rows
                elif 2*n+1 <= idx and idx <= 3*n :
                    layer3_skipped_rows += rows

            elif args.model == 'WRN16-4_Q':
                n = 3
                if 0 <= idx and idx <= n-1 :
                    layer1_skipped_rows += rows
                elif n <= idx and idx <= 2*n-1:
                    layer2_skipped_rows += rows
                elif 2*n <= idx and idx <= 3*n-1 :
                    layer3_skipped_rows += rows

layer1_skipped_rows = int(layer1_skipped_rows/n)
layer2_skipped_rows = int(layer2_skipped_rows/n)
layer3_skipped_rows = int(layer3_skipped_rows/n)

total_skipped_rows = layer1_skipped_rows + layer2_skipped_rows + layer3_skipped_rows

logger.info("="*60)
logger.info(f'pwr = {pwr[n-1]}, pwh = {pwh[n-1]}, Average skipped rows = {layer1_skipped_rows}')
logger.info(f'Ratio of row-skipping = {layer1_skipped_rows*100/(pwh[n-1]*pwh[n-1]*ic)}')
logger.info(f'pwr = {pwr[2*n-1]}, pwh = {pwh[2*n-1]}, Average skipped rows = {layer2_skipped_rows}')
logger.info(f'Ratio of row-skipping = {layer1_skipped_rows*100/(pwh[2*n-1]*pwh[2*n-1]*ic)}')
logger.info(f'pwr = {pwr[3*n-1]}, pwh = {pwh[3*n-1]}, Average skipped rows = {layer3_skipped_rows}')
logger.info(f'Ratio of row-skipping = {layer1_skipped_rows*100/(pwh[3*n-1]*pwh[3*n-1]*ic)}')

logger.info(f'# total average skipped rows = {total_skipped_rows}')
logger.info(f'Ratio of total row-skipping = {total_skipped_rows*100/((pwh[n-1]*pwh[n-1]+pwh[2*n-1]*pwh[2*n-1]+pwh[3*n-1]*pwh[3*n-1])*ic)}')
logger.info("-"*70)
logger.info(f'best acc = {best_prec}')
logger.info("-"*70)

print("="*100)
print(f'pwr = {pwr[n-1]}, pwh = {pwh[n-1]}, Average skipped rows = {layer1_skipped_rows}')
print(f'Ratio of row-skipping = {layer1_skipped_rows*100/(pwh[n-1]*pwh[n-1]*ic)}')
print(f'pwr = {pwr[2*n-1]}, pwh = {pwh[2*n-1]}, Average skipped rows = {layer2_skipped_rows}')
print(f'Ratio of row-skipping = {layer1_skipped_rows*100/(pwh[2*n-1]*pwh[2*n-1]*ic)}')
print(f'pwr = {pwr[3*n-1]}, pwh = {pwh[3*n-1]}, Average skipped rows = {layer3_skipped_rows}')
print(f'Ratio of row-skipping = {layer1_skipped_rows*100/(pwh[3*n-1]*pwh[3*n-1]*ic)}')

print(f'# total average skipped rows = {total_skipped_rows}')
print(f'Ratio of total row-skipping = {total_skipped_rows*100/((pwh[n-1]*pwh[n-1]+pwh[2*n-1]*pwh[2*n-1]+pwh[3*n-1]*pwh[3*n-1])*ic)}')
print("-"*70)
print(f'best acc = {best_prec}')
print("-"*70)
print("="*100)



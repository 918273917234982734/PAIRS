import os
import time

print('input the mode for training mode')
mode = int(input())

# seeds = [1992, 1106, 1016, 2023, 2022]
seeds = [231106, 231016, 232023, 232022, 231992]
'''
2048x1024
512x512
'''
if mode == 0 : 
    models = ['WRN16-4_Q']
    method = ['patdnn']
    dataset = ['cifar100'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    withoc_list = [2]
    wb = 2
    ar = 2048
    ac = 1024
    epoch = 40
    re_epoch = 100
    GPU = 0


elif mode == 1 : 
    models = ['WRN16-4_Q']
    method = ['random']
    dataset = ['cifar100'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    withoc_list = [2]
    wb = 2
    ar = 2048
    ac = 1024
    epoch = 40
    re_epoch = 100
    GPU = 1

elif mode == 2 : 
    models = ['WRN16-4_Q']
    method = ['ours']
    dataset = ['cifar100'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    withoc_list = [2]
    wb = 2
    ar = 2048
    ac = 1024
    epoch = 40
    re_epoch = 100
    GPU = 2

elif mode == 3 : 
    models = ['WRN16-4_Q']
    method = ['original']
    dataset = ['cifar100'] 
    mask_list = [1] # 1,3,5
    withoc_list = [2]
    wb = 2
    ar = 2048
    ac = 1024
    epoch = 40
    re_epoch = 100
    GPU = 3



for seed in seeds:
    for model in models:
        for dst in dataset :
            for name in method :
                for mask in mask_list :
                    for withoc in withoc_list :
                        if mask == 1 or mask == 3 or mask == 5:
                            numsets = 4
                        elif mask == 2:
                            numsets = 14
                        elif mask == 4:
                            numsets = 16

                        os.system('python3 main_loss3_v1.py' + ' --model ' + str(model) + ' --num_sets ' + str(numsets) + ' --dataset ' + str(dst) 
                        + ' --mask ' + str(mask) + ' --method ' + str(name) + ' --withoc ' + str(withoc) + ' --epoch ' + str(epoch) + ' --re_epoch ' + str(re_epoch)
                        + ' --wb ' + str(wb) + ' --GPU ' + str(GPU) + ' --ac ' + str(ac) + ' --ar ' + str(ar) + ' --seed ' + str(seed) )

                        time.sleep(10)


# for name in name_list :
#     for numsets in numsets_list :
#         for mask in mask_list :
#             for withoc in withoc_list :
#                 for wb in wb_list :
#                     os.system('python3 main_loss3_v1.py --lr 1e-3 --rho 1 --num_sets ' + str(numsets) 
#                     + ' --mask ' + str(mask) + ' --method ' + str(name) + ' --withoc ' + str(withoc) 
#                     + ' --wb ' + str(wb) + ' --gradual 1' + ' --GPU ' + str(GPU) + ' --epo ' + str(epo) + ' --ac ' + str(ac) + ' --ar ' + str(ar))
#                     time.sleep(10)




import os
import time

'''
lr = 0.001
batch = 512
epoch = 120
PIM array = 512x512
a-bit = 32 bit
b-bit = 2 bit
'''

"""
PLEASE CHECK THIS TEXT BEFORE RUNNING.
This code runs main_loss3_v1.py.

1. name_list :
    'original' - without pattern
    'patdnn'   - pattern set from the pretrained model
    'random'   - randomly choice the weight element to make it zero
    'pconv'    - pre-defined pattern set that is only available in the 4-entry
    'ours'     - PAIRS

2. masks : zero weight element
    They are equal
    1-mask = 8-entry
    2-mask = 7-entry
    3-mask = 6-entry
    4-mask = 5-entry
    5-mask = 4-entry

3. numsets_list :
    8-entry has 4 patterns 
    7-entry has 12 patterns 
    6-entry has 4 patterns
    5-entry has 16 patterns
    4-entry has 4 patterns

    if 8-, 6-, 4-entry :
        input 4
    elif 7-entry :
        input 12
    elif 5-entry :
        input 16

4. withoc : pattern dimension 
    0 : kernel-wise pattern 
    1 : array-wise pattern
    2 : block-wise pattern

5. wb_list : weight bit precision
    2 : 2-bit weight (default)

"""



print('input the mode for training mode')
mode = int(input())


if mode == 0 :
    name_list = ['original']
    numsets_list = [1]
    mask_list = [0]
    withoc_list = [1]
    wb_list = [2]
    ac = [512]
    GPU = 1

elif mode == 1 : 
    name_list = ['ours']
    numsets_list = [4] 
    mask_list = [1] # 1,3,5
    withoc_list = [1]
    wb_list = [2]
    ac = [512]
    ar = [512]
    GPU = 0



for name in name_list :
    for numsets in numsets_list :
        for mask in mask_list :
            for withoc in withoc_list :
                for wb in wb_list :
                    for acc in ac :
                        for arr in ar :
                            os.system('python3 main_loss3_v1.py --lr 1e-3 --rho 1 --num_sets ' + str(numsets) 
                            + ' --mask ' + str(mask) + ' --method ' + str(name) + ' --withoc ' + str(withoc) 
                            + ' --wb ' + str(wb) + ' --GPU ' + str(GPU) + ' --ac ' + str(acc) + ' --ar ' + str(arr) )

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




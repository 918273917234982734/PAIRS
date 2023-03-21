import math

import numpy as np

def vwsdk (image_col, image_row, filter_col, filter_row, in_channel, out_channel, \
                    array_row, array_col) :

    i = 0 # initialize # overlap col
    j = 1 # overlap row

    reg_pw =[]
    reg_total_cycle = [] # initialize
    reg_overlap_row = []
    reg_overlap_col = []
    reg_row_cycle = []
    reg_col_cycle = []
    reg_ICt = []
    reg_OCt = []
    
    while True :
        try :
            i += 1
            if (i + filter_col) > image_col : 
                i = 1
                j += 1
                if j + filter_row > image_row : 
                    break

            # for parallel_window computing
            reg_N_parallel_window_row = math.ceil((image_row - (filter_row + i) + 1)/i) + 1
            reg_N_parallel_window_col = math.ceil((image_col - (filter_col + j) + 1)/j) + 1
            
            # for cycle computing
            # Tiled IC
            if in_channel == 3 :
                ICt = math.floor(array_row /((filter_row + i - 1)*(filter_col + j - 1)))
                if ICt > in_channel :
                    ICt = 3
                row_cycle = math.ceil(in_channel / ICt)
            else :
                ICt = math.floor(array_row /((filter_row + i - 1)*(filter_col + j - 1)))
                row_cycle = math.ceil(in_channel / ICt)
            
            # Tiled OC
            OCt =  math.floor(array_col / (i * j))
            col_cycle = math.ceil(out_channel / OCt)
    
            reg_N_of_computing_cycle = reg_N_parallel_window_row * reg_N_parallel_window_col \
                                    * row_cycle * col_cycle
            
            if i == 1 : # initialize
                reg_pw.append(reg_N_parallel_window_row * reg_N_parallel_window_col)
                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)

            if reg_total_cycle[0] > reg_N_of_computing_cycle :
                del reg_pw[0]
                del reg_total_cycle[0]
                del reg_overlap_row[0]
                del reg_overlap_col[0]
                del reg_row_cycle[0]
                del reg_col_cycle[0]
                del reg_ICt[0]
                del reg_OCt[0]

                reg_pw.append(reg_N_parallel_window_row * reg_N_parallel_window_col)
                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)

    
        except ZeroDivisionError :
            continue

    return reg_pw[0], reg_total_cycle[0], reg_ICt[0], reg_OCt[0], filter_row - 1 + reg_overlap_row[0], filter_col - 1 + reg_overlap_col[0]


def SDK (image_col, image_row, filter_col, filter_row, in_channel, out_channel, array_row, array_col) :
    
    row_vector = filter_row * filter_col * in_channel
    col_vector = out_channel
    
    used_row = math.ceil(row_vector/array_row)
    used_col = math.ceil(col_vector/array_col)
    
    new_array_row = array_row * used_row
    new_array_col = array_col * used_col

    # initialize
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
        
    return cycle[0], int(math.sqrt(w[0])), int(math.sqrt(w[0]))


def counting (mask, layer_shape, pwr, pwh, mode = True) :
    mask = mask.cpu().numpy()
    OC, IC, kr, kh = layer_shape

    cnt = 0

    kernel = []
    for i in range(kr*kh) :
        kernel.append(i)
    
    for i in range(IC) :
        pw = []
        for j in range(pwr*pwh) :
            pw.append([])
            
        for a in range(pwh-kh+1) :
            for b in range(pwr-kr+1) :
                for c in range(len(kernel)) :
                    divider = c // 3
                    residue = c % 3
                    pw_idx = (divider+a)*pwr+(residue+b)
                    pw[pw_idx].append(kernel[c])
        
        zero_list = []
        for j in range(kr) :
            for k in range(kh) :
                cal = mask[:, i, j, k].sum()
                if cal == 0 :
                    idx = j*kr + k
                    zero_list.append(idx)

        for q in range(len(pw)) :
            for j in zero_list :
                if j in pw[q] :
                    pw[q].remove(j)

        for m in pw :
            if m == [] :
                cnt+=1

    if mode == True :
        print("="*60)
        for iccc in range(IC) :
            if iccc < 3 :
                print(mask[0][iccc])

    return cnt

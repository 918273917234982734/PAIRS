# PAIRS: Pruning-AIded Row-Skipping for In-Memory Convolutional Weight Mapping

## Abstract
Due to their energy-efficient computation, processing-in-memory (PIM) architecture is becoming a promising candidate for convolutional neural network (CNN) inference. Utilizing the weight sparsity of CNNs, computation of rows with zero-valued weights can be skipped to reduce the computing cycles. However, inherent sparsity of CNNs does not produce a high row-skipping ratio, since row-skipping requires the entire cells in a specific row to be zeros. In this paper, we propose pairing row-skipping
with pattern-based pruning to skip more rows of CNN inference computation on a PIM array. The proposed PAIRS (pruning-aided row-skipping) method first determines the pattern dimension that shares a certain pattern shape, considering the PIM array and channel size. Then, for each pattern dimension, PAIRS determines the pattern shape that maximizes the row-skip ratio and minimizes the accuracy loss. When the 6-entry pattern is used, the simulation with a 512×512 PIM array in ResNet-20 shows that compared to no pruning, our proposed method achieves the cycle reduction by 21.1% within 2% accuracy loss, while the prior work fails to reduce the computing cycles. 

## Requiredments
  Python3.x+
  Pytorch
 
## Usage
 ### This code is based on https://github.com/7bvcxz/PatDNN
 
 ### main_loss_v1.py
  #### This is a training code inclduing the network, pattern set and counting the number of skipped rows.
  
 ### run_script.py
  #### This code is for running main_loss_v1.py.
  
  #### Check a text before running this code.
  * You have to input below parameters.
  * Before the running, please check the python code file for the exact usage
  
  name_list : ["original", "patdnn", "pconv", "ours"] - define the pattern set
  numsets_list : [1, 4, 12, 16] - 
  mask_list : [1, 2, 3, 4, 5] - the number of masks that have zero
  withoc_list : [0, 1] - defined the pattern dimension (kernel-wise or array-wise 4D-shaped pattern)
  wb_list : [2] - weight bit-precision
  ac : [512] - the size of PIM array columns

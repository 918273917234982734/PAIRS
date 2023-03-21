# PAIRS: Pruning-AIded Row-Skipping for SDK-Based Convolutional Weight Mapping in Processing-In-Memory Architectures

## Abstract
Processing-in-memory (PIM) architecture is becoming a promising candidate for convolutional neural network (CNN) inference. A recent weight mapping method called shift and duplicate kernel (SDK) improves the utilization by the deployment of shifting the same kernels into idle columns. However, this method inevitably generates idle cells with an irregular distribution, which limits reducing the size of the weight matrix. To effectively compress the weight matrix in the PIM array, prior works have introduced a row-wise pruning scheme, one of the structured weight pruning schemes, that aims to skip the operation on a row by zeroing out all weight in the specific row (we call it row-skipping). However, due to the deployment of shifting kernels, SDK mapping complicates zeroing out all the weight in the same row. To address this issue, we propose pruning-aided row-skipping (PAIRS) that effectively reduces the number of rows of convolutional weights that are mapped with SDK mapping. By pairing the SDK mapping-aware pruning pattern design and row-wise pruning, PAIRS achieves a higher row-skipping ratio. In comparison to pruning methods, PAIRS achieves up to 1.95× rows skipped and 4× higher compression rate with similar or even better inference accuracy.


### This code is based on https://github.com/7bvcxz/PatDNN

### To train ResNet-20 on cifar10, try below codes
  #### main_loss3_v1.py 
    * base code
  #### run_script.py 
    * running for main_loss_v1.py with defining the parameters
  ### pattern_setter.py
    * several functions for generating the pattern

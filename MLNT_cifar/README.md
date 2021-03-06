# MLNT - Cifar

This code is an unofficial adapation of the *Meta-Learning based Noise-Tolerant Training (MLNT)* CVPR Paper:

**Learning to Learn from Noisy Labeled Data**
 <a href="https://arxiv.org/pdf/1812.05214.pdf">[pdf]</a>
  <a href="https://github.com/LiJunnan1992/MLNT">[original code github]</a>  
Junnan Li, Y. Wong, Qi Zhao, M. Kankanhalli
To use the code, please cite our paper.


The aim of the adaptation is to enable the original code to work on the Cifar-10 dataset instead of the Clothing1M.  

The main differences are importing a resnet-32 and adapting the model to enable an optimum synthethic label generation.
However we take care of every detail as generating noise, visualizing the results and setting a config.py file with the hyper-parameters.

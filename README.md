# Auxiliary-Training

Implementation of CVPR paper "Auxiliary Training: Towards Accurate and Compact Models."
Please download the CIFAR-C dataset from https://github.com/hendrycks/robustness

The folder should be like this

```
Auxiliary-Training
---./CIFARC
------brightness.npy
------contrast.npy
------defocus_blur.npy
----- ...
----- ...
```

Then, you can start auxiliary training by

```
python train.py
```

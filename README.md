# DF-Net: Dynamic and Folding Network for 3D Point Cloud Completion
This repository is forked and modified from Pointr(https://github.com/yuxumin/PoinTr).

This repository contains PyTorch implementation for DF-Net: Dynamic and Folding Network for 3D Point Cloud Completion
If you have any questions about the code, please email me. Thanks!

The requirements and installation process is the same as that of Pointr.

1)Dataset:

ShapeNet-13 can be found in PF-Net(https://github.com/zztianzz/PF-Net-Point-Fractal-Network)

ShapeNet-55 can be found in Pointr(https://github.com/yuxumin/PoinTr)

2)Train and Evaluate on ShapeNet-13:

python main.py --gan --config ./cfgs/ShapeNet13_models/DGNet.yaml --exp name trainDFNet



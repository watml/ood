# Out-of-distribution detection with flow generative models
This code repo corresponds to the ICLR 2022 paper: [Revisiting flow generative models for Out-of-distribution detection](https://openreview.net/forum?id=6y2KBh-0Fd9).  

## Datasets
All datasets can be downloaded by torchvision.dataset without extra effort, except the following three:
+ CelebA. Please download the dataset as indicated in https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. Put all annotation files as well as img_align_celeba.zip (~ 1.34 GB) under the folder /data/CELEBA/celeba (which is the default path), then unzip img_align_celeba.zip. If you already have the dataset, you can simply change the path to your exisiting dataset by modifiying the path in the function **setDataset** in main.py.  
+ LSUN. Please download the test set (test_lmdb.zip, ~172 MB) as instructed by https://github.com/fyu/lsun. Put it under the default path: /data/LSUN, then unzip it.  
+ EEG/ECG. Please download slp03.edf, slp04.edf, slp14.edf, slp16.edf from MIT-BIH Polysomnographic Database, and put them under path /data/PSG. One way to do so is going to https://archive.physionet.org/cgi-bin/atm/ATM, then do the following steps: Input -> Database -> *MIT-BIH Polysomnographic Database (slpdb)*,  Output -> Length -> *to end*, Toolbox -> *Export signals as EDF*. Finally you can download the EDF file below.   

## Environments
This project is implemented based on Python 3.6, PyTorch 1.1.0, and torchvision 0.3.0.
The rest required packages may not necessarily have to match exactly.  

`matplotlib==3.3.4`  
`mne==0.23.4`  
`numpy==1.16.4`  
`pandas==1.1.5`  
`scikit_learn==0.24.2`  
`scipy==1.5.4`  
`seaborn==0.11.1`  
`tqdm==4.64.0`  

## Command lines
Please refer to the following specifications to train different flow models on different in-distribution dataset.
### Training simple models:  
`python3 main.py  --dataset fmnist  --lr 0.00001  --ind fmnist  --estimator GLOW --Train --epochs 201 --K 3 --num_blocks 2  --hidden_size 64  --batch_size 64`  
`python3 main.py  --dataset fmnist  --lr 0.000001  --ind fmnist --estimator REALNVP  --Train  --epochs 51   --num_blocks 1  --hidden_size 512`  
`python3 main.py  --dataset cifar10  --lr 0.001  --ind cifar10  --estimator GLOW --Train --epochs 2 --K 3 --num_blocks 3 --hidden_size 64  --batch_size 64`  
`python3 main.py  --dataset cifar10  --lr 0.00001  --ind cifar10 --estimator REALNVP  --Train  --epochs 100  --num_blocks 1  --hidden_size 2048`  
`python3 main.py  --dataset cifar100  --lr 0.001  --ind cifar100  --estimator GLOW --Train --epochs 5 --K 3 --num_blocks 3 --hidden_size 64  --batch_size 64`   
`python3 main.py  --dataset cifar100  --lr 0.00001  --ind cifar100 --estimator REALNVP  --Train  --epochs 50  --num_blocks 1  --hidden_size 2048`  
`python3 main.py  --dataset svhn  --lr 0.001  --ind svhn  --estimator GLOW --Train --epochs 200 --K 3 --num_blocks 3 --hidden_size 64  --batch_size 64`   
`python3 main.py  --dataset svhn  --lr 0.00001  --ind svhn --estimator REALNVP  --Train  --epochs 100  --num_blocks 1  --hidden_size 2048`  
`python3 main.py  --dataset celeba  --lr 0.001  --ind celeba  --estimator GLOW --Train --epochs 200 --K 3 --num_blocks 3 --hidden_size 64  --batch_size 64`   
`python3 main.py  --dataset celeba  --lr 0.00001  --ind celeba --estimator REALNVP  --Train  --epochs 50  --num_blocks 1  --hidden_size 2048`  
### Training complex models:
`python3 main.py  --dataset cifar10  --lr 0.001  --ind cifar10  --estimator GLOW --Train --epochs 201 --K 16 --num_blocks 3 --hidden_size 128  --batch_size 64`    
`python3 main.py  --dataset celeba  --lr 0.00001  --ind celeba  --estimator REALNVP  --Train  --epochs 201  --num_blocks 16  --hidden_size 512  --log_step 10`  

### Testing
To reproduce our main results, run the following example commands for CIFAR10/SVHN pairs with batch size = 10 (for other dataset pairs you can simply parse different dataset names to --dataset and --ind):  
#### Table 1 and 5
`python3 main.py --dataset svhn  --ind cifar10  --estimator REALNVP  --Test  --num_blocks 1 --hidden_size 2048  --kst_rule  --batch_size 10  --num_project 50`  
`python3 main.py --dataset svhn  --ind cifar10  --estimator REALNVP  --Test  --num_blocks 1 --hidden_size 2048  --klod  --batch_size 10`  
`python3 main.py --dataset svhn  --ind cifar10  --estimator REALNVP  --Test  --num_blocks 1 --hidden_size 2048  --typical  --batch_size 10`  
#### Table 6  
`python3 main.py  --dataset svhn  --ind cifar10  --estimator GLOW --Test  --K 3 --num_blocks 3 --hidden_size 64  --kst_rule  --batch_size 10  --num_project 200`  
`python3 main.py  --dataset svhn  --ind cifar10  --estimator GLOW --Test  --K 3 --num_blocks 3 --hidden_size 64  --klod  --batch_size 10`  
`python3 main.py  --dataset svhn  --ind cifar10  --estimator GLOW --Test  --K 3 --num_blocks 3 --hidden_size 64  --typical  --batch_size 10`

### Generating images
If you set --log_step 10 (or other steps for saving models), you can generate images by loading differently trained models as follows (the default num_epochs is -1, which means the last training epoch):  
`python3 generateImgs.py --dataset celeba  --ind celeba --estimator REALNVP  --num_blocks 16  --hidden_size 512  --num_epochs 10`  
`python3 generateImgs.py --dataset celeba  --ind celeba --estimator REALNVP  --num_blocks 16  --hidden_size 512  --num_epochs 100`  
`python3 generateImgs.py --dataset celeba  --ind celeba --estimator REALNVP  --num_blocks 16  --hidden_size 512  --num_epochs 150`  
`python3 generateImgs.py --dataset celeba  --ind celeba --estimator REALNVP  --num_blocks 16  --hidden_size 512  --num_epochs 200`  

## Reference
The PyTorch implementation of models used in this project is based on existing public code repo:
+ RealNVP (https://github.com/ikostrikov/pytorch-flows)
+ Glow (https://github.com/y0ast/Glow-PyTorch)
+ LSA (https://github.com/aimagelab/novelty-detection)

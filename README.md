# **Out-of-distribution detection with flow generative models**
This code repo corresponds to the ICLR 2022 paper: [Revisiting flow generative models for Out-of-distribution detection](https://openreview.net/forum?id=6y2KBh-0Fd9).

## Datasets
All datasets can be downloaded by torchvision.dataset without extra effort, except the following two:
+ CelebA. Please download the dataset as indicated in https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. Put all annotation files as well as img_align_celeba.zip (~ 1.34 GB) under the folder /data/CELEBA/celeba (which is the default path), then unzip img_align_celeba.zip. If you already have the dataset, you can simply change the path to your exisiting dataset by modifiying the path in the function **setDataset** in main.py.
+ LSUN. Please download the test set as instructed by https://github.com/fyu/lsun. Put it under the default path: /data/LSUN, then unzip it.

## Environments
This project is implemented based on Python 3.6, PyTorch 1.1.0, and torchvision 0.3.0.
The rest required packages may not necessarily have to match exactly.
'matplotlib==3.3.4'  
'mne==0.23.4'
> numpy==1.16.4
> pandas==1.1.5
> scikit_learn==0.24.2
> scipy==1.5.4
> seaborn==0.11.1
> tqdm==4.64.0

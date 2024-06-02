# DRCT: Saving Image Super-resolution away from Information Bottleneck

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-set5-4x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set5-4x-upscaling?p=drct-saving-image-super-resolution-away-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-urban100-4x)](https://paperswithcode.com/sota/image-super-resolution-on-urban100-4x?p=drct-saving-image-super-resolution-away-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-set14-4x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set14-4x-upscaling?p=drct-saving-image-super-resolution-away-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-manga109-4x)](https://paperswithcode.com/sota/image-super-resolution-on-manga109-4x?p=drct-saving-image-super-resolution-away-from)


### DRCT [[Paper Link]](https://arxiv.org/abs/2404.00722) [[Project Page]](https://allproj002.github.io/drct.github.io/) [[Poster]](https://drive.google.com/file/d/1zR9wSwqCryLeKVkJfTuoQILKiQdf_Vdz/view?usp=sharing)

[Chih-Chung Hsu](https://cchsu.info/), [Chia-Ming Lee](https://ming053l.github.io/), [Yi-Shiuan Chou](https://scholar.google.com/citations?&user=iGX8FBcAAAAJ)

Advanced Computer Vision LAB, National Cheng Kung University

## Overview

<img src=".\figures\overview.png" width="500"/>

<img src=".\figures\drct_fix.gif" width="600"/>

<img src=".\figures\4.png" width="400"/>

**Benchmark results on SRx4 without x2 pretraining. Mulit-Adds are calculated for a 64x64 input.**
| Model | Params | Multi-Adds | Forward | FLOPs | Set5 | Set14 | BSD100 | Urban100 | Manga109 |
|:-----------:|:---------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [HAT](https://github.com/JingyunLiang/SwinIR) |   9.62M    | 11.22G | 2053M | 42.18G | 33.04 | 29.23 | 28.00 | 27.97 | 32.48 |
| DRCT |   10.44M  | 5.92G | 1857M | 7.92G | 33.11 | 29.35 | 28.18 | 28.06 | 32.59 |
| [HAT-L](https://github.com/JingyunLiang/SwinIR) |   40.84M    | 76.69G | 5165M | 79.60G | 33.30 | 29.47 | 28.09 | 28.60 | 33.09 |
| [DRCT-L](https://drive.google.com/file/d/1bVxvA6QFbne2se0CQJ-jyHFy94UOi3h5/view?usp=sharing) |  27.58M    | 9.20G | 4278M | 11.07G | 33.37 | 29.54 | 28.16 | 28.70 | 33.14 |


## Updates

- ✅ 2024-03-31: Release the first version of the paper at Arxiv.
- ✅ 2024-04-14: DRCT is accepted by NTIRE 2024, CVPR.
- ✅ 2024-06-02: The pretrained model is released. 
- ❌ 2024-06-02: MambaDRCT is released. [MODEL.PY](https://drive.google.com/file/d/1di4XKslSxkDyl8YeQ284qp3vDx3zP0ZL/view?usp=sharing)
  * (Training process for DRCT + Mamba is very long, if you are interested about it, you can try to optimize/fit it up. Due to resources contraints, we do not intend to fix it.)
- Real_DRCT_GAN will be released.

## Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8!!! It would cause abnormal performance.)**
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 
### Installation
```
git clone https://github.com/ming053l/DRCT.git
conda create --name drct python=3.8 -y
conda activate drct
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd DRCT
pip install -r requirements.txt
python setup.py develop
```
## How To Test

- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
- Then run the following codes (taking `DRCT_SRx4_ImageNet-pretrain.pth` as an example):
```
python drct/test.py -opt options/test/DRCT_SRx4_ImageNet-pretrain.yml
```
The testing results will be saved in the `./results` folder.  

- Refer to `./options/test/DRCT_SRx4_ImageNet-LR.yml` for **inference** without the ground truth image.

**Note that the tile mode is also provided for limited GPU memory when testing. You can modify the specific settings of the tile mode in your custom testing option by referring to `./options/test/DRCT_tile_example.yml`.**

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- Validation data can be download at [this page](https://github.com/ChaofWang/Awesome-Super-Resolution/blob/master/dataset.md).
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 drct/train.py -opt options/train/train_DRCT_SRx2_from_scratch.yml --launcher pytorch
```

The training logs and weights will be saved in the `./experiments` folder.

## Citations
#### BibTeX
    @misc{hsu2024drct,
      title={DRCT: Saving Image Super-resolution away from Information Bottleneck}, 
      author={Chih-Chung Hsu and Chia-Ming Lee and Yi-Shiuan Chou},
      year={2024},
      eprint={2404.00722},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

## Thanks
A part of our work has been facilitated by the [HAT](https://github.com/XPixelGroup/HAT) framework, and we are grateful for its outstanding contribution.

## Contact
If you have any question, please email zuw408421476@gmail.com to discuss with the author.

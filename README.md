# DRCT: Saving Image Super-resolution away from Information Bottleneck 

### ✨✨ [CVPR NTIRE Oral Presentation]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-set5-4x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set5-4x-upscaling?p=drct-saving-image-super-resolution-away-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-urban100-4x)](https://paperswithcode.com/sota/image-super-resolution-on-urban100-4x?p=drct-saving-image-super-resolution-away-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-set14-4x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set14-4x-upscaling?p=drct-saving-image-super-resolution-away-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-manga109-4x)](https://paperswithcode.com/sota/image-super-resolution-on-manga109-4x?p=drct-saving-image-super-resolution-away-from)

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ming053l/DRCT/issues)



## [[Paper Link]](https://arxiv.org/abs/2404.00722) [[Project Page]](https://allproj002.github.io/drct.github.io/) [[Poster]](https://drive.google.com/file/d/1zR9wSwqCryLeKVkJfTuoQILKiQdf_Vdz/view?usp=sharing) [[Model zoo]](https://drive.google.com/drive/folders/1QJHdSfo-0eFNb96i8qzMJAPw31u9qZ7U?usp=sharing) [[Visual Results]](https://drive.google.com/drive/folders/15raaESdkHD-7cHWBVDzDitTH8_h5_0uE?usp=sharing) [[Slide]](https://docs.google.com/presentation/d/1MxPPtgQZ61GFSr3YfGOm9scm23bbbXRj/edit?usp=sharing&ouid=105932000013245886245&rtpof=true&sd=true) [[Video]](https://drive.google.com/file/d/17dB47E8I2ME-shhxAWDlQCyCuJRn79d_/view?usp=sharing)

[Chih-Chung Hsu](https://cchsu.info/), [Chia-Ming Lee](https://ming053l.github.io/), [Yi-Shiuan Chou](https://nelly0421.github.io/)

Advanced Computer Vision LAB, National Cheng Kung University

## Overview (SwinIR with Dense Connection)

- Background and Motivation

In CNN-based super-resolution (SR) methods, dense connections are widely considered to be an effective way to preserve information and improve performance. (introduced by RDN / RRDB in ESRGAN...etc.)

However, SwinIR-based methods, such as HAT, CAT, DAT, etc., generally use Channel Attention Block or design novel and sophisticated Shift-Window Attention Mechanism to improve SR performance. These works ignore the information bottleneck that information flow will be lost deep in the network.

- Main Contribution

Our work simply adds dense connections in SwinIR to improve performance and re-emphasizes the importance of dense connections in Swin-IR-based SR methods. Adding dense-connection within deep-feature extraction can stablize information flow, thereby boosting performance and keeping lightweight design (compared to the SOTA methods like HAT).



<img src=".\figures\overview.png" width="500"/>

<img src=".\figures\drct_fix.gif" width="600"/>

<img src=".\figures\4.png" width="400"/>

**Benchmark results on SRx4 without x2 pretraining. Mulit-Adds are calculated for a 64x64 input.**
| Model | Params | Multi-Adds | Forward | FLOPs | Set5 | Set14 | BSD100 | Urban100 | Manga109 | Training Log |
|:-----------:|:---------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [HAT](https://github.com/XPixelGroup/HAT) |   20.77M    | 11.22G | 2053M | 42.18G | 33.04 | 29.23 | 28.00 | 27.97 | 32.48 | - |
| [DRCT](https://drive.google.com/file/d/1jw2UWAersWZecPq-c_g5RM3mDOoc_cbd/view?usp=sharing) |   14.13M  | 5.92G | 1857M | 7.92G | 33.11 | 29.35 | 28.18 | 28.06 | 32.59 | - |
| [HAT-L](https://github.com/XPixelGroup/HAT) |   40.84M    | 76.69G | 5165M | 79.60G | 33.30 | 29.47 | 28.09 | 28.60 | 33.09 | - |
| [DRCT-L](https://drive.google.com/file/d/1bVxvA6QFbne2se0CQJ-jyHFy94UOi3h5/view?usp=sharing) |  27.58M   | 9.20G | 4278M | 11.07G | 33.37 | 29.54 | 28.16 | 28.70 | 33.14 | - |
| [DRCT-XL (pretrained on ImageNet)](https://drive.google.com/file/d/1uLGwmSko9uF82X4OPOMw3xfM3stlnYZ-/view?usp=sharing) |  -  | - | - | - | 32.97 / 0.91 | 29.08 / 0.80  | - | - | - | [log](https://drive.google.com/file/d/1kl2r9TbQ8TR-sOdzvCcOZ9eqNsmIldGH/view?usp=drive_link)


**Real DRCT GAN SRx4. (Updated)**

| Model | Training Data | Checkpoint | Log |
|:-----------:|:---------:|:-------:|:--------:|
| [Real-DRCT-GAN_MSE_Model](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing) | [DF2K + OST300](https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost/code)  | [Checkpoint](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing) |  [Log](https://drive.google.com/file/d/1kl2r9TbQ8TR-sOdzvCcOZ9eqNsmIldGH/view?usp=drive_link) | 
| [Real-DRCT-GAN_Finetuned from MSE](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing) | [DF2K + OST300](https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost/code)  |  [Checkpoint](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing)  |  [Log](https://drive.google.com/file/d/15aBV-FFi7I4esUb1vzRmrjMccc5cEEY4/view?usp=drive_link) | 

## Real DRCT GAN Comparison (Thanks to [Phhofm](https://github.com/Phhofm)!)

The images below demonstrate the enhancement capabilities of the 4xRealWebPhoto_v4_drct-l model compared to standard 4x Nearest Neighbor upscaling:

Showcase:
[Slow.pic](https://slow.pics/s/VOKVChT9) link as interactive comparison with a Slider

<img src=".\figures\real-drct.png" width="1000"/>

## Updates

- ✅ 2024-03-31: Release the first version of the paper at Arxiv.
- ✅ 2024-04-14: DRCT is accepted by NTIRE 2024, CVPR.
- ✅ 2024-06-02: The pretrained DRCT-L is released.
- ❌ 2024-06-02: MambaDRCT is released. [MODEL.PY](https://drive.google.com/file/d/1di4XKslSxkDyl8YeQ284qp3vDx3zP0ZL/view?usp=sharing)
  * Training process for DRCT + [MambaIR](https://github.com/csguoh/MambaIR) is very slow, if you are interested about it, you can try to optimize/fit it up. This may be caused by the Recurrent nature of mambaIR, coupled with the feature map reusing from DRCT, which causes the training speed to be too slow (it may also be a problem with the equipment we use or the package version).
  * We try to combine DRCT with SS2D in mambaIR. However, the CUDA version of our GPU cannot be updated to the latest version, which leads to difficulties in installing the package and optimizing the training speed. So we don't plan to continue fixting MambaDRCT. If you are interested, you are welcome to use this code.
- ✅ 2024-06-09: The pretrained DRCT model is released. [model zoo](https://drive.google.com/drive/folders/1QJHdSfo-0eFNb96i8qzMJAPw31u9qZ7U?usp=sharing)
- ✅ 2024-06-11: We have received a large number of requests to release pre-trained models and training records from ImageNet for several downstream applications, please refer to the following link:
  
[[Training log on ImageNet]](https://drive.google.com/file/d/1kl2r9TbQ8TR-sOdzvCcOZ9eqNsmIldGH/view?usp=drive_link) [[Pretrained Weight (without fine-tuning on DF2K)]](https://drive.google.com/file/d/1uLGwmSko9uF82X4OPOMw3xfM3stlnYZ-/view?usp=sharing)

- ✅ 2024-06-12: DRCT have been selected for oral presentation in NTIRE!
- ✅ 2024-06-14: We have received a large number of requests to release Feature maps and LAM Visualization, please refer to *./Visualization/*.
- 2024-06-24: DRCT-v2 is on the development.
- ✅ 2024-07-08: Update the inference (with half precision) file, big thanks for @zelenooki87!
- ✅ 2024-12-04: Update the mismatched part after re-training (SRx2). (Please see Arxiv)
- ✅ 2024-12-04: Update the Model architecture description errors. (Please see Arxiv)
- ✅ 2024-12-04: Update the Real-DRCT-GAN in google drive.
  
## Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8 and **1.12** !!! It would cause abnormal performance.)**
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
## How To Inference on your own Dataset?

```
python inference.py --input_dir [input_dir ] --output_dir [input_dir ]  --model_path[model_path]
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

If our work is helpful to your reaearch, please kindly cite our work. Thank!

#### BibTeX
    @misc{hsu2024drct,
      title={DRCT: Saving Image Super-resolution away from Information Bottleneck}, 
      author = {Hsu, Chih-Chung and Lee, Chia-Ming and Chou, Yi-Shiuan},
      year={2024},
      eprint={2404.00722},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
    @InProceedings{Hsu_2024_CVPR,
      author    = {Hsu, Chih-Chung and Lee, Chia-Ming and Chou, Yi-Shiuan},
      title     = {DRCT: Saving Image Super-Resolution Away from Information Bottleneck},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      month     = {June},
      year      = {2024},
      pages     = {6133-6142}
    }

## Thanks
A part of our work has been facilitated by [HAT](https://github.com/XPixelGroup/HAT), [SwinIR](https://github.com/JingyunLiang/SwinIR), [LAM](https://github.com/XPixelGroup/X-Low-level-Interpretation) framework, and we are grateful for their outstanding contributions.

A part of our work are contributed by @zelenooki87, thanks for your big contributions and suggestions!

Special thanks to [Phhofm](https://github.com/Phhofm) for providing the 4xRealWebPhoto_v4_drct-l model, which has significantly enhanced our image processing capabilities. The model is available at [Phhofm/models](https://github.com/Phhofm/models/releases/tag/4xRealWebPhoto_v4_drct-l).

## Contact
If you have any question, please email zuw408421476@gmail.com to discuss with the author.

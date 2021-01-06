# Visual_Recognition_HW4
This is howework 4 for selected topics in visual recongnition using deep learning. The goal is single image super-resolution for given test images. 

I use residual channel attention network (RCAN) to upscale images with scale factor of 3.

It propose residual in residual (RIR) structure and use channel attention (CA) mechanism to achieve great perfomance. 
 
For the given testing dataset, it can achieve **25.964 PSNR** evaluted by TAs.

The details please refer to original paper [RCAN](https://arxiv.org/pdf/1807.02758.pdf).

**Important: the implementation is based on [EDSR-Pytorch](https://github.com/thstkdgus35/EDSR-PyTorch), which provides implementations of several super-resolution models.**

## Hardware
The following specs were used to train and test the model:
- Ubuntu 16.04 LTS
- 2x RTX 2080 with CUDA=10.1

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Inference](#inference)
5. [Make Submission](#make-submission)

## Installation
Run the following command to install the required packages:

```shell
conda create -n SR
conda activate SR

# install requirements
conda install python=3.6
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install numpy
conda install scikit-image
conda install imageio
conda install matplotlib
conda install tqdm
conda install -c menpo opencv
```

## Dataset Preparation
Download the dataset from the [Google drive](https://drive.google.com/drive/u/0/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x) which is provided by TAs.

Then, unzip them and put it under the **./datasets/** directory.

Next, to create a low resolution counterparts for training images, run the following commands:
```shell
cd EDSR-PyTorch/src
python lrgenerator.py 
```

Hence, the data directory is structured as:
```
./datasets/
  +- testing_lr_images/
  |  +- 00.png
  |  +- 01.png...
  +- training_hr_images/
  |  +- 2092.png
  |  +- 8049.png...
  +- training_lr_images/
  |  +- X3
     |  +- 2092x3.png
     |  +- 8049x3.png...
```


## Training
**Important: This step is optional. If you don't want to retrain the whole model, please download [my trained weights](https://drive.google.com/file/d/1-kKSOut5vV5O8Ou9VNU4rirI5tZ7t8Ob/view?usp=sharing).**

- **model_best.pt**: my weights trained on given training dataset (provided by TAs) for 400 epochs with early termination (stop at 359th epochs). 

Then, move **model_best.pt** to the **./EDSR-PyTorch/experiment/RCAN_BIX3_G10R20P24_RS/model/** directory.

Hence, the weights directory is structured as:
```
./EDSR-PyTorch/
  +- experiment/
  |  +- RCAN_BIX3_G10R20P24_RS/
     |  +- model/
         |  +- model_best.pt
  +- src/
  +- ...
```

### Train model from scratch on the given dataset (optional)
P.S. If you don't want to spend two days training a model, you can skip this step and just use the **model_best.pt** I provided to inference. 

Now, let's  train the RCAN on given training dataset:

1. please ensure ./EDSR-PyTorch/experiment/RCAN_BIX3_G10R20P24_RS/model/model_best.pt exists.

2. run the following training command:

```
cd ./EDSR-PyTorch/src/
python main.py --template RCAN --save RCAN_BIX3_G10R20P24_RS --scale 3 --reset --save_results --patch_size 72 --epochs 400 --n_GPUs 2 --batch_size 64
```

It takes about two days to train the model on 2 RTX 2080 GPUs.

Finally, we can find the best weights **model_best.pt** in **./EDSR-PyTorch/experiment/RCAN_BIX3_G10R20P24_RS/model/** directory.


## Inference
With the testing dataset and trained model, you can run the following commands to obtain the predicted results:

```
python main.py --data_test Demo --scale 3 --pre_train ../experiment/RCAN_BIX3_G10R20P24_RS/model/model_best.pt  --test_only --save_results --self_ensemble --template RCAN
```

After that, you will get predicted high resolution counterparts for test images in **./EDSR-PyTorch/experiment/test/results-Demo** folder.


## Make Submission
1. rename all predicted high resolution images to be the same as low resolution ones, for example, "00_x3_SR.png -> 00.png"

2. submit all predicted high resolution counterparts to [here](https://drive.google.com/drive/folders/1sbb527to9S8Ej-25QOb0IrQ-d2TDBcYK).

**Note**: The repo has provided **psnr_25.964_0856610** which is my submission of predicted high resolution results with **25.964 PSNR**.



# Fashion-MNIST meets SiamMask

SiamMask is a semi supervised object segmentation Siamese network. In this adaptation, it is used as a front end for segmentation of clothing accessories on a catwalk. The back end is comprised of pre-processing of the segmentation mask, inference of my Keras (Tensorflow backend) shallow CNN (a multiclass classifier) trained on fashion MNIST dataset and postprocessing with YOLO-style bounding box results display.

This is my forked version of the official implementation of SiamMask by Wang et al. For technical details on SiamMask, such as training (note ~10h on 4 Tesla V100 GPUs of the base model), evaluation (VOT, Youtube-VOS, DAVIS), and running inference tests please refer to:

**[[Paper](https://arxiv.org/abs/1812.05050)] [[Video](https://youtu.be/I_iOVrcpEBw)] [[Project Page](http://www.robots.ox.ac.uk/~qwang/SiamMask)]** <br />

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fast-online-object-tracking-and-segmentation/visual-object-tracking-vot201718)](https://paperswithcode.com/sota/visual-object-tracking-vot201718?p=fast-online-object-tracking-and-segmentation)


## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Pretrained SiamMask models](#pretrained-models)

## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, Keras 2.1.4, Tensorflow 1.13.1, ffmpeg 2.8.15

- Clone the repository 
```
git clone https://github.com/foolwood/SiamMask.git && cd SiamMask
export SiamMask=$PWD
```
- Setup python environment
```
conda create -n siammask python=3.6
source activate siammask
pip install -r requirements.txt
bash make.sh
```
- Add the project to your PYTHONPATH
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Demo
- [Setup](#environment-setup) your environment
- Download the SiamMask model
```shell
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Run `demo.py`

```shell
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json
```

<div align="center">
  <img src="http://www.robots.ox.ac.uk/~qwang/SiamMask/img/SiamMask_demo.gif" width="500px" />
</div>


## Pretrained SiamMask models

- Download pretrained models
```shell
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT_LD.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
.

## License
Licensed under an MIT license.


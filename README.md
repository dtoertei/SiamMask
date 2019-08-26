# Fashion-MNIST meets SiamMask

SiamMask is a semi supervised object segmentation Siamese network. In this adaptation, it is used as a front end for segmentation of clothing accessories on a catwalk. The back end is comprised of pre-processing of the segmentation mask, inference of my Keras (Tensorflow backend) shallow CNN trained on fashion MNIST dataset and postprocessing with YOLO-style bounding box results display. <br />

This repository is my forked version of the official implementation of SiamMask by Wang et al. For technical details on SiamMask, such as training (note ~10h on 4 Tesla V100 GPUs of the base model), evaluation (VOT, Youtube-VOS, DAVIS), and running inference tests please refer to: <br />

**[[Paper](https://arxiv.org/abs/1812.05050)] [[Video](https://youtu.be/I_iOVrcpEBw)] [[Project Page](http://www.robots.ox.ac.uk/~qwang/SiamMask)]** <br />



## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Testing](#testing)


## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, Keras 2.1.4, Tensorflow 1.13.1, ffmpeg 2.8.15

- Clone the repository 
```
git clone https://github.com/dtoertei/SiamMask.git && cd SiamMask
export SiamMask=$PWD
```
- Setup python environment
```
conda create -n siammask python=3.6
source activate siammask
pip install -r requirements.txt
bash make.sh
```
- Install ffmpeg
```shell
sudo add-apt-repository ppa:jonathonf/ffmpeg-3
sudo apt update
sudo apt install ffmpeg libav-tools x264 x265
```
- Add the project to your PYTHONPATH
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Demo
- [Setup](#environment-setup) your environment
- No need to download MNIST CNN, it is located under $SiamMask/experiments/fashion_mnist_cnn
- Download the SiamMask model
```shell
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- To reproduce *Dress*/*Sandal*/*Top* results
```shell
rm -r $SiamMask/data/fashionMNIST/test_frames/*.png
cp -r $SiamMask/data/fashionMNIST/ReproduceResults/Dress/*.png $SiamMask/data/fashionMNIST/test_frames
```
- Run demo_MNIST.py
```shell
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../tools/demo_MNIST.py --resume SiamMask_DAVIS.pth --config config_davis.json
```
When prompted, draw a rectangular RoI around dress/sandal/top and hit the return key.

- Generating GIF 
```shell
cd $SiamMask/experiments/siammask_sharp
python ../../data/process_mnist.py
```
GIF file will be stored in $SiamMask/data/fashion_MNIST/GIFs directory.

<div align="center">
  <img src="http://www.robots.ox.ac.uk/~qwang/SiamMask/img/SiamMask_demo.gif" width="500px" />
</div>


## Testing
- [Setup](#environment-setup) your environment
- Extract video frames
```shell
rm -r $SiamMask/data/fashionMNIST/test_frames/*.png
rm -r $SiamMask/data/fashionMNIST/frames
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../data/process_mnist.py --split
```
This will use the provided $SiamMask/data/fashion_MNIST/fashion_show.mp4 as input video file by default. If you wish to test with your own video, you must set flags: i) --split_dir for your video file location and ii) --split_name for your video file name.
The resulting frames will be extracted to *frames* directory.

- Copy sequence of frames of interest
```shell
cp ../../data/fashion_MNIST/frames/thumb{0010..0021}.png ../../data/fashion_MNIST/test_frames/
```

- Run demo_MNIST.py 
```shell
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../tools/demo_MNIST.py --resume SiamMask_DAVIS.pth --config config_davis.json
```
The resulting frames will owerwrite the input frames in *test_frames* directory

- Generating GIF
```shell
python ../../data/process_mnist.py
```
This will store the GIF in $SiamMask/data/fashion_MNIST/GIFs directory by default. If you wish to store it somewhere else, you must specify the --gif_frames_dir flag.


## License
Licensed under an MIT license.


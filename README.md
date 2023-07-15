# Exploring MMPretrain and MMDetection

This repository is about an exploration the potential of using MMDetections models with self-supervised training techniques.

As a addition, I'm also interested in explore the potential of the Fiftyone tool, to explore the datasets, mainly due to their latent space visualization widget.

The networks and techniques explored here are:


```block
TODO
```

# Instalation process

Create env and activate it

```bash
$ conda create -n mmdet-ssl python=3.9
$ conda activate mmdet-ssl
```

Basic dependencies

```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ pip install -U openmim
$ mim install mmengine
$ mim install "mmcv>=2.0.0"
```

## Fiftyone

```
$ pip install fiftyone
$ pip install "fiftyone[desktop]"
$ pip install 'ipywidgets>=7.5,<8'
```

Troubleshooting for my linux:

```
$ pip install fiftyone-db-ubuntu2204
```

## MMPretrain

## MMDetection

## Install python dependencies

```bash
$ git clone https://github.com/open-mmlab/mmdetection.git
$ cd mmdetection && pip install -v -e .
```

## Download a specific configuration and weights from MMDetection

`Tip`: See full list of configuration [here](https://github.com/open-mmlab/mmdetection/tree/main/configs)

```bash
mim download mmdet --config <configuration-name>
```

# Glossary:

* TTA: [Test time augmentation](https://mmdetection.readthedocs.io/en/latest/user_guides/test.html?highlight=tta#test-time-augmentation-tta)

# TODO - exploration
[x] Explore Pretrained models from MMPretrain

[] Pre-training the ConvNext using SSL

[] Train a RTMDET-ins on a standard dataset

[] Train a RTMDET-ins using ConvNext as backbone

[] Train a RTMDET-ins model using the pre-trained ConvNext with SSL

[] Embedding space exploration using Tensorboard

[] Evaluate script for the SSL

[] Script to pick the most unique x% from a given dataset

[] Explore COCO-ish dataset using fiftyone
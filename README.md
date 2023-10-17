# ICASSP 2024

This repo contains the data and code for the following paper:

Xinyu Wu, Xugong Qin, YongSheng Han, Peng Zhang, Sihang Yang, Xinjian Huang, Ming Zhou, Qinfeng Tan. Make It Simple: Efficient Rumor Detection with Dynamic Evolving Graph and CLIP-Guided Modality Alignment



## Requirements

We implement our model and its variants based on PyTorch 2.0.0 with CUDA 11.8, and train them on a server running Ubuntu 18.04 with NVIDIA RTX 3090 GPU.

Main dependencies include:

```
python==3.8.0
numpy==1.24.4
torch==2.0.0+cu118
transformers==4.33.1
```

install the virtual environment with

```
pip install -r requirement.txt
```

## Data
The datasets can be respectively downloaded from: https://www.dropbox.com/scl/fi/9wttb2fa3wgmwsp4spyko/dataset.zip?rlkey=r3tv9f1gbv2r0dr3nial51eas&dl=0

## Run Model

```
cd ./graph_part
python MDEGCN_pheme.py
python MDEGCN_weibo.py
```



## Contact

if you have any problems, please contact the author: sinywu@njust.edu.cn


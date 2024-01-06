![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.7](https://img.shields.io/badge/PyTorch->=1.7-blue.svg)

# THCB-Net: Text-Guided and Hierarchical Context Blending Network for Occluded Person Re-identification
Official PyTorch implementation of the paper "**THCB-Net: Text-Guided and Hierarchical Context Blending Network for Occluded Person Re-identification**". 

The code will be released.

## Installation

We use python=3.7.0, PyTorch=1.7.1, CUDA=10.1 and torchvision=0.8.2. You can install environment dependencies by running the following command.

```bash
conda create -n THCB-Net python=3.7
conda activate THCB-Net
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


### Prepare Datasets

```bash
First cd THCB-Net/
then cd ../
mkdir person_reid_datasets
```

Download the person datasets Market-1501, DukeMTMC-reID, Occluded-Duke, Occluded-REID, Partial-iLIDS and Partial-REID. 

**Note:** The text information used in the paper will be released.

Then unzip them and rename them under the '../person_reid_datasets' directory like

```
|-- person_reid_datasets
|   ├── Occluded-Duke
|   │   └── bounding_box_train
|   │   └── bounding_box_test
|   │   └── query
|   │   └── bounding_box_train.json
|   │   └── bounding_box_test.json
|   │   └── bounding_box_query.json
|   |
|   ├── Occluded-REID
|   │   └── bounding_box_test
|   │   └── query
|   │   └── bounding_box_test.json
|   │   └── query.json
|   |
|   ├── Partial-REID
|   │    └── partial_body_images
|   │    └── whole_body_images
|   │    └── partial_body_images.json
|   │    └── whole_body_images.json
|   ├── Partial-iLIDS
|   │   └── bounding_box_test
|   │   └── query
|   │   └── bounding_box_test.json
|   │   └── query.json
|   |
|   ├── market1501
|   │   └── bounding_box_train
|   │   └── bounding_box_test
|   │   └── query
|   │   └── bounding_box_train.json
|   │   └── bounding_box_test.json
|   │   └── bounding_box_query.json
|   |
|   └── DukeMTMC-reID
|       └── bounding_box_train
|       └── bounding_box_test
|       └── query
|       └── bounding_box_train.json
|       └── bounding_box_test.json
|       └── bounding_box_query.json
|
└-- THCB—Net
    ├── config
    ├── configs
    ├── data
    ├── datasets
    ├── log
    ├── loss
    ├── model
    ├── processor
    ├── solver
    ├── utils
    ├── train.py
    └── test.py

```

## Training

We utilize 2 TITAN RTX GPUs or 4 RTX 5000 GPUs for training. Then, you can run the following commands for training.


```bash
# Occluded-Duke Swin Base
CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file configs/occ_duke/swin_base.yml
# Market1501 Swin Base
CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file configs/market/swin_base.yml
# DukeMTMC-reID Swin Base
CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file configs/dukemtmc/swin_base.yml
# Occluded-REID Swin Base
CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file configs/occ_reid/swin_base.yml
# Partial-REID Swin Base
CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file configs/Partial_REID/swin_base.yml
# Partial-iLIDS Swin Base
CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file configs/Partial_iLIDS/swin_base.yml
```

## Test

We utilize 1 TITAN RTX GPUs or 1 RTX 5000 GPUs for testing.

```bash
# For example, Occluded-Duke Swin Base
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/occ_duke/swin_base.yml
```

## Performance

#### The results on occluded and partial person re-ID datasets:

|       Method       | Publication | Occluded-Duke | Occluded-REID | Partial-REID  | Partial -iLIDS |
| :----------------: | :---------: | :-----------: | :-----------: | :-----------: | :------------: |
|        PAT         |  CVPR 2021  |   64.5/53.6   |   81.6/72.1   |    88.0/--    |    76.5/--     |
|     TransReID      |  ICCV 2021  |   66.4/59.2   |     --/--     |   71.3/68.6   |     --/--      |
|        FED         |  CVPR2022   |   68.1/56.4   |   86.3/79.3   |   84.6/82.3   |     --/--      |
|        PFD         |  AAAI 2022  |   69.5/61.8   |   81.5/83.0   |     --/--     |     --/--      |
|        FRT         |  TIP 2022   |   70.7/61.3   |   80.4/71.0   |    88.2/--    |    73.0/--     |
|        DPM         |  ACMMM2022  |   71.4/61.8   |   85.5/79.7   |     --/--     |     --/--      |
| CLIP-ReID+SIE+OLP  |  AAAI 2023  |   67.2/60.3   |     --/--     |     --/--     |     --/--      |
|        CAAO        |  TIP 2023   |   68.5/59.5   |   87.1/83.4   |     --/--     |     --/--      |
|        SAP         |  AAAI 2023  |   70.0/62.2   |   83.0/76.8   |     --/--     |     --/--      |
|       RGANet       |  TIFS 2023  |   71.6/62.4   |   86.4/80.0   |    87.2/--    |    77.0/--     |
|        FODN        |  ACMMM2023  |   72.6 61.9   |   87.1/80.8   |     --/--     |     --/--      |
|      SOLIDER       |  CVPR 2023  |   71.3/62.3   |   85.0/82.0   |   79.0/78.7   |   79.0/84.6    |
|       [TSD](https://arxiv.org/abs/2312.09797)        | ICASSP 2024 |   74.5/62.8   |     --/--     |     --/--     |     --/--      |
| **THCB-Net** (our) |   Waiting   | **80.0/70.7** | **88.9/84.8** | **91.3/88.0** | **83.2/89.0**  |

#### The results on holistic person re-ID datasets:

|       Method       | Publication |  Market1501   | DukeMTMC-reID |
| :----------------: | :---------: | :-----------: | :-----------: |
|        PAT         |  CVPR2021   |   95.4/88.0   |   88.8/78.2   |
|     TransReID      |  ICCV 2021  |   95.2/89.5   |   90.7/82.6   |
|        FED         |  CVPR2022   |   95.0/86.3   |   89.4/78.0   |
|        PFD         |  AAAI 2022  |   95.5/89.7   |   89.4/78.0   |
|        FRT         |  TIP 2022   |   95.5/88.1   |   90.5/81.7   |
|        DPM         |  ACMMM2022  |   95.5/89.7   |   91.0/82.6   |
|        PASS        |  ECCV2022   |   96.9/93.3   |     --/--     |
| CLIP-ReID+SIE+OLP  |  AAAI 2023  |   95.4/90.5   |     --/--     |
|        CAAO        |  TIP 2023   |   95.3/88.0   |   89.8/80.9   |
|        SAP         |  AAAI 2023  |   96.0/90.5   |     --/--     |
|       RGANet       |  TIFS 2023  |   95.5/89.7   |     --/--     |
|        FODN        |  ACMMM2023  |   95.5 89.2   |   91.2/83.3   |
|    PLIP+ABDNet     | Arxiv 2023  |   96.7/91.2   |   90.9/81.6   |
|        PATH        |  CVPR 2023  |   \--/91.8    |     --/--     |
|        RAM         | Arxiv 2023  |   96.3/90.1   |     --/--     |
|       UniHCP       |  CVPR 2023  |   \--/90.3    |     --/--     |
|      SOLIDER       |  CVPR 2023  |   96.9/93.9   |   92.8/85.2   |
|       [TSD](https://arxiv.org/abs/2312.09797)        | ICASSP 2024 |     --/--     |   90.8/82.8   |
| **THCB-Net** (our) |   Waiting   | **97.4/93.9** | **94.7/87.2** |

## Citation

If you find the **THCB-Net** useful for your research, please cite our paper.

```
@article{
Please be patient and wait for the release of the arxiv version.
}
```

# EX-MSLC: Malicious Traffic Detection from Noisy Labels via Expanded Metadata



![ex-mslc8](https://github.com/bo-work/EX-MSLC/blob/main/ex-mslc8.png)



## Description: 

An official PyTorch implementation of the "EX-MSLC: Malicious Traffic Detection from Noisy Labels via Expanded Metadata" paper.

## Results

CICIDS2018 with `1%` of training set:

| Model / Noise | Sym-10% | Sym-30% | Sym-50% | Sym-70% | Asym-10% | Asym-30% | Asym-50% | Asym-70% |
| :-----------: | :-----: | :-----: | :-----: | :-----: | :------: | :------: | :------: | :------: |
|    EX-MSLC    |  97.04  |  96.98  |  96.83  |  94.81  |  96.64   |  96.41   |  96.23   |  94.74   |

CICIDS2018 with `N=[500, 100, 50]` sample:

| N / Noise | Sym-50% | Asym-50% | Sym-70% | Asym-70% |
| :-------: | :-----: | :------: | :-----: | :------: |
|    500    |  97.02  |  97.02   |  94.67  |  94.78   |
|    100    |  96.84  |  96.98   |  92.66  |  94.34   |
|    50     |  96.51  |  96.64   |  91.12  |  93.36   |

CICIDS2018 with `N=[500]` sample for unannotated attack:

Infiltration

![1aft3](https://github.com/bo-work/EX-MSLC/blob/main/1aft3.png)

BruteForce

![3aft](https://github.com/bo-work/EX-MSLC/blob/main/3aft.png)



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Running

Before training the models please:

1. Put the dataset in the `/data/CICIDS18` . 
   1. Origian dataset is [here](https://www.unb.ca/cic/datasets/ids-2018.html) and put in `/data/CICIDS18/origindata`, run `python3 preprocess_ids18.py`;
   2. Preprocessed data is [here](https://www.unb.ca/cic/datasets/ids-2018.html) and put in `/data/CICIDS18/noisedata`
2. You can use Pretrained model [here]() for warmup phase, and put them in the `/results/models/warmup/IDS18`.

train phase:
```
python exmslc.py
```
paramaters can be set in `config.py`

## Contributing

>This repository is heavily based on [MSLC](https://github.com/WuYichen-97/Learning-to-Purify-Noisy-Labels-via-Meta-Soft-Label-Corrector).

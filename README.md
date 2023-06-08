# TMSON

TMSON is a framework for Multimodal Sentiment Analysis. The code is built on [MMSA](https://github.com/thuiar/MMSA).

## 1. Get Started

### 1.1 Datasets
Use the following links to download raw videos, feature files and label files.

- Feature and label files only: [Google Drive](https://drive.google.com/drive/folders/12M5AeBnpjVzeNIcLromJRDq_-jNg0vHY?usp=sharing)
- MOSEI unaligned_50.pkl: [Google Drive](https://drive.google.com/drive/folders/19Nurt_SbWbmZqXgLFepaWOGQOgxlSv_C?usp=sharing)

Dataset paths and parameters are set in `config_regression.json` file. Modify the `config_regression.json` to update dataset paths.

```python
 {
  "datasetCommonParams": {
    "dataset_root_dir": "/home/WorkSpace/Dataset/MMSA-Standard",
    "mosi": {
      "aligned": {
        "featurePath": "MOSI/Processed/aligned_50.pkl",
        "seq_lens": [50, 50, 50],
        "feature_dims": [768, 5, 20],
        "train_samples": 1284,
        "num_classes": 3,
        "language": "en",
        "KeyEval": "Loss"
      },
      ...
  }
```

For MOSI and MOSEI, pre-trained models are downloadable at [Google Drive](https://drive.google.com/drive/folders/1Pfc37oLQhLF7d_4VPsvhnOKeMJSEDBjR?usp=sharing).

### 1.2 Environment

```python
 pytorch  	1.12.0
 torchvision     0.13.0
 tensorboard     2.8.0
 matplotlib	3.3.4	
 numpy		1.22.3	
 python		3.8.13
 pandas  	1.3.5	
 scipy		1.9.1	
 tqdm 		4.64.0
 easydict	1.10
 transformers 	>= 4.4.0
```

## 2. Run codes 

```bash
  $ cd TMSON-master/
  $ python src/MMSA/run.py
```

Refer to [MMSA](https://github.com/thuiar/MMSA) for more details.

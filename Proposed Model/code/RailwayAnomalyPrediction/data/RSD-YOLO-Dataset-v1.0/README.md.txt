# RSD-YOLO Rail Surface Defect Dataset

This repository provides the rail surface defect dataset used in the paper:
## Data Source and License

This dataset was prepared based on publicly available rail surface defect images downloaded from Roboflow. The downloaded metadata file indicates that the original dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

In this repository, the dataset has been reorganized into YOLO format with fixed training, validation, and test splits for reproducible experiments. Users should comply with the CC BY 4.0 license terms and cite the associated paper when using this dataset.

## Dataset Download

The complete dataset is available in the GitHub Release:

RSD-YOLO-Dataset v1.0.0

The released zip file contains rail surface defect images, YOLO-format annotation files, fixed train/validation/test split files, class definitions, and the data.yaml configuration file.

**RSD-YOLO: A Lightweight Model for Rail Surface Defect Detection**

## 1. Dataset Description

The dataset contains rail surface defect images for object detection. It includes three defect categories:

| Class ID | Class Name |
|---|---|
| 0 | Spalling |
| 1 | Wheel burn |
| 2 | Squat |

The annotations are provided in YOLO txt format. Each image has a corresponding label file with the same file name.

## 2. Dataset Size

The dataset contains 4,020 images in total.

| Split | Number of Images |
|---|---:|
| Train | 3,216 |
| Validation | 402 |
| Test | 402 |
| Total | 4,020 |

## 3. Directory Structure

```text
RSD-YOLO-Dataset/
├── data.yaml
├── classes.txt
├── train.txt
├── val.txt
├── test.txt
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── stats/
    └── class_distribution.csv


The released zip file contains rail surface defect images, YOLO-format annotation files, fixed train/validation/test split files, class definitions, and the data.yaml configuration file.

# CLEAN PII

This repository implements functionality for detecting and cleaning PII from X-ray images.

It includes functionality for detecting personally identifiable information on chest X-ray images for removal. It detects PII and returns a bounding box of around the PII for removal.

* [Dataset YOLO FORMAT](dataset_yolo_format.py). This file is used to build a YOLO compartible dataset from a csv file of bounding boxes in xyxy format.
* [Train YOLO Models](train_yolo_models.py). This file is used to train a yolo model according to the yaml file provided.
* [Inference YOLO Model](inference_yolo_models.py). This file is used to run inference on the YOLO model.
* [config](yolo11m_config.yaml). This is the model configuration file.

## Features 

### 1. Data Processing

The [data processing](datasets/dataset_yolo_format.py) scipt builds a yolo dataset.

### 2. Model Training

The [model training](train/train_yolo_models.py) script can be used to train a YOLO11 model to detect PII on a custom dataset.

### 3. Inference

The [inference](inference/inference_yolo_models.py) script runs the model on X-ray images to detect PII and returns bounding boxes for the PII.

### Clean and Save
See [instructions](Instructions.md).
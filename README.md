# Vehicle Detection and Counting System

This project demonstrates **real-time vehicle detection and counting** using the **YOLOv8** object detection model.  
It includes scripts for data preparation, model training, fine-tuning, evaluation, and inference.  
The system can be used to analyze traffic density and vehicle flow in video footage.

---

## 1. Introduction

Accurate vehicle detection and counting play a key role in intelligent transportation systems,  
traffic monitoring, and road safety management. Traditional manual counting is labor-intensive  
and prone to human error. This project implements an automated system that leverages  
deep learning–based object detection to identify and count vehicles from live or recorded video streams.  

Using the **YOLOv8** model, the system achieves high accuracy in detecting multiple vehicle classes  
(cars, buses, trucks, motorcycles) across various conditions. The workflow covers the entire pipeline —  
from dataset preparation to fine-tuning and real-time deployment.

---

## 2. How It Works

The system processes video frames and performs vehicle detection in real time using YOLOv8.  
Bounding boxes and class labels are generated for each detected vehicle, which are then  
used to count the number of vehicles passing through a defined region of interest.

Key components of the pipeline:
1. **Dataset Preparation** — Frames are extracted from raw videos and annotated through Roboflow.  
2. **Model Training** — The YOLOv8 model is trained or fine-tuned on the prepared dataset.  
3. **Evaluation** — Model accuracy is assessed using mean Average Precision (mAP) metrics.  
4. **Inference and Counting** — Vehicles are detected and counted from video input streams.  

---

## 3. Project Files

- **`frames_extraction.py`** — Extracts frames from raw videos for dataset creation.  
- **`roboflow_data.py`** — Downloads training data directly from Roboflow.  
- **`trained_data.py`** — Loads or references previously downloaded datasets.  
- **`train_model.py`** — Trains the YOLOv8 model on the prepared dataset.  
- **`finetune.py`** — Fine-tunes a pre-trained YOLOv8 model for improved accuracy.  
- **`mAP.py`** — Computes the mean Average Precision (mAP) for model evaluation.  
- **`vehicle-counter.py`** — Performs real-time vehicle detection and counting on video input.

---

## 4. Requirements

Install dependencies using:
```bash
pip install -r requirements.txt

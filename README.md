# Hockey Referee Gesture Classification  
**Computer Vision & Machine Learning Project**

This repository contains the code and dataset used in a bachelor’s thesis project focused on classifying ice hockey referee gestures using pose estimation and machine learning methods.

The goal of this project is to automatically recognize referee hand signals in hockey videos using computer vision techniques. These signals can potentially be used for analytics, automated tagging, or assistive systems in sports monitoring.

---

## Project Overview

In this project, we:

- Perform pose estimation to extract key body landmarks.  
- Train a machine learning model to recognize distinct referee gestures.  
- Provide tools to evaluate videos and make predictions.  
- Build a simple GUI to interact with the trained model.

The project integrates computer vision and machine learning to detect and classify gestures from video streams.

---

## How It Works

1. **Pose Estimation** – Extract body keypoints from video frames using a pose detection library.  
2. **Feature Extraction** – Create meaningful features (e.g., landmark distances, angles) from pose keypoints.  
3. **Model Training** – Train an XGBoost classifier (or another model) on the extracted features.  
4. **Prediction & Evaluation** – Predict gesture classes on new inputs.  
5. **GUI** – A simple interface to load videos and test gesture recognition interactively.

---

## Repository Contents

- `train_xgb.py` – Script to train the gesture classification model.  
- `predict_video.py` – Inference script to classify gestures in video files.  
- `gui.py` – Simple graphical interface for users to test gesture predictions.  
- `features_5.csv` – Example feature dataset used for training.  
- `xgb_gesture_model_5.pkl` – Trained model file.  
- `scaler_5.pkl` / `feature_names_5.pkl` – Supporting artifacts for preprocessing.  
- `test_videos/` – Example videos for evaluation (without original broadcast content).

---

## Dataset

The dataset used to train models was created by extracting frames from publicly available hockey broadcasts.  
Original video recordings are **not included** in this repository due to copyright.

### Data Source and Rights

The original video materials were obtained from publicly available broadcasts on:  
https://www.khl.ru/  

All rights for these materials belong to the KHL.  
The videos were used only for **research and educational purposes**.  
No copyrighted video content is redistributed in this repository.

> **IMPORTANT:** This dataset is provided for academic and non‑commercial use only.

---

## Technologies Used

- Python
- Machine Learning: XGBoost
- Computer Vision: pose estimation libraries
- Data processing: pandas, NumPy
- GUI: Tkinter (optional UI)
- Version control: Git & GitHub

---

## Requirements

Before running the code locally, install dependencies:

```bash
pip install -r requirements.txt





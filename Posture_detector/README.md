#  Posture Detection Module

This module contains the core machine learning pipeline for training and running real-time posture classification.

---

##  Purpose

The goal of this component is to take structured pose landmark data and produce a model capable of classifying posture in real time.

It bridges the gap between raw data collection and live system deployment.

---

##  Functionality

This module handles:

* Loading labeled posture datasets
* Preprocessing and normalization of features
* Training a machine learning classifier
* Evaluating model performance
* Running real-time posture prediction using webcam input

---

##  Model Pipeline

### 1. Data Loading

* Reads landmark-based datasets stored as CSV files
* Each row represents a single posture sample

### 2. Preprocessing

* Applies feature scaling using `StandardScaler`
* Ensures consistent input distribution for the model

### 3. Training

* Uses `KNeighborsClassifier`
* Splits data into training and testing sets

### 4. Prediction

* Extracts pose landmarks in real time
* Applies trained model to classify posture
* Outputs predictions continuously from webcam feed

---

##  How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Prepare Dataset

* Use the `DataCollection` module to generate labeled posture data
* Ensure your dataset is stored in the expected folder structure
* Update dataset paths in the script if needed

---

### 3. Run the Model

```bash
python code.py
```

---

### 4. What Happens

* The model is trained on your dataset
* Your webcam will open
* The system will:

  * detect your body posture
  * classify it (good / bad / neutral)
  * display predictions in real time

---

###  Notes

* You may need to modify dataset paths depending on your setup
* Good lighting and camera positioning improve performance
* Make sure your webcam is accessible

---

##  Why This Approach

* **Lightweight**: Runs efficiently without requiring GPU
* **Interpretable**: KNN allows understanding of classification behavior
* **Modular**: Can be easily replaced with more advanced models

---

##  Limitations

* Model performance depends on dataset quality and size
* Sensitive to camera angle and environmental conditions
* Does not incorporate temporal information (frame-to-frame movement)

---

##  Possible Improvements

* Replace KNN with more advanced models (e.g., neural networks)
* Incorporate temporal data for smoother predictions
* Improve generalization with more diverse datasets
* Add evaluation metrics (accuracy, confusion matrix)

---

##  Notes

This module focuses on simplicity and clarity of the machine learning pipeline, prioritizing rapid experimentation and real-time performance.


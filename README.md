# 🧠 Posture Detection System (PostureBuddy)

A real-time computer vision system that detects and classifies human posture using body landmarks from a webcam feed. This project combines data collection, machine learning, and live inference into a complete end-to-end pipeline.

## 🚀 Overview

Poor posture is a common issue among students and professionals working long hours at a desk. This project aims to provide a lightweight, real-time solution that:

- Tracks human body posture using computer vision
- Classifies posture into categories (e.g., good, bad, neutral)
- Runs live predictions from webcam input

The system is built using Python and leverages pose estimation to extract meaningful features for classification.

---

## 🧩 Project Structure
posture-detection/
│
├── DataCollection/ # Collect labeled posture data
├── Posture_detector/ # Train model + real-time detection
├── datasets/ # Saved posture landmark data
└── README.md

---

## ⚙️ How It Works

### 1. Data Collection
- Uses webcam + pose estimation
- Captures body landmarks (keypoints)
- Labels data manually using keyboard input:
  - `G` → Good posture
  - `B` → Bad posture
  - `N` → Neutral posture

### 2. Feature Engineering
- Extracts pose landmark coordinates
- Normalizes and structures them into CSV datasets

### 3. Model Training
- Uses:
  - `StandardScaler` for normalization
  - `KNeighborsClassifier` for classification
- Splits dataset into training and testing sets

### 4. Real-Time Prediction
- Live webcam feed
- Extracts pose landmarks in real-time
- Applies trained model to classify posture instantly

---

## 🧠 Tech Stack

- Python
- OpenCV
- MediaPipe (pose estimation)
- NumPy / Pandas
- Scikit-learn

---

## 📊 Results & Performance

This project demonstrates a functional ML pipeline from raw data to deployment.

Key strengths:
- Real-time inference
- End-to-end system (data → model → live prediction)
- Lightweight and runs locally

Limitations:
- Accuracy depends on lighting and camera angle
- Dataset is user-specific (generalization can be improved)

---

## 🎯 Future Improvements

- Improve model accuracy with larger dataset
- Use deep learning (e.g., LSTM or CNN for pose sequences)
- Add posture correction alerts
- Deploy as desktop/mobile app

---

## 💡 Why This Project Matters

This project is not just a model — it is a **complete machine learning system** that includes:

- Data engineering
- Model training
- Real-time deployment

It reflects practical understanding of how AI systems are built and used in real-world applications.

---

## 👤 Author

Shane Sok  
High school student passionate about AI, computer vision, and real-world applications of machine learning.

---

## 📌 Notes

This project was built independently as part of my exploration into applied AI systems and human-centered technology.

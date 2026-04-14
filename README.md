# ✨ Real-Time Air Writing Recognition

This repository presents a **Real-Time Air Writing Recognition System** based on a **Fusion Architecture** that combines **Coarse Classification (CNN)** and **Fine Classification (CNN + BiLSTM + Attention)**.

The system enables users to **write words and draw doodles in the air using hand gestures**, captured via a webcam & no physical tools required.

It recognizes:
- 📝 **49 predefined words**
- 🎨 **Multiple doodles (gesture-based drawings)**

---

## 🚀 Key Features
- Real-time air-writing using webcam
- Fusion-based deep learning model for improved accuracy
- Recognizes **49 words + doodles**
- Two-stage prediction:
  - **Coarse Model (CNN)** → Fast category filtering based on similar patterns
  - **Fine Model (CNN + BiLSTM + Attention)** → Precise recognition
- Robust to variations in -
  - Writing speed  
  - Style  
  - Direction
  - Time frames
- Reduces Character Error Rate (CER)

---

## 🧠 Fusion Model Architecture

### 🔹 Stage 1: Coarse Model (CNN)
- Lightweight CNN model
- Performs **initial classification** based on similar patterns, repeated sequence, etc.
- Filters input into **broad categories**
- Reduces search space for fine model

---

### 🔹 Stage 2: Fine Model (CNN + BiLSTM + Attention)

- **CNN Layers**
  - Extract spatial features from input trajectory

- **Bidirectional LSTM (BiLSTM)**
  - Captures temporal dependencies in both directions

- **Attention Layer**
  - Focuses on crucial parts of the sequence
  - Improves recognition of complex patterns

- **Dropout**
  - Prevents overfitting

- **Dense + Softmax**
  - Final classification into **49 words / doodles**

---

### 🔄 Why Fusion?
- Faster inference (coarse filtering)
- Higher accuracy (fine-grained attention-based model)
- Switching between Doodles & Words models

---

## 📊 Dataset

### Word Dataset
- Custom dataset of **49 words**
- Designed for air-writing trajectories
- Preprocessed into:
  - Normalized sequences
  - Fixed-size inputs (64X64)

---

### Doodle Dataset
- Kaggle dataset with doodle drawings
- Integrated into same pipeline

---

## 🖐️ Real-Time Recognition Pipeline

1. Webcam captures hand movement  
2. **MediaPipe** detects and tracks index finger  
3. Finger trajectory drawn on virtual canvas  
4. Pause detection segments input  
5. Canvas resized and preprocessed  
6. **Coarse CNN predicts category**  
7. **Fine model refines prediction using Attention**  
8. Output displayed in real time  

---

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- MediaPipe  
- NumPy  
- Matplotlib  

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

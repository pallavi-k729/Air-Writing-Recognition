# Real-Time Air Writing Recognition


This repository contains a **Air Writing Recognition** model using a Attention based Bidirectional LSTM model. This is an innovative interaction technique that allows users to write characters in free space using hand gestures and movements without the need for traditional tools like pens or paper. 
The system recognizes **air-written characters and 40+ doodles in real time** using hand gestures captured via a webcam.

<br>


## 🚀 Key Features
-  Real-time air-writing recognition using webcam
-  Recognizes **49 words**
-  Recognizes **40+ hand-drawn doodles**
-  Hybrid **Conv1D + Bidirectional GRU (CBRNN)** Architecture
-  Robust to writing speed, style and direction
-  Character-level word recognition with pause detection

<br>

---

## 🧠 Model Architecture (CBRNN)
The proposed **CBRNN** combines spatial and temporal learning:

- **Conv1D Layers**
  - Extract local spatial patterns from sequential pixel data
- **MaxPooling1D**
  - Reduces dimensionality while retaining key features
- **Bidirectional GRU Layers**
  - Capture temporal dependencies in both forward and backward directions
- **Dropout Layers**
  - Prevent overfitting
- **Dense + Softmax**
  - Final classification of characters and doodles

**Input Shape:** `(784, 1)`  
**Output:** Character / Doodle Class Probability

<br>

---

## 📊 Dataset
### EMNIST Dataset
- Original format: `28 × 28` grayscale images
- Preprocessing:
  - Flattened into **1D sequences (784 pixels)**
  - Normalized to `[0, 1]`
- Purpose:
  - Simulates sequential air-writing trajectories

### Doodle Dataset
- Custom curated dataset
- Contains **40+ distinct doodle classes**
- Integrated into the same CBRNN pipeline


---



## 🖐️ Real-Time Recognition Pipeline
1. Webcam captures hand movement
2. **MediaPipe** tracks index finger tip
3. Finger trajectory drawn on virtual canvas
4. Motion pause detected → character segmentation
5. Canvas resized to `28 × 28`
6. Flattened into sequence → CBRNN prediction
7. Output displayed instantly

---

## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- **Matplotlib**


---
<br>

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Train the Model
```bash
python train.py
```
### 3️⃣ Run Real-Time Air Writing
```bash
python app.py
```
<br>

## 🔮 Future Work

- Continuous word and sentence recognition
- Sign language recognition
- Language model integration
- Multilingual support
- VR / AR system integration
- Improved robustness to noisy backgrounds

<br>


## 🌍 Applications

- 🏥 Healthcare (sterile interaction)
- ♿ Assistive technology for mobility-impaired users
- 🏭 Industrial & agricultural environments
- 🎓 Education in rural and resource-constrained areas
- 🕶️ AR / VR interfaces

<br>

---

<br>

## 📸 Screenshots

<img width="500" height="450" alt="Screenshot 2025-11-12 184912" src="https://github.com/user-attachments/assets/e75d702d-fad1-48f5-bbe6-3ae908196816" />

<img width="700" height="500" alt="Screenshot 2025-11-12 183636" src="https://github.com/user-attachments/assets/dfd95fb6-a476-4409-bc5d-4274c92536db" />


---


## Results

### CNN
<img width="600" height="300" alt="CNN" src="https://github.com/user-attachments/assets/dab40471-217c-4dc6-83f6-8092f1559fac" />

---

### CNN+RNN
<img width="600" height="300" alt="CNN+RNN" src="https://github.com/user-attachments/assets/c9aa4172-71dd-4bd3-8f5e-374fed26f10a" />

---

### CNN+BiLSTM
<img width="600" height="300" alt="CNN+BiLSTM" src="https://github.com/user-attachments/assets/68957c17-7826-4813-857b-7fc7aa48a215" />

---

### CNN+BiGRU
<img width="600" height="300" alt="CNN+BiGRU" src="https://github.com/user-attachments/assets/37df105c-a6d0-467c-8a11-2c6a2f91c626" />

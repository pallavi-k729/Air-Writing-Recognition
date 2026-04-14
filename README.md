# Real-Time Air Writing Recognition

This repository presents a **Real-Time Air Writing Recognition System** based on a **Fusion Architecture** that combines **Coarse Classification (CNN)** and **Fine Classification (CNN + BiLSTM + Attention)**.

The system enables users to **write words and draw doodles in the air using hand gestures**, captured via a webcam & no physical tools required.

It recognizes:
- **49 predefined words**
- **Multiple doodles (gesture-based drawings)**
<br>



## Key Features
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
<br>



## Fusion Model Architecture

### 🔹 Stage 1: Coarse Model (CNN)
- Lightweight CNN model
- Performs **initial classification** based on similar patterns, repeated sequence, etc.
- Filters input into **broad categories**
- Reduces search space for fine model
<br>



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
<br>



### Why Fusion?
- Faster inference (coarse filtering)
- Higher accuracy (fine-grained attention-based model)
- Switching between Doodles & Words models
<br>
<br>

## 📊 Dataset

### Word Dataset
- Custom dataset of **49 words**
- Designed for air-writing trajectories
- Preprocessed into:
  - Normalized sequences
  - Fixed-size inputs (64X64)



### Doodle Dataset
- Kaggle dataset with doodle drawings
  ```bash
  https://www.kaggle.com/datasets/linhlthk19/doodle
  ```
- Integrated into same pipeline

<br>

## Real-Time Recognition Pipeline

1. Webcam captures hand movement  
2. **MediaPipe** detects and tracks index finger  
3. Finger trajectory drawn on virtual canvas  
4. Pause detection segments input  
5. Canvas resized and preprocessed  
6. **Coarse CNN predicts category**  
7. **Fine model refines prediction using Attention**  
8. Output displayed in real time  

<br>

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- MediaPipe  
- NumPy  
- Matplotlib  

<br>

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Train the Model
```bash
python train_coarse.py
python train_grps.py
```
### 3️⃣ Run Real-Time Air Writing
```bash
python app.py
```
<br>

## 🔮 Future Work

- Multilingual support
- VR / AR system integration
- Improved robustness to noisy backgrounds

<br>


## 🌍 Applications

- 📊 Password writing (keylogging prevention)
- 🏥 Healthcare (sterile interaction)
- ♿ Assistive technology for mobility-impaired users
- 🏭 Industrial & agricultural environments
- 🎓 Education in rural and resource-constrained areas
- 🕶️ AR / VR interfaces

<br>

---

<br>

## 📸 Screenshots

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/bc74b9ea-72b7-498a-8d9c-647685125592" />
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/0327096e-ca3f-4e30-9d09-eedc4407e9de" />


<br>


## Results
- Coarse Model
  <img width="900" height="300" alt="coarse_plot" src="https://github.com/user-attachments/assets/baac3422-839b-477a-8a16-a480796ee4c1" />
<br>

- Fine models
  <img width="900" height="300" alt="plot" src="https://github.com/user-attachments/assets/03dc1c68-b1a4-4713-bb46-7a89dd84078f" />
  <img width="900" height="300" alt="plot" src="https://github.com/user-attachments/assets/5fa74e3d-0224-46b2-8ad5-1707b5008b8d" />
  <img width="900" height="300" alt="plot" src="https://github.com/user-attachments/assets/cf7d69ea-3028-4b06-95c4-f871933526cb" />
  <img width="900" height="300" alt="plot" src="https://github.com/user-attachments/assets/9fc68790-a1cd-4aea-b470-04541bf0b2e9" />
  <img width="900" height="300" alt="plot" src="https://github.com/user-attachments/assets/c3978aa6-f625-408a-b6b5-65a9e1c216ff" />
  <img width="900" height="300" alt="plot" src="https://github.com/user-attachments/assets/ade5791f-f8de-4b7f-a38c-b1dee2b5f596" />

# ANN-Classification 🚀

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end **Artificial Neural Network (ANN)** project for **Customer Churn Prediction**, including data preprocessing, model training, evaluation, TensorBoard visualization, and a deployed **Streamlit web application** for real-time inference.

This repository demonstrates a complete **machine learning lifecycle**: from raw data → preprocessing → model training → evaluation → deployment.

---


---

## 🧩 Project Overview

Customer churn prediction helps companies identify customers who are likely to stop using their services.

In this project, we:

- Load and clean structured customer data  
- Encode categorical variables  
- Scale numerical features  
- Train an ANN classifier using TensorFlow  
- Save trained model and preprocessing objects  
- Build a **Streamlit UI** for real-time predictions  

---

## 🎯 Problem Statement

Given customer details such as:

- Credit Score  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Credit Card status  
- Active Membership  
- Estimated Salary  

Predict whether the customer is **likely to churn** (`Exited = 1`) or **not churn** (`Exited = 0`).

This is a **binary classification problem**.

---

## 📂 Dataset

- File: `Churn_Modelling.csv`  
- Source: Banking customer dataset  
- Target column: `Exited`  

### Features:

| Feature | Description |
|-------|------------|
| CreditScore | Customer credit score |
| Geography | Country |
| Gender | Male/Female |
| Age | Age of customer |
| Tenure | Years with bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | Has credit card (0/1) |
| IsActiveMember | Active member (0/1) |
| EstimatedSalary | Annual salary |

---

## ⚙️ Tech Stack

- **Python 3.11**
- **TensorFlow / Keras 2.15**
- Scikit-learn  
- Pandas, NumPy  
- Streamlit  
- TensorBoard  

---

## 🧹 Data Preprocessing

Steps applied:

1. Drop unnecessary columns:
   - `RowNumber`
   - `CustomerId`
   - `Surname`

2. Encode categorical features:
   - `Gender` → Label Encoding  
   - `Geography` → One-Hot Encoding  

3. Feature Scaling:
   - `StandardScaler` applied to all numerical features  

4. Train-test split:
   - 80% training  
   - 20% testing  

All encoders and scalers are saved using `pickle` for deployment.

---

## 🧠 Model Architecture

ANN Classification Model:

Details:

- Hidden Layer 1: 64 neurons, ReLU  
- Hidden Layer 2: 32 neurons, ReLU  
- Output Layer: 1 neuron, Sigmoid  

Compilation:

- Optimizer: Adam (lr = 0.01)  
- Loss: Binary Crossentropy  
- Metric: Accuracy  

---

## 🏋️ Training Strategy

- Epochs: up to 100  
- Early Stopping:
  - Monitor: `val_loss`  
  - Patience: 5  
  - Restore best weights  

- TensorBoard logging enabled  

Saved artifacts:

- `model.h5`  
- `label_encoder_gender.pkl`  
- `onehot_encoder_geo.pkl`  
- `scaler.pkl`  

---

## 📊 Model Evaluation

The model is evaluated on the test set using:

- Accuracy  
- Validation loss  

Prediction output:

- Churn probability (0 to 1)  
- Final decision:
  - `> 0.5` → Likely to churn  
  - `<= 0.5` → Not likely to churn  


---

## 🚀 Installation & Setup

### 1. Clone repository

```bash
git clone https://github.com/victorjanni/ANN-Classification.git
cd ANN-Classification





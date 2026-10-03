# ✈️ AI-Powered Aviation Analytics

Aircraft Engine Predictive Maintenance using Machine Learning.

🌐 **Live Demo:** https://aviation-predictive-maintenance.streamlit.app/

## 📌 Overview

This project predicts the **Remaining Useful Life (RUL)** of aircraft engines using sensor data from the **NASA C-MAPSS FD001** dataset.

It compares two approaches:

- 🌲 **Random Forest** — tabular baseline
- 🧠 **LSTM** — sequence-based model

The trained models are integrated into a **Streamlit dashboard** where users can select an engine, view its predicted RUL, maintenance status, sensor trends, and model performance.

### Workflow

NASA C-MAPSS FD001 → Preprocessing → Feature Engineering → Random Forest + LSTM → RUL Prediction → Evaluation → Streamlit Dashboard

## 📊 Results

Both models were evaluated on **100 unseen test engines**.

| Model | MAE | RMSE | PHM08 |
|---|---:|---:|---:|
| Random Forest | 13.63 | 18.82 | 662.92 |
| LSTM | **10.63** | **15.24** | **355.81** |

The LSTM is used as the **primary prediction model**, while Random Forest is retained as the baseline.

## 🖥️ Dashboard

The dashboard provides:

- Fleet health overview for 100 engines
- Engine-level RUL prediction
- Maintenance status
- Sensor trend visualization
- Actual vs Predicted RUL
- Prediction error
- Random Forest vs LSTM performance comparison

### Maintenance Status

| Predicted RUL | Status |
|---:|---|
| > 50 | Healthy |
| 30–50 | Monitor |
| 10–30 | Maintenance Recommended |
| < 10 | Critical |

These are project-level interpretation thresholds and are not official aviation maintenance limits.

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- TensorFlow / Keras
- Plotly
- Streamlit
- Joblib

## 📁 Project Structure

AI-Powered--Aviation--Analysis-
│
├── app/
│   └── dashboard.py
├── dashboard/
│   └── app.py
├── data/
│   └── raw/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
├── models/
│   ├── random_forest.pkl
│   └── lstm_final.keras
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluate.py
│   ├── losses.py
│   ├── visualization.py
│   ├── rf_test.py
│   └── lstm_test.py
├── reports/
├── requirements.txt
└── README.md

## 🚀 Run Locally

### 1. Clone the repository

    git clone https://github.com/sukriti-8/AI-Powered--Aviation--Analysis-.git
    cd AI-Powered--Aviation--Analysis-

### 2. Create a virtual environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Start the dashboard

    streamlit run dashboard/app.py

The dashboard will open in your browser.

## 📦 Dataset & Models

The repository includes the NASA C-MAPSS FD001 dataset and the trained model files required to run the dashboard.

The dashboard uses the saved models directly and **does not retrain the models when it starts**.

## ⚠️ Disclaimer

This is an academic/conceptual predictive-maintenance project based on the NASA C-MAPSS FD001 benchmark dataset.

It does not use live aircraft data and should not be treated as a certified aviation maintenance system.

---

**AI-Powered Aviation Analytics — Aircraft Engine Predictive Maintenance**

Python • Machine Learning • LSTM • Random Forest • Streamlit

**Live Dashboard:**

https://aviation-predictive-maintenance.streamlit.app/

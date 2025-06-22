# 🛒 Walmart Sales Forecasting Using Temporal Fusion Transformer (TFT)

This project forecasts Walmart weekly sales using a combination of classical, machine learning, and deep learning models — culminating in a deployed TFT-inspired LSTM model served with **FastAPI** and **Streamlit**.

---

## 📁 Project Structure

```
Time-Series-Sales-Forecasting/
├── models_comparison/        # All models: ARIMA, Prophet, ML, DL
│   ├── Walmart_Time-Series.ipynb
│   ├── project_dataset.csv
│   ├── *.keras, *.h5, *.pth   # Trained models
│
├── tft_deployment/           # End-to-end deployment using FastAPI + Streamlit
│   ├── backend/              # FastAPI backend
│   ├── frontend/             # Streamlit frontend
│   └── requirements.txt      # All dependencies
│
├── Project Description.pdf
├── README.md
```

---

## 📊 Time Series Analysis

- ✅ Trend & seasonality decomposition
- ✅ ACF / PACF plots
- ✅ External regressors: temperature, CPI, unemployment, holidays

---

## 🔍 Models Implemented

| Model              | Type              | RMSE     | MAE     |
|-------------------|-------------------|----------|---------|
| ANN (FNN)          | Deep Learning     | 3186.07  | 2561.58 |
| CNN                | Deep Learning     | 3267.31  | 2653.90 |
| Random Forest      | Machine Learning  | 3772.52  | 3366.71 |
| Prophet            | Statistical       | 17486.18 | 13366.34 |
| **TFT-LSTM**       | Innovative (DL)   | **3247.98** | **2366.94** |

---

## 🚀 Deployment

### 🔗 Streamlit Frontend  
A simple UI to input 24 weeks of data and predict the next 12 weeks of sales.

```bash
cd tft_deployment/frontend
streamlit run app.py
```

### 🔌 FastAPI Backend

Model is served via REST API using FastAPI.

```bash
cd tft_deployment/backend
uvicorn main:app --reload --port 8000
```

---

## 🧠 Innovative Technique: TFT-Inspired LSTM

* ✅ 12-week multi-step forecasting using 24-week input
* ✅ Multi-head attention over historical input
* ✅ Combined LSTM + attention + feedforward layers
* ✅ Categorical embeddings and lag features

---

## ⚙️ Setup & Requirements

```bash
# Install all dependencies
pip install -r tft_deployment/requirements.txt
```

Python 3.10 or higher recommended.


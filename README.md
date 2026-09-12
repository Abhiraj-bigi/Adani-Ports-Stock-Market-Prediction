# 📈 Adani Ports Stock Market Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Regressor-ff69b4.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning project developed for academic portfolio, resume, and technical viva purposes. This project analyzes historical stock market data of **Adani Ports and Special Economic Zone Ltd (`ADANIPORTS.csv`)** from **2007 to 2021** and builds an optimized **XGBoost Regression model** to predict future closing stock prices with **98.97% $R^2$ accuracy**.

---

## 📌 Project Overview

Stock market prediction is a classic time-series regression challenge. This repository implements a complete Machine Learning lifecycle:
- **Data Cleaning & Deduplication**
- **Exploratory Data Analysis (EDA)**
- **Data-Leakage-Free Feature Engineering**
- **Chronological Train-Test Split (80/20)**
- **MinMaxScaler Feature Scaling**
- **Multi-Model Regression Training & Evaluation**
- **Artifact Serialization (Joblib)**
- **Interactive Multi-Page Streamlit Web Dashboard**

---

## 🛠️ Technologies Used

- **Programming Language:** Python 3.11
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn, Plotly Express
- **Machine Learning Libraries:** Scikit-Learn, XGBoost
- **Model Persistence:** Joblib
- **Web Application:** Streamlit
- **Version Control & Cloud Deployment:** Git, GitHub, Streamlit Community Cloud

---

## 📊 Dataset Description

The dataset used is `ADANIPORTS.csv`, containing historical daily trading data from **November 27, 2007 to April 30, 2021**:
- **Total Records:** 3,322 rows
- **Total Features:** 15 columns
- **Key Columns:** `Date`, `Symbol`, `Series`, `Prev Close`, `Open`, `High`, `Low`, `Last`, `Close`, `VWAP`, `Volume`, `Turnover`, `Trades`, `Deliverable Volume`, `%Deliverble`
- **Target Column:** `Close` (Closing Price in INR)

---

## 🛡️ Data Leakage Prevention Strategy

In financial prediction systems, using same-day intraday metrics ($Open_t$, $High_t$, $Low_t$, $Volume_t$) to predict same-day $Close_t$ causes severe **Data Leakage**, as these metrics are unavailable prior to market close.

To ensure realistic, production-ready predictions, all model input features strictly utilize historical information ($t-1$ lag data and rolling windows):
- `Prev_Close`, `Prev_Open`, `Prev_High`, `Prev_Low`, `Prev_Volume` ($t-1$)
- `Daily_Price_Change` ($Prev\_Close - Prev\_Open$)
- `Daily_Pct_Change` ($\frac{Prev\_Close - Prev\_Close_{t-1}}{Prev\_Close_{t-1}} \times 100$)
- `MA_7` (7-Day Moving Average of $Prev\_Close$)
- `MA_30` (30-Day Moving Average of $Prev\_Close$)
- `Rolling_Std_7` (7-Day Rolling Volatility)
- `Year`, `Month`, `Day`, `Day_of_Week`

---

## ⚡ Model Performance & Evaluation

Models were evaluated on a chronological 20% test set (659 trading days from August 2018 to April 2021):

| Model Name | MAE (₹) | MSE | RMSE (₹) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Regressor (Primary)** | **7.3158** | **120.4059** | **10.9730** | **0.9897 (98.97%)** |
| Linear Regression | 6.8275 | 116.1628 | 10.7779 | 0.9901 |
| Gradient Boosting Regressor | 7.7917 | 137.2456 | 11.7152 | 0.9883 |
| Random Forest Regressor | 8.0688 | 150.0817 | 12.2508 | 0.9872 |

---

## 📁 Project Structure

```
Adani-Ports-Stock-Market-Prediction/
│
├── app.py                      # Multi-Page Interactive Streamlit Web Application
├── train_model.py              # End-to-End Machine Learning Pipeline Script
├── requirements.txt            # Python Package Dependencies
├── README.md                   # Project Documentation
├── .gitignore                  # Git Ignore Specifications
│
├── data/
│   └── ADANIPORTS.csv          # Raw Dataset (3,322 rows, 15 columns)
│
├── models/
│   ├── xgboost_model.pkl       # Serialized XGBoost Model
│   ├── scaler.pkl              # Serialized MinMaxScaler Object
│   ├── feature_columns.pkl     # Feature Order List
│   └── model_metrics.pkl       # Serialized Metrics Summary
│
├── notebooks/
│   └── StockMarketPrediction.ipynb # Self-Contained Portfolio Jupyter Notebook
│
└── assets/
    └── images/                 # Generated Charts & Plots
        ├── actual_vs_predicted.png
        ├── feature_importance.png
        ├── closing_price_trend.png
        ├── moving_averages.png
        └── correlation_heatmap.png
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Adani-Ports-Stock-Market-Prediction.git
cd Adani-Ports-Stock-Market-Prediction
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / MacOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model & Generate Visualizations
```bash
python train_model.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🌐 Cloud Deployment (Streamlit Community Cloud)

1. Push your repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Complete Adani Ports Stock Prediction System"
   git branch -M main
   git remote add origin https://github.com/your-username/Adani-Ports-Stock-Market-Prediction.git
   git push -u origin main
   ```
2. Navigate to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account and select your repository `Adani-Ports-Stock-Market-Prediction`.
4. Set Main file path to `app.py` and click **Deploy**.

---

## ⚠️ Financial Disclaimer

This application and its underlying machine learning predictions are generated strictly for educational, portfolio, and academic demonstration purposes. Stock trading involves substantial market risk. Predictions from this model should **never** be used as investment advice or financial guidance.

---

## 👨‍💻 Author

**Abhiraj Kumar**  
*B.Tech Computer Science Engineering (CSE)*  
*Specialization in Data Science & Machine Learning*


# 📊 Restaurant Foot Traffic Prediction and Analysis System

A full-stack data science web application built for CS506 at Boston University. This system forecasts daily restaurant foot traffic based on weather conditions and calendar context using machine learning. It features a prediction interface, historical trend analysis, and model interpretability.

---
Youtube Link:https://youtu.be/p3iGm5mgMyw

## Project Overview

Restaurants often struggle to predict how many customers will visit on a given day. Weather, holidays, and weekends significantly impact traffic. Our project combines machine learning with historical data and weather features to help restaurant managers make better staffing and inventory decisions.

---

## Team Members

- **Jaile Quan**  
- **Jiahao Wang**  
- **Jiaqing Xu**

---

## Key Features

- Analysis dashboard for time trends and seasonal insights
- Real-time foot traffic prediction
- Model performance comparison and feature importance
- Historical data viewer

---

## Tech Stack

| Layer        | Tools Used                              |
|--------------|------------------------------------------|
| Frontend     | HTML, Bootstrap, JavaScript, Jinja2      |
| Backend      | Flask                                    |
| ML Models    | XGBoost, Gradient Boosting, Random Forest, Ridge, Linear Regression |
| Visuals      | Matplotlib, Seaborn                      |
| Data         | Pandas, NumPy, holidays                  |

---

## Project Structure

```
├── app.py
├── data_preprocessing.py
├── exploratory_data_analysis.py
├── modeling.py
├── templates/index.html
├── static/images/
├── model_outputs/
│   ├── best_foot_traffic_model.joblib
│   └── *.png
├── simulated_foot_traffic_enhanced.csv
└── requirements.txt
```

---

## How to Replicate Results

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CS506-FinalProject.git
cd CS506-FinalProject
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

If needed:

```bash
pip install flask pandas numpy matplotlib seaborn scikit-learn xgboost holidays joblib
```

### 4. Add data

Include `simulated_foot_traffic_enhanced.csv` in the root directory with at least:

```csv
date,temperature,humidity,precipitation
2023-01-01,4.5,78,0.1
2023-01-02,2.0,60,0.0
...
```

### 5. Run the pipeline

```bash
python data_preprocessing.py
python exploratory_data_analysis.py
python modeling.py
```

### 6. Start the app

```bash
python app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Outputs

| File                                | Description                          |
|-------------------------------------|--------------------------------------|
| `processed_foot_traffic.csv`        | Cleaned, feature-rich dataset        |
| `best_foot_traffic_model.joblib`    | Saved ML model used in app           |
| `static/images/*.png`               | Visual charts                        |
| `model_outputs/*.png`               | Model comparison and importance plots|

---

## Model Performance Summary

| Model              | RMSE    | MAE    | R² Score |
|--------------------|---------|--------|----------|
| Linear Regression  | ~11.5   | ~9.2   | 0.69     |
| Ridge Regression   | ~11.3   | ~9.0   | 0.71     |
| Random Forest      | ~10.6   | ~8.5   | 0.74     |
| Gradient Boosting  | ~10.4   | ~8.4   | 0.74     |
| **XGBoost**        | **10.26** | **8.44** | **0.75 ✅** |

---

## One-Command Pipeline

```bash
python data_preprocessing.py && \
python exploratory_data_analysis.py && \
python modeling.py && \
python app.py
```

---


## Acknowledgments

This project was built as a final project for **CS506: Data Science with Python** at Boston University.  
Special thanks to our professor and TAs for their guidance and feedback.

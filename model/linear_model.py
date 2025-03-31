# linear_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("../data/simulated_foot_traffic_feb1_to_mar15_2025.csv")

# Features and target
X = df[['temperature', 'humidity', 'precipitation']]
y = df['foot_traffic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Linear Regression RMSE: {rmse:.2f}")

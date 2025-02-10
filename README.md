# CS506-FinalProject
Jiale Quan
Jiahao Wang
Jiaqing Xu

# Proposal:

# Predicting Daily Restaurant Foot Traffic

## Description of the Project
Understanding and predicting restaurant foot traffic is crucial for optimizing staffing, inventory management, and improving customer experience. This project aims to develop a machine learning model to predict the number of daily customers for a specific restaurant based on external factors such as weather conditions and online reviews. By analyzing historical data, our model will help restaurant owners make data-driven decisions to improve efficiency and profitability.

## Clear Goal(s)
The primary goal of this project is to accurately predict the number of customers visiting a restaurant on a given day using a combination of weather data and social media activity. Specifically, we aim to:
- Predict daily customer foot traffic based on historical weather data and restaurant reviews.
- Identify key external factors that significantly impact restaurant attendance.
- Provide data-driven insights to optimize staffing and inventory management.

## Data Collection
To train and validate our model, we will collect data from the following sources:
- **Yelp API**: Retrieve restaurant reviews, ratings, and peak visit times.
- **OpenWeatherMap API**: Gather historical and real-time weather data, including temperature, precipitation, and humidity.
- **Restaurant Sales Data (If available)**: If possible, we will collect historical sales or reservation data to improve prediction accuracy. If real-world sales data is not available, we may simulate data based on general trends.

We will extract and preprocess this data to create a structured dataset that aligns customer visit counts with weather conditions and online engagement.

## Modeling Approach
We will explore various modeling techniques to predict daily foot traffic, including:
- **Linear Regression**: Establish a simple baseline model to observe correlations between factors.
- **XGBoost**: A powerful gradient boosting technique to capture non-linear patterns.
- **Prophet (Time-Series Forecasting Model)**: Designed for predicting time-series data, suitable for capturing seasonal variations in customer visits.

We will experiment with these models and evaluate their performance based on standard metrics such as RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

## Data Visualization Plan
To better understand patterns and model predictions, we will implement various visualizations, including:
- **Time Series Plots**: Display foot traffic trends over time.
- **Scatter Plots**: Show relationships between weather factors (temperature, humidity) and customer visits.
- **Heatmaps**: Illustrate correlations between different variables affecting restaurant attendance.
- **Interactive Dashboards**: If time permits, we will build an interactive tool that allows users to explore predicted foot traffic under different weather conditions.

## Test Plan
To ensure the robustness of our model, we will:
- **Split the dataset into 80% training and 20% testing**.
- **Use time-based validation**, where models are trained on past months and tested on the most recent months.
- **Evaluate different models using RMSE, MAE, and R-squared scores** to select the best-performing approach.
- **Perform seasonal comparisons**, such as training on summer data and testing on winter data, to assess model generalizability.

## Timeline and Scope
This project will span approximately **two months**, during which we will:
1. **Week 1-2**: Data collection and preprocessing.
2. **Week 3-4**: Initial exploratory data analysis and feature engineering.
3. **Week 5-6**: Train and refine machine learning models.
4. **Week 7**: Evaluate model performance and fine-tune hyperparameters.
5. **Week 8**: Finalize the report and create visualizations.

By the end of this project, we aim to deliver a **predictive model**, **a comprehensive report**, and **visual insights** into restaurant foot traffic patterns.




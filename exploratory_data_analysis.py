import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import os


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def load_processed_data(filepath):
    """Load processed data"""
    return pd.read_csv(filepath, parse_dates=['date'])

def time_series_analysis(df):
    """Time series analysis showing foot traffic trends over time"""
    plt.figure(figsize=(15, 6))
    plt.plot(df['date'], df['foot_traffic'])
    plt.title('Foot Traffic Trends Over Time', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Foot Traffic')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis date format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_PATH, 'time_series.png'))
    plt.close()

def weather_impact_analysis(df):
    """Analyze the impact of weather factors on foot traffic"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Impact of temperature on foot traffic
    sns.scatterplot(x='temperature', y='foot_traffic', data=df, ax=axes[0], alpha=0.6)
    axes[0].set_title('Temperature vs Foot Traffic')
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Foot Traffic')
    
    # 2. Impact of humidity on foot traffic
    sns.scatterplot(x='humidity', y='foot_traffic', data=df, ax=axes[1], alpha=0.6)
    axes[1].set_title('Humidity vs Foot Traffic')
    axes[1].set_xlabel('Humidity (%)')
    axes[1].set_ylabel('Foot Traffic')
    
    # 3. Impact of precipitation on foot traffic
    # Only consider days with precipitation
    rain_df = df[df['precipitation'] > 0]
    sns.scatterplot(x='precipitation', y='foot_traffic', data=rain_df, ax=axes[2], alpha=0.6, color='blue')
    axes[2].set_title('Precipitation vs Foot Traffic')
    axes[2].set_xlabel('Precipitation (mm)')
    axes[2].set_ylabel('Foot Traffic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_PATH, 'weather_impact.png'))
    plt.close()

def seasonal_patterns(df):
    """Analyze seasonal patterns"""
    # Calculate average foot traffic by month and day of week
    monthly_avg = df.groupby('month')['foot_traffic'].mean().reset_index()
    weekday_avg = df.groupby('day_of_week')['foot_traffic'].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Monthly average foot traffic
    sns.barplot(x='month', y='foot_traffic', data=monthly_avg, ax=axes[0])
    axes[0].set_title('Monthly Average Foot Traffic')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Average Foot Traffic')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Day of week average foot traffic
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.barplot(x='day_of_week', y='foot_traffic', data=weekday_avg, ax=axes[1])
    axes[1].set_title('Average Foot Traffic by Day of Week')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Average Foot Traffic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_PATH, 'seasonal_patterns.png'))
    plt.close()

def weather_category_analysis(df):
    """Analyze impact of different weather categories on foot traffic"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Impact of temperature categories
    sns.boxplot(x='temp_category', y='foot_traffic', data=df, ax=axes[0])
    axes[0].set_title('Foot Traffic Distribution by Temperature Category')
    axes[0].set_xlabel('Temperature Category')
    axes[0].set_ylabel('Foot Traffic')
    
    # Impact of precipitation categories
    sns.boxplot(x='rain_category', y='foot_traffic', data=df, ax=axes[1])
    axes[1].set_title('Foot Traffic Distribution by Precipitation Category')
    axes[1].set_xlabel('Precipitation Category')
    axes[1].set_ylabel('Foot Traffic')
    
    # Impact of humidity categories
    sns.boxplot(x='humidity_category', y='foot_traffic', data=df, ax=axes[2])
    axes[2].set_title('Foot Traffic Distribution by Humidity Category')
    axes[2].set_xlabel('Humidity Category')
    axes[2].set_ylabel('Foot Traffic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_PATH, 'weather_category_impact.png'))
    plt.close()

def holiday_vs_regular_analysis(df):
    """Analyze differences in foot traffic between holidays and regular days"""
    plt.figure(figsize=(10, 6))
    
    # Create grouping variable: Weekday, Weekend (non-holiday), Holiday
    df['day_type'] = 'Weekday'  # Default to weekday
    df.loc[df['is_weekend'] == 1, 'day_type'] = 'Weekend'  # Weekend
    df.loc[df['is_holiday'] == 1, 'day_type'] = 'Holiday'  # Holiday
    
    # Plot boxplot for different day types
    sns.boxplot(x='day_type', y='foot_traffic', data=df, order=['Weekday', 'Weekend', 'Holiday'])
    plt.title('Foot Traffic Comparison: Weekdays, Weekends, and Holidays')
    plt.xlabel('Day Type')
    plt.ylabel('Foot Traffic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_PATH, 'holiday_impact.png'))
    plt.close()
    
    return df

def correlation_analysis(df):
    """Correlation analysis"""
    # Select numeric features for correlation analysis
    numeric_features = ['temperature', 'humidity', 'precipitation', 'foot_traffic', 
                        'day_of_week', 'is_weekend', 'month', 'is_holiday']
    
    # Calculate correlation coefficients
    corr = df[numeric_features].corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_PATH, 'correlation_heatmap.png'))
    plt.close()

def run_all_analyses(filepath):
    """Run all analyses"""
    df = load_processed_data(filepath)
    
    print("Running time series analysis...")
    time_series_analysis(df)
    
    print("Analyzing weather impact on foot traffic...")
    weather_impact_analysis(df)
    
    print("Analyzing seasonal patterns...")
    seasonal_patterns(df)
    
    print("Analyzing impact of weather categories...")
    weather_category_analysis(df)
    
    print("Analyzing holiday vs regular day differences...")
    df = holiday_vs_regular_analysis(df)
    
    print("Running correlation analysis...")
    correlation_analysis(df)
    
    print("All analyses completed!")
    
    return df

if __name__ == "__main__":
    # If processed data exists, load it directly
    try:
        processed_file = os.path.join(ROOT_PATH, "processed_foot_traffic.csv")
        df = run_all_analyses(processed_file)
    except FileNotFoundError:
        print("Processed data file not found. Please run data_preprocessing.py first")

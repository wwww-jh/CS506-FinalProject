import pandas as pd
import numpy as np
from datetime import datetime
import holidays
import os

# Define root path
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def load_data(filepath):
    """Load CSV data and process date column"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def add_date_features(df):
    """Add date-related features"""
    # Add day of week feature (0=Monday, 6=Sunday)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add month and season
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].apply(lambda x: 
        'Winter' if x in [12, 1, 2] else 
        'Spring' if x in [3, 4, 5] else 
        'Summer' if x in [6, 7, 8] else 'Fall')
    
    # Add US holiday flag
    us_holidays = holidays.US()
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in us_holidays else 0)
    
    return df

def engineer_weather_features(df):
    """Engineer weather-related features"""
    # Temperature categories
    df['temp_category'] = pd.cut(
        df['temperature'], 
        bins=[-20, 0, 10, 20, 35], 
        labels=['Cold', 'Cool', 'Mild', 'Hot']
    )
    
    # Precipitation categories
    df['rain_category'] = pd.cut(
        df['precipitation'], 
        bins=[-0.01, 0.01, 0.5, 1, 50], 
        labels=['None', 'Light', 'Moderate', 'Heavy']
    )
    
    # Humidity categories
    df['humidity_category'] = pd.cut(
        df['humidity'], 
        bins=[0, 40, 70, 101], 
        labels=['Low', 'Medium', 'High']
    )
    
    return df

def process_data(filepath):
    """Main function for data processing"""
    df = load_data(filepath)
    df = add_date_features(df)
    df = engineer_weather_features(df)
    
    # Check for missing dates
    df = handle_missing_dates(df)
    
    return df

def handle_missing_dates(df):
    """Handle missing dates in the dataset"""
    # Get date range
    start_date = df['date'].min()
    end_date = df['date'].max()
    
    # Create complete date range
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Check missing dates
    missing_dates = set(full_date_range) - set(df['date'])
    print(f"Found {len(missing_dates)} missing dates")
    
    if missing_dates:
        print("Missing dates examples:", sorted(list(missing_dates))[:5], "...")
    
    return df

if __name__ == "__main__":
    # Define input and output paths relative to root
    input_file = os.path.join(ROOT_PATH, "simulated_foot_traffic_enhanced.csv")
    df = process_data(input_file)
    
    # Save processed data
    output_file = os.path.join(ROOT_PATH, "processed_foot_traffic.csv")
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    # Display dataset info
    print("\nDataset Information:")
    print(f"Number of records: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print("\nFeature list:")
    for col in df.columns:
        print(f"- {col}")

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import holidays

app = Flask(__name__)

# Define project root path
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# Global variables with relative paths
MODEL_PATH = os.path.join(ROOT_PATH, "model_outputs", "best_foot_traffic_model.joblib")
DATA_PATH = os.path.join(ROOT_PATH, "processed_foot_traffic.csv")
IMAGES_DIR = ROOT_PATH
MODEL_OUTPUTS_DIR = os.path.join(ROOT_PATH, "model_outputs")

# Global variables to store model and data
model = None
df = None
feature_names = []

# Load model and data function (no decorator used)
def load_model_and_data():
    global model, df, feature_names
    
    # Load model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = None
        print(f"Warning: Model file {MODEL_PATH} does not exist")
    
    # Load data
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=['date'])
        # Get feature names (excluding date and foot_traffic)
        feature_names = [col for col in df.columns if col not in ['date', 'foot_traffic']]
    else:
        df = None
        feature_names = []
        print(f"Warning: Data file {DATA_PATH} does not exist")

# Home page route
@app.route('/')
def index():
    # Get image list
    image_paths = {
        'time_series': os.path.join(IMAGES_DIR, 'time_series.png'),
        'weather_impact': os.path.join(IMAGES_DIR, 'weather_impact.png'),
        'seasonal_patterns': os.path.join(IMAGES_DIR, 'seasonal_patterns.png'),
        'weather_category_impact': os.path.join(IMAGES_DIR, 'weather_category_impact.png'),
        'holiday_impact': os.path.join(IMAGES_DIR, 'holiday_impact.png'),
        'correlation_heatmap': os.path.join(IMAGES_DIR, 'correlation_heatmap.png'),
    }
    
    # Model comparison images
    model_comparison_images = {
        'xgboost_pred': os.path.join(MODEL_OUTPUTS_DIR, 'XGBoost_actual_vs_predicted.png'),
        'gradient_boosting_pred': os.path.join(MODEL_OUTPUTS_DIR, 'Gradient_Boosting_actual_vs_predicted.png'),
        'random_forest_pred': os.path.join(MODEL_OUTPUTS_DIR, 'Random_Forest_actual_vs_predicted.png'),
        'ridge_pred': os.path.join(MODEL_OUTPUTS_DIR, 'Ridge_Regression_actual_vs_predicted.png'),
        'linear_pred': os.path.join(MODEL_OUTPUTS_DIR, 'Linear_Regression_actual_vs_predicted.png'),
        'xgboost_feat': os.path.join(MODEL_OUTPUTS_DIR, 'XGBoost_feature_importance.png'),
        'random_forest_feat': os.path.join(MODEL_OUTPUTS_DIR, 'Random_Forest_feature_importance.png'),
        'gradient_boosting_feat': os.path.join(MODEL_OUTPUTS_DIR, 'Gradient_Boosting_feature_importance.png'),
    }
    
    # Check if image files exist and convert to relative URLs
    available_images = {}
    for name, path in image_paths.items():
        if os.path.exists(path):
            # Ensure static/images directory exists
            if not os.path.exists("static/images"):
                os.makedirs("static/images", exist_ok=True)
            
            # Copy the file to static/images if it doesn't exist
            static_path = os.path.join("static/images", os.path.basename(path))
            if not os.path.exists(static_path):
                import shutil
                shutil.copyfile(path, static_path)
            
            # Convert path to relative URL
            available_images[name] = f'static/images/{os.path.basename(path)}'
    
    # Check model comparison images and copy them to static directory
    model_images = {}
    for name, path in model_comparison_images.items():
        if os.path.exists(path):
            # Ensure static/images directory exists
            if not os.path.exists("static/images"):
                os.makedirs("static/images", exist_ok=True)
            
            # Copy the file to static/images if it doesn't exist
            static_path = os.path.join("static/images", os.path.basename(path))
            if not os.path.exists(static_path):
                import shutil
                shutil.copyfile(path, static_path)
            
            # Convert path to relative URL
            model_images[name] = f'static/images/{os.path.basename(path)}'
    
    # Get model performance information
    model_info = {}
    info_path = os.path.join(MODEL_OUTPUTS_DIR, 'model_info.txt')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    model_info[key.strip()] = value.strip()
    
    return render_template('index.html', 
                          images=available_images,
                          model_images=model_images,
                          model_info=model_info)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not feature_names:
        return jsonify({'error': 'Model or feature names not loaded'}), 500
    
    try:
        # Get form data
        data = request.form.to_dict()
        prediction_date = datetime.strptime(data['date'], '%Y-%m-%d')
        
        # Create prediction features
        features = {}
        
        # Add basic features
        features['temperature'] = float(data['temperature'])
        features['humidity'] = float(data['humidity'])
        features['precipitation'] = float(data['precipitation'])
        
        # Add date features
        features['day_of_week'] = prediction_date.weekday()
        features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0
        features['month'] = prediction_date.month
        
        # Add holiday features
        us_holidays = holidays.US()
        features['is_holiday'] = 1 if prediction_date in us_holidays else 0
        
        # Add temperature category
        temp = features['temperature']
        if temp < 0:
            features['temp_category_Cold'] = 1
            features['temp_category_Cool'] = 0
            features['temp_category_Mild'] = 0
            features['temp_category_Hot'] = 0
        elif temp < 10:
            features['temp_category_Cold'] = 0
            features['temp_category_Cool'] = 1
            features['temp_category_Mild'] = 0
            features['temp_category_Hot'] = 0
        elif temp < 20:
            features['temp_category_Cold'] = 0
            features['temp_category_Cool'] = 0
            features['temp_category_Mild'] = 1
            features['temp_category_Hot'] = 0
        else:
            features['temp_category_Cold'] = 0
            features['temp_category_Cool'] = 0
            features['temp_category_Mild'] = 0
            features['temp_category_Hot'] = 1
        
        # Add precipitation category
        precip = features['precipitation']
        if precip <= 0.01:
            features['rain_category_None'] = 1
            features['rain_category_Light'] = 0
            features['rain_category_Moderate'] = 0
            features['rain_category_Heavy'] = 0
        elif precip <= 0.5:
            features['rain_category_None'] = 0
            features['rain_category_Light'] = 1
            features['rain_category_Moderate'] = 0
            features['rain_category_Heavy'] = 0
        elif precip <= 1:
            features['rain_category_None'] = 0
            features['rain_category_Light'] = 0
            features['rain_category_Moderate'] = 1
            features['rain_category_Heavy'] = 0
        else:
            features['rain_category_None'] = 0
            features['rain_category_Light'] = 0
            features['rain_category_Moderate'] = 0
            features['rain_category_Heavy'] = 1
        
        # Add humidity category
        humid = features['humidity']
        if humid <= 40:
            features['humidity_category_Low'] = 1
            features['humidity_category_Medium'] = 0
            features['humidity_category_High'] = 0
        elif humid <= 70:
            features['humidity_category_Low'] = 0
            features['humidity_category_Medium'] = 1
            features['humidity_category_High'] = 0
        else:
            features['humidity_category_Low'] = 0
            features['humidity_category_Medium'] = 0
            features['humidity_category_High'] = 1
        
        # Add season
        month = features['month']
        if month in [12, 1, 2]:
            features['season_Winter'] = 1
            features['season_Spring'] = 0
            features['season_Summer'] = 0
            features['season_Fall'] = 0
        elif month in [3, 4, 5]:
            features['season_Winter'] = 0
            features['season_Spring'] = 1
            features['season_Summer'] = 0
            features['season_Fall'] = 0
        elif month in [6, 7, 8]:
            features['season_Winter'] = 0
            features['season_Spring'] = 0
            features['season_Summer'] = 1
            features['season_Fall'] = 0
        else:
            features['season_Winter'] = 0
            features['season_Spring'] = 0
            features['season_Summer'] = 0
            features['season_Fall'] = 1
        
        # Create feature vector
        X = pd.DataFrame([features])
        
        # Ensure feature order matches training
        X = X.reindex(columns=feature_names, fill_value=0)
        
        # Prediction
        # Before prediction, ensure feature names match exactly
        # Print model's expected feature names for debugging
        if hasattr(model, 'feature_names_in_'):
            print("Model's expected feature names:", model.feature_names_in_)
        
        # Use a more reliable method to ensure feature columns match
        # Create a feature DataFrame that exactly matches model expectations - fix feature order to match model
        expected_features = ['temperature', 'humidity', 'precipitation', 'day_of_week', 
                            'is_weekend', 'month', 'is_holiday', 
                            'temp_category_Cold', 'temp_category_Cool', 'temp_category_Hot', 'temp_category_Mild', 
                            'rain_category_Heavy', 'rain_category_Light', 'rain_category_Moderate', 
                            'humidity_category_High', 'humidity_category_Low', 'humidity_category_Medium', 
                            'season_Fall', 'season_Spring', 'season_Summer', 'season_Winter']
        
        # Create an empty DataFrame with all expected features
        prediction_data = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # Fill numeric features
        for feature in ['temperature', 'humidity', 'precipitation', 'day_of_week', 'is_weekend', 'month', 'is_holiday']:
            if feature in features:
                prediction_data[feature] = features[feature]
        
        # Set temperature category
        temp = features['temperature']
        if temp < 0:
            prediction_data['temp_category_Cold'] = 1
        elif temp < 10:
            prediction_data['temp_category_Cool'] = 1
        elif temp < 20:
            prediction_data['temp_category_Mild'] = 1
        else:
            prediction_data['temp_category_Hot'] = 1
        
        # Set precipitation category - modify precipitation category handling, remove rain_category_None
        precip = features['precipitation']
        if precip <= 0.01:
            # For no precipitation, don't set any rain_category features
            # Model training may have set them all to 0 to represent no precipitation
            pass
        elif precip <= 0.5:
            prediction_data['rain_category_Light'] = 1
        elif precip <= 1:
            prediction_data['rain_category_Moderate'] = 1
        else:
            prediction_data['rain_category_Heavy'] = 1
        
        # Set humidity category
        humid = features['humidity']
        if humid <= 40:
            prediction_data['humidity_category_Low'] = 1
        elif humid <= 70:
            prediction_data['humidity_category_Medium'] = 1
        else:
            prediction_data['humidity_category_High'] = 1
        
        # Set season
        month = features['month']
        if month in [12, 1, 2]:
            prediction_data['season_Winter'] = 1
        elif month in [3, 4, 5]:
            prediction_data['season_Spring'] = 1
        elif month in [6, 7, 8]:
            prediction_data['season_Summer'] = 1
        else:
            prediction_data['season_Fall'] = 1
        
        # Make prediction
        foot_traffic_prediction = round(float(model.predict(prediction_data)[0]))
        
        # Format date for response
        formatted_date = prediction_date.strftime('%Y-%m-%d')
        
        # Determine day type
        if features['is_holiday'] == 1:
            day_type = "Holiday"
        elif features['is_weekend'] == 1:
            day_type = "Weekend"
        else:
            day_type = "Weekday"
        
        # Determine temperature category
        if temp < 0:
            temp_category = "Cold"
        elif temp < 10:
            temp_category = "Cool"
        elif temp < 20:
            temp_category = "Mild"
        else:
            temp_category = "Hot"
        
        # Determine precipitation category
        if precip <= 0.01:
            rain_category = "None"
        elif precip <= 0.5:
            rain_category = "Light"
        elif precip <= 1:
            rain_category = "Moderate"
        else:
            rain_category = "Heavy"
        
        # Determine humidity category
        if humid <= 40:
            humidity_category = "Low"
        elif humid <= 70:
            humidity_category = "Medium"
        else:
            humidity_category = "High"
        
        # Find similar historical days (if data is available)
        similar_days = []
        if df is not None:
            # Filter by similar conditions: same season, similar temperature (±5°C)
            if month in [12, 1, 2]:
                season_filter = df['month'].isin([12, 1, 2])
            elif month in [3, 4, 5]:
                season_filter = df['month'].isin([3, 4, 5])
            elif month in [6, 7, 8]:
                season_filter = df['month'].isin([6, 7, 8])
            else:
                season_filter = df['month'].isin([9, 10, 11])
            
            # Filter by similar day type
            if features['is_holiday'] == 1:
                day_filter = df['is_holiday'] == 1
            elif features['is_weekend'] == 1:
                day_filter = df['is_weekend'] == 1
            else:
                day_filter = (df['is_weekend'] == 0) & (df['is_holiday'] == 0)
            
            # Filter by similar temperature and precipitation
            temp_filter = (df['temperature'] >= temp - 5) & (df['temperature'] <= temp + 5)
            
            # Combine filters
            similar_days = df[season_filter & day_filter & temp_filter]
        
        avg_foot_traffic = 0
        if len(similar_days) > 0:
            avg_foot_traffic = round(similar_days['foot_traffic'].mean())
        
        response = {
            'prediction': str(foot_traffic_prediction),
            'date': formatted_date,
            'day_type': day_type,
            'holiday': "Yes" if features['is_holiday'] == 1 else "No",
            'weather_info': {
                'temperature': features['temperature'],
                'humidity': features['humidity'],
                'precipitation': features['precipitation'],
                'temp_category': temp_category,
                'humidity_category': humidity_category,
                'rain_category': rain_category
            },
            'historical_reference': {
                'similar_days_count': len(similar_days),
                'avg_foot_traffic': avg_foot_traffic if len(similar_days) > 0 else "N/A"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical_data')
def historical_data():
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Get the most recent 30 days of data
        recent_data = df.sort_values('date', ascending=False).copy()
        
        # Format data for JSON response
        result = []
        for _, row in recent_data.iterrows():
            result.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'foot_traffic': int(row['foot_traffic']),
                'temperature': float(row['temperature']),
                'humidity': float(row['humidity']),
                'precipitation': float(row['precipitation']),
                'is_weekend': bool(row['is_weekend']),
                'is_holiday': bool(row['is_holiday'])
            })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ensure model outputs directory exists
@app.route('/check_images')
def check_images():
    image_dirs = ["static", "static/images"]
    for dir_path in image_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Create symlinks to the image files
    image_files = [
        'time_series.png', 
        'weather_impact.png', 
        'seasonal_patterns.png',
        'weather_category_impact.png', 
        'holiday_impact.png', 
        'correlation_heatmap.png'
    ]
    
    # Add model comparison images
    model_comparison_images = [
        'XGBoost_actual_vs_predicted.png',
        'Gradient_Boosting_actual_vs_predicted.png',
        'Random_Forest_actual_vs_predicted.png',
        'Ridge_Regression_actual_vs_predicted.png',
        'Linear_Regression_actual_vs_predicted.png',
        'XGBoost_feature_importance.png',
        'Random_Forest_feature_importance.png',
        'Gradient_Boosting_feature_importance.png'
    ]
    
    image_files.extend(model_comparison_images)
    
    results = []
    for image_file in image_files:
        src_path = os.path.join(IMAGES_DIR, image_file)
        model_src_path = os.path.join(MODEL_OUTPUTS_DIR, image_file)
        dst_path = os.path.join("static/images", image_file)
        
        # Check if image exists in main directory
        if os.path.exists(src_path):
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)
            results.append(f"Linked {image_file} from main directory")
        # Check if image exists in model_outputs directory
        elif os.path.exists(model_src_path):
            if not os.path.exists(dst_path):
                os.symlink(model_src_path, dst_path)
            results.append(f"Linked {image_file} from model_outputs")
        else:
            results.append(f"Image {image_file} not found")
    
    return jsonify({"status": "success", "results": results})

if __name__ == "__main__":
    # Load model and data before starting the app
    load_model_and_data()
    
    # Create static/images directory and symlinks
    if not os.path.exists("static/images"):
        os.makedirs("static/images", exist_ok=True)
    
    app.run(debug=True, port=5000)

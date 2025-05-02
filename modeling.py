import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def load_data(filepath):
    """Load processed data"""
    df = pd.read_csv(filepath, parse_dates=['date'])
    return df

def prepare_features_target(df):
    """Prepare features and target variable"""
    # Convert categorical features to numeric encoding
    df = pd.get_dummies(df, columns=['temp_category', 'rain_category', 'humidity_category', 'season'])
    
    # Select feature variables
    features = [col for col in df.columns if col not in ['date', 'foot_traffic']]
    
    X = df[features]
    y = df['foot_traffic']
    
    return X, y, features

def train_evaluate_models(X_train, X_test, y_train, y_test, features):
    """Train and evaluate various models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
        
        # Training set evaluation
        train_preds = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_mae = mean_absolute_error(y_train, train_preds)
        train_r2 = r2_score(y_train, train_preds)
        
        # Test set evaluation
        test_preds = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        results[name] = {
            'model': model,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_preds': test_preds
        }
        
        print(f"  {name} test set RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.2f}")
    
    return results

def plot_feature_importance(models, features, save_path):
    """Plot feature importance chart"""
    # Only select models with feature importance attribute
    for name, model_data in models.items():
        model = model_data['model']
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            
            # Get feature importance
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot top 15 important features
            top_n = min(15, len(features))
            plt.title(f'{name} - Feature Importance Ranking')
            plt.barh(range(top_n), importances[indices][:top_n], align='center')
            plt.yticks(range(top_n), [features[i] for i in indices[:top_n]])
            plt.xlabel('Feature Importance')
            plt.gca().invert_yaxis()  # Most important features shown at top
            plt.tight_layout()
            
            plt.savefig(os.path.join(save_path, f'{name.replace(" ", "_")}_feature_importance.png'))
            plt.close()

def plot_actual_vs_predicted(y_test, predictions, model_name, save_path):
    """Plot actual vs predicted values comparison chart"""
    plt.figure(figsize=(12, 6))
    
    # Find index range
    indices = range(len(y_test))
    
    # Plot actual and predicted values
    plt.plot(indices, y_test.values, label='Actual Values', color='blue')
    plt.plot(indices, predictions, label='Predicted Values', color='red', alpha=0.7)
    
    plt.title(f'{model_name} - Actual vs Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Foot Traffic')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f'{model_name.replace(" ", "_")}_actual_vs_predicted.png'))
    plt.close()

def save_best_model(models, save_path):
    """Save the best performing model"""
    # Select best model based on test set R²
    best_model_name = max(models, key=lambda name: models[name]['test_r2'])
    best_model = models[best_model_name]['model']
    
    print(f"Best model is {best_model_name}, test set R²: {models[best_model_name]['test_r2']:.2f}")
    
    # Save model
    model_path = os.path.join(save_path, 'best_foot_traffic_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")
    
    # Create model info file
    model_info = {
        'model_name': best_model_name,
        'test_rmse': models[best_model_name]['test_rmse'],
        'test_mae': models[best_model_name]['test_mae'],
        'test_r2': models[best_model_name]['test_r2']
    }
    
    info_path = os.path.join(save_path, 'model_info.txt')
    with open(info_path, 'w') as f:
        for k, v in model_info.items():
            f.write(f"{k}: {v}\n")
    
    return best_model_name, best_model

def run_modeling(filepath, output_path=os.path.join(ROOT_PATH, "model_outputs")):
    """Run the modeling process"""
    # Create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load data
    df = load_data(filepath)
    print(f"Loaded {len(df)} records")
    
    # Prepare features and target variable
    X, y, features = prepare_features_target(df)
    print(f"Prepared {len(features)} features")
    
    # Split data
    # Use time series split to maintain time order
    df = df.sort_values('date')
    train_size = int(0.8 * len(df))
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Train and evaluate models
    models = train_evaluate_models(X_train, X_test, y_train, y_test, features)
    
    # Plot feature importance
    plot_feature_importance(models, features, output_path)
    
    # Plot actual vs predicted comparison for each model
    for name, model_data in models.items():
        plot_actual_vs_predicted(y_test, model_data['test_preds'], name, output_path)
    
    # Save best model
    best_model_name, best_model = save_best_model(models, output_path)
    
    return best_model_name, best_model

if __name__ == "__main__":
    processed_file = os.path.join(ROOT_PATH, "processed_foot_traffic.csv")
    output_dir = os.path.join(ROOT_PATH, "model_outputs")
    
    try:
        best_model_name, _ = run_modeling(processed_file, output_dir)
        print(f"Modeling completed, best model: {best_model_name}")
    except Exception as e:
        print(f"Error during modeling process: {str(e)}")

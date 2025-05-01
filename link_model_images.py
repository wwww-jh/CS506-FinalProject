import requests
import os
import time
import sys
import shutil

# URL of the Flask application
BASE_URL = "http://localhost:5000"
MODEL_OUTPUTS_DIR = "c:\\Users\\llj\\Desktop\\sihuo\\2025\\4.4\\22\\model_outputs"
STATIC_IMAGES_DIR = "static/images"

def copy_model_images():
    """Copy model comparison images to static directory"""
    try:
        print("Copying model comparison images...")
        
        # Create static images directory if it doesn't exist
        if not os.path.exists(STATIC_IMAGES_DIR):
            os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
        
        # List of model comparison images
        model_images = [
            'XGBoost_actual_vs_predicted.png',
            'Gradient_Boosting_actual_vs_predicted.png',
            'Random_Forest_actual_vs_predicted.png',
            'Ridge_Regression_actual_vs_predicted.png',
            'Linear_Regression_actual_vs_predicted.png',
            'XGBoost_feature_importance.png',
            'Random_Forest_feature_importance.png',
            'Gradient_Boosting_feature_importance.png'
        ]
        
        # Copy each image
        copied_count = 0
        for image_file in model_images:
            src_path = os.path.join(MODEL_OUTPUTS_DIR, image_file)
            dst_path = os.path.join(STATIC_IMAGES_DIR, image_file)
            
            if os.path.exists(src_path):
                shutil.copyfile(src_path, dst_path)
                print(f"  Copied: {image_file}")
                copied_count += 1
            else:
                print(f"  Missing: {image_file}")
        
        if copied_count > 0:
            print(f"Successfully copied {copied_count} images.")
            return True
        else:
            print("No images were found to copy.")
            return False
    except Exception as e:
        print(f"Error copying images: {str(e)}")
        return False

def wait_for_server(timeout=5):
    """Wait for the Flask server to start"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/")
            if response.status_code == 200:
                print("Server is running.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print("Waiting for server to start...")
        time.sleep(1)
    
    print(f"Timeout after {timeout} seconds. Server may not be running.")
    return False

if __name__ == "__main__":
    print("Model Comparison Image Copy Tool")
    print("================================")
    print("This tool copies model comparison images to make them available in the web application.")
    
    # Check if the server is running
    if wait_for_server():
        print("Server is running, but we'll copy the images directly anyway.")
    else:
        print("Server is not running, but we'll proceed to copy the images anyway.")
    
    # Copy images directly
    success = copy_model_images()
    if success:
        print("\nSuccess! Model comparison images have been copied.")
        print("Open the application in your browser and navigate to the 'Model Comparison' tab.")
    else:
        print("\nFailed to copy model comparison images.")
        print("Please ensure all image files exist in the model_outputs directory.")
    
    # Keep terminal window open on Windows
    if sys.platform.startswith('win'):
        input("\nPress Enter to exit...") 
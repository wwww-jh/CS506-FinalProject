# get_weather_data.py

import requests

API_KEY = "YOUR_API_KEY"
CITY_NAME = "Boston"
BASE_URL = "http://api.openweathermap.org/data/2.5/onecall/timemachine"

def get_weather_for_day(lat, lon, date_unix):
    params = {
        'lat': lat,
        'lon': lon,
        'dt': date_unix,
        'appid': API_KEY,
        'units': 'metric'
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch weather for timestamp {date_unix}")
        return None

# Example usage (replace with real loop later)
# weather_data = get_weather_for_day(42.3601, -71.0589, 1709251200)

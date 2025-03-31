import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configuration
start_date = datetime(2025, 2, 1)
end_date = datetime(2025, 3, 30)
popular_time_weights = {
    'Monday': 80,
    'Tuesday': 90,
    'Wednesday': 100,
    'Thursday': 110,
    'Friday': 150,
    'Saturday': 180,
    'Sunday': 160
}

# Generate dates
dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# Simulate weather and foot traffic
data = []
for date in dates:
    weekday = date.strftime('%A')
    base = popular_time_weights[weekday]
    temp = round(np.random.normal(loc=8, scale=6), 1)
    humidity = round(np.random.uniform(40, 85), 1)
    precipitation = np.random.choice([0, 0.5, 1.0, 2.0, 5.0])

    temp_factor = 1 + 0.015 * (temp - 8)
    rain_penalty = 1 - 0.1 * min(precipitation, 5)
    variation = random.uniform(0.9, 1.1)
    traffic = int(base * temp_factor * rain_penalty * variation)

    data.append({
        'date': date.date(),
        'temperature': temp,
        'humidity': humidity,
        'precipitation': precipitation,
        'foot_traffic': max(traffic, 0)
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('../data/simulated_foot_traffic_feb_to_mar2025.csv', index=False)
print("Simulated data saved.")

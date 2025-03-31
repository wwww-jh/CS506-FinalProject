# get_yelp_reviews.py

import requests

API_KEY = "YOUR_YELP_API_KEY"
HEADERS = {'Authorization': f'Bearer {API_KEY}'}
BUSINESS_ID = "victoria-seafood-boston"

def get_reviews(business_id):
    url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()["reviews"]
    else:
        print("Failed to fetch reviews")
        return []

# Example usage:
# reviews = get_reviews(BUSINESS_ID)

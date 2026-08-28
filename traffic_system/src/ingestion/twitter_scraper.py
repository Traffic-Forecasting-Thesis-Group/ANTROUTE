import os
import json
import requests
from dotenv import load_dotenv
from datetime import datetime
import pytz

def retrieve_twitter_data():
    load_dotenv()
    api_key = os.getenv("TWITTER_API_KEY")
    
    if not api_key:
        print("Error: TWITTER_API_KEY is not set in the .env file.")
        return

    url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    headers = {"X-API-Key": api_key}
    
    keywords = '("traffic jam" OR trapik OR aksidente OR "flash flood" OR "road closure")'
    locations = '(EDSA OR Makati OR "Quezon City" OR Pasay OR Manila)'
    
    # ---------------------------------------------------------
    # NEW: Exact Time Targeting using Unix Timestamps
    # ---------------------------------------------------------
    # Define your CCTV timezone
    manila_tz = pytz.timezone('Asia/Manila')
    
    # Set the exact start and end times to match your CCTV footage
    # Format: datetime(Year, Month, Day, Hour, Minute, Second)
    start_time = manila_tz.localize(datetime(2026, 5, 15, 8, 0, 0))  # May 15, 2026, 8:00 AM
    end_time = manila_tz.localize(datetime(2026, 5, 15, 12, 0, 0)) # May 15, 2026, 12:00 PM
    
    # Convert to Unix timestamps in seconds
    since_unix = int(start_time.timestamp())
    until_unix = int(end_time.timestamp())
    
    # Apply the since_time and until_time operators
    date_window = f'since_time:{since_unix} until_time:{until_unix}'
    # ---------------------------------------------------------

    query = f'{keywords} {locations} {date_window} -is:retweet'
    
    params = {
        "query": query,
        "queryType": "Latest"
    }
    
    all_tweets = []
    cursor = None
    max_pages = 10 
    pages_fetched = 0
    
    print(f"Starting data retrieval for query: {query}")
    
    while pages_fetched < max_pages:
        if cursor:
            params["cursor"] = cursor
            
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            break
            
        data = response.json()
        
        tweets = data.get("tweets") or data.get("data") or []
        
        if not tweets and isinstance(data, list):
             tweets = data
             
        all_tweets.extend(tweets)
        pages_fetched += 1
        
        cursor = data.get("next_cursor")
        if not cursor:
            break
            
    output_path = "data/raw/tweets.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"data": all_tweets}, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved {len(all_tweets)} tweets to {output_path}")

if __name__ == "__main__":
    retrieve_twitter_data()
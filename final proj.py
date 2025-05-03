# 1. Setup: imports and globals

import os
import time
import json
import requests
import sqlite3
from datetime import datetime
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt 

#MLB stats API base URL
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
#Local file to store verified players/team names
VALID_ENTRIES_FILE = "valid_mlb_entities.json"
#Trade deadline timestamp to know when local data may be outdated
TRADE_DEADLINE = datetime(2024, 7, 30)
#News API key (need to replace with actual API key)
NEWS_API_KEY = "YOUR_API_KEY" #Replace with actual API key
NEWS_API_URL = "https://gnews.io.api/v4/search" #GNews base URL

# 2. Entity verificaiton and iniialization

#Check if local file with valid player/team data exists and is up-to-date
def is_file_up_to_date(file_path, deadline): #deadline is date to compare against
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return False #invalid JSON
    last_updated_str = data.get('timestamp','1900-01-01')
    try:
        last_updated = datetime.strptime(last_updated_str, '%Y-%m-%d') #converts timestamp string to datetime
        return last_updated > deadline #compare timestamp with deadline
    except ValueError:
        return False #invalid date format

#If file is missing/outdated, fetch player and tea data from the API and save to local file
def fetch_and_cache_valid_entries():
    print("Fetching from MLB StatsAPI...")
    try:
        players_resp = requests.get(f"{MLB_STATS_API_BASE}/people") #get player data
        players_resp.raise_for_status() #raise exception for bad status codes
        teams_resp = requests.get(f"{MLB_STATS_API_BASE}/teams?sportId=1") #get team data
        teams_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from MLB Stats API: {e}")
        return #Exit if error

    try:
        player_data = players_resp.json().get('people',[])
        team_data = teams_resp.json().get('teams',[])
        valid_data = {
            "players": [{"id":p['id'],"name":p['fullName']} for p in player_data],
            "teams": [{"id":t['id'],"name":t['name']} for t in team_data],
            "timestamp": datetime.now().strftime('%Y-%m-%d')
        }

    except json.JSONDecodeError:
        print("Error decoding JSON data from MLB Stats API.")
        return
    
    try:
        with open(VALID_ENTRIES_FILE, 'w') as f: #open file for writing
            json.dump(valid_data, f, indent=2) #write JSON data to the file w/ indentation
    except OSError as e:
        print(f"Error writing to file {VALID_ENTRIES_FILE}: {e}")

#Load valid entries from local file if up-to-date, otherwise re-fetch from API 
#Returns a dictionary with valid player and team data
def load_or_update_valid_entries():
    if is_file_up_to_date(VALID_ENTRIES_FILE, TRADE_DEADLINE):
        try:
            with open(VALID_ENTRIES_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error reading or decoding {VALID_ENTRIES_FILE}: {e}. Re-fetching.")
            #if file is corrupt, re-fetch... 
            fetch_and_cache_valid_entries()
            with open(VALID_ENTRIES_FILE, 'r') as f:
                return json.load(f)
    else:
        fetch_and_cache_valid_entries() #fetch and cache data if file is outdated/missing
        try:
            with open(VALID_ENTRIES_FILE, 'r') as f:
                return json.load(f)
        except(json.JSONDecodeError, OSError) as e:
            print(f"Error reading {VALID_ENTRIES_FILE} after fetching: {e}")
            return {"players": [], "teams": []} #return empty dictionary 

# 3. News retrieval and sentiment analysis

#Fetch news from GNews API 
#Returns a list of news articles
def fetch_news(query,from_date,to_date):
    url = f"{NEWS_API_URL}?q={query}&from={from_date}&to={to_date}&token={NEWS_API_KEY}" #construct the API URL
    try:
        response = requests.get(url) #make API request
        response.raise_for_status()
        time.sleep(1) #sleep to respect rate limits
        data = response.json()
        return data.get("articles", []) #return the articles or an empty list 
    except json.JSONDecodeError:
        print("Error decoding JSON from News API")
        return []

#Use NLTK sentiment analyzer to attach a compound sentiment score to each article
def analyze_sentiment(articles):
    sia = SentimentIntensityAnalyzer() #initialize sentiment analyzer
    for article in articles:
        content = article.get('description','') or '' #get article description, default to empty string
        sentiment = sia.polarity_scores(content) #get sentiment scores
        article['sentiment_score'] = sentiment['compound'] #add compound score to article
    return articles

#Store news articles and their sentiment scores in SQLite DB for long-term analysis 
def store_articles(articles,db_path="news_articles.db"):
    try:
        conn = sqlite3.connect(db_path) #connect to database
        c = conn.cursor() #get cursor object
        c.execute('''CREATE TABLE IF NOT EXISTS articles (
                title TEXT,
                date TEXT,
                author TEXT,
                url TEXT,
                source TEXT,
                sentiment REAL
                )''') #create table 
        for article in articles:
            c.execute("INSERT INTO articles VALUES (?,?,?,?,?,?)", ( #insert each article's data
                article.get('title'),
                article.get('publishedAt'),
                article.get('author'),
                article.get('url'),
                article.get('source', {}).get('name'),
                article.get('sentiment_score')
            ))
        conn.commit() #commit changes
        conn.close() #close connection
    except sqlite3.Error as e:
        print(f"Error interacting with the database: {e}")

# 4. Statistical lookup (example)

#Retrieve season stats for a given player using their ID
def fetch_player_stats(player_id):
    url = f"{MLB_STATS_API_BASE}/people/{player_id}/stats?stats=season" #construct API URL
    try:
        response = requests.get(url) #make API request
        response.raise_for_status()
        return response.json() #return JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player stats: {e}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding player stats JSON")
        return {}

#example usage, replace 'Your Team' and date range
if __name__ == "__main__":
    valid_entities = load_or_update_valid_entries()
    print(f"Loaded {len(valid_entities.get('players', []))} players and {len(valid_entities.get('teams',[]))} teams.")

    query = "New York Yankees"
    from_date_str = "2025-05-01"
    to_date_str = "2025-05-02"
    news_articles = fetch_news(query, from_date_str, to_date_str)
    if news_articles:
        print(f"Fetched {len(news_articles)} news articles about '{query}.")
        analyzed_articles = analyze_sentiment(news_articles)
        store_articles(analyzed_articles)
        print("News articles and sentiment scores stored in the database")
    else:
        print(f"No news articles found for '{query}' between {from_date_str} and {to_date_str}")

# 6. Putting it Together
#main flow of program: 
# 1. take user input
# 2. validate it against the updated entity list
# 3. pull stats and news
# 4. analyze sentiment
# 5. display everything in the user interface
def main():
    valid_entries = load_or_update_valid_entries() #load valid player and team data
    user_input = input("Enter player, team, or league: ").strip() #get user input

    entity_id = None
    entity_type = None
    entity_name = None 

    #try to match player names 
    if not entity_id:
        for team in valid_entries.get('teams', []):
            if user_input.lower() in team['name'].lower():
                entity_id = team['id']
                entity_type = 'team'
                entity_name = team['name']
                break

    #try to match team names
    if not entity_id:
        for team in valid_entries.get('teams', []):
            if user_input.lower() in team['name'].lower():
                entity_id = team['id']
                entity_type = 'team'
                entity_name = team['name']
                break

    # league search
    if  not entity_id and user_input.lower() == "mlb":
        entity_type = 'league'
        entity_name = 'MLB'
        entity_id = None #set to None so stats aren't fetched

    if not entity_id and entity_type != 'league':
        print("Invalid input. Please enter a valid player or team name.")
        return #exit if the input is invalid
    
    #data retrieva
    stats = {}
    if entity_type == 'player':
        stats = fetch_player_stats('entity_id') #fetch player stats
    elif entity_type == 'team':
        #stats = fetch_team_stats(entity_id) #need to implement fetch_team_stats
        stats = {} # PLACEHOLDER !!
    elif entity_type == 'league':
        stats = {} # PLACEHOLDer !! 

    news_query = entity_name if entity_type != "league" else "MLB" #news query
    #fetch news for a more relevant time period
    now = datetime.now()
    one_month_ago = now.replace(day=1)
    from_date_str = one_month_ago.strftime('%Y-%m-%d')
    to_date_str = now.strftime('%Y-%m-%d')
    articles = fetch_news(news_query, "2024-09-01", "2024-09-30") #fetch news articles
    scored_articles = analyze_sentiment(articles)
    store_articles(scored_articles) #store articles in the database

    display_results(entity_name, stats, scored_articles) #display results

def display_results(entity_name, stats, articles):
    print(f"\n--- Results for {entity_name}---")

    if stats:
        print("\n---Stats---")
        print(json.dumps(stats,indent=2))
    else:
        print("\n---Stats---")
        print("No stats available.")

    if articles:
        print("\n --- Recnt news and sentiment---")
        for article in articles[:5]:
            print(f"Title: {article.get('title')}")
            print(f"Source {article.get('source',{}).get('name')}")
            print(f"Sentiment Score: {article.get('sentiment_score'):.2f}")
            print(f"URL: {article.get('url')}")
            print("-"*20)
        else:
            print("\n---Recent news---")
            print("No recent news articles found.")
                  

if __name__ == "__main__":
    main()

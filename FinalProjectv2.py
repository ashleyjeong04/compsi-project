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

# MLB stats API base URL
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
# Local file to store verified players/team names
VALID_ENTRIES_FILE = "valid_mlb_entities.json"
# Trade deadline timestamp to know when local data may be outdated
TRADE_DEADLINE = datetime(2024, 7, 30)
# News API key (replace with your actual API key)
NEWS_API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
NEWS_API_URL = "https://gnews.io/api/v4/search"  # GNews base URL

# 2. Entity verification and initialization

def is_file_up_to_date(file_path, deadline):
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return False
    last_updated_str = data.get('timestamp', '1900-01-01')
    try:
        last_updated = datetime.strptime(last_updated_str, '%Y-%m-%d')
        return last_updated > deadline
    except ValueError:
        return False

def fetch_and_cache_valid_entries():
    print("Fetching from MLB StatsAPI...")
    try:
        players_resp = requests.get(f"{MLB_STATS_API_BASE}/people")
        players_resp.raise_for_status()
        teams_resp = requests.get(f"{MLB_STATS_API_BASE}/teams?sportId=1")
        teams_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from MLB Stats API: {e}")
        return

    try:
        player_data = players_resp.json().get('people', [])
        team_data = teams_resp.json().get('teams', [])
        valid_data = {
            "players": [{"id": p['id'], "name": p['fullName']} for p in player_data],
            "teams": [{"id": t['id'], "name": t['name']} for t in team_data],
            "timestamp": datetime.now().strftime('%Y-%m-%d')
        }
    except json.JSONDecodeError:
        print("Error decoding JSON data from MLB Stats API.")
        return

    try:
        with open(VALID_ENTRIES_FILE, 'w') as f:
            json.dump(valid_data, f, indent=2)
    except OSError as e:
        print(f"Error writing to file {VALID_ENTRIES_FILE}: {e}")

def load_or_update_valid_entries():
    if is_file_up_to_date(VALID_ENTRIES_FILE, TRADE_DEADLINE):
        try:
            with open(VALID_ENTRIES_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error reading or decoding {VALID_ENTRIES_FILE}: {e}. Re-fetching.")
            fetch_and_cache_valid_entries()
            with open(VALID_ENTRIES_FILE, 'r') as f:
                return json.load(f)
    else:
        fetch_and_cache_valid_entries()
        try:
            with open(VALID_ENTRIES_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error reading {VALID_ENTRIES_FILE} after fetching: {e}")
            return {"players": [], "teams": []}

# 3. News retrieval and sentiment analysis

def fetch_news(query, from_date, to_date):
    url = f"{NEWS_API_URL}?q={query}&from={from_date}&to={to_date}&token={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        time.sleep(1)  # respect rate limits
        data = response.json()
        return data.get("articles", [])
    except json.JSONDecodeError:
        print("Error decoding JSON from News API")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

def analyze_sentiment(articles):
    sia = SentimentIntensityAnalyzer()
    for article in articles:
        content = article.get('description', '') or ''
        sentiment = sia.polarity_scores(content)
        article['sentiment_score'] = sentiment['compound']
    return articles

def store_articles(articles, db_path="news_articles.db"):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS articles (
            title TEXT,
            date TEXT,
            author TEXT,
            url TEXT,
            source TEXT,
            sentiment REAL
        )''')
        for article in articles:
            c.execute("INSERT INTO articles VALUES (?,?,?,?,?,?)", (
                article.get('title'),
                article.get('publishedAt'),
                article.get('author'),
                article.get('url'),
                article.get('source', {}).get('name'),
                article.get('sentiment_score')
            ))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Database error: {e}")

# 4. Statistical lookup (example)

def fetch_player_stats(player_id):
    url = f"{MLB_STATS_API_BASE}/people/{player_id}/stats?stats=season"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player stats: {e}")
        return {}
    except json.JSONDecodeError:
        print("Error decoding player stats JSON")
        return {}

# Example usage, replace with actual flow
if __name__ == "__main__":
    valid_entities = load_or_update_valid_entries()
    print(f"Loaded {len(valid_entities.get('players', []))} players and {len(valid_entities.get('teams', []))} teams.")

    # Example search
    query = "New York Yankees"
    from_date_str = "2025-05-01"
    to_date_str = "2025-05-02"
    news_articles = fetch_news(query, from_date_str, to_date_str)
    if news_articles:
        print(f"Fetched {len(news_articles)} news articles about '{query}'.")
        analyzed_articles = analyze_sentiment(news_articles)
        store_articles(analyzed_articles)
        print("News articles and sentiment scores stored in the database.")
    else:
        print(f"No news articles found for '{query}' between {from_date_str} and {to_date_str}.")

# 6. Putting it together

def main():
    valid_entries = load_or_update_valid_entries()
    user_input = input("Enter player, team, or league: ").strip()

    entity_id = None
    entity_type = None
    entity_name = None

    # Match team name (case insensitive)
    for team in valid_entries.get('teams', []):
        if user_input.lower() in team['name'].lower():
            entity_id = team['id']
            entity_type = 'team'
            entity_name = team['name']
            break

    # Match player name if needed (assuming you add player matching logic)
    for player in valid_entries.get('players', []):
        if user_input.lower() in player['name'].lower():
            entity_id = player['id']
            entity_type = 'player'
            entity_name = player['name']
            break

    # Check for league
    if not entity_id and user_input.lower() == "mlb":
        entity_type = 'league'
        entity_name = 'MLB'
        entity_id = None

    if not entity_id and entity_type != 'league':
        print("Invalid input. Please enter a valid player or team name.")
        return

    # Fetch stats based on entity type
    stats = {}
    if entity_type == 'player':
        stats = fetch_player_stats(entity_id)
    elif entity_type == 'team':
        # You can implement fetch_team_stats similar to fetch_player_stats
        stats = {}  # Placeholder
    elif entity_type == 'league':
        stats = {}  # Placeholder

    # Fetch news
    now = datetime.now()
    from_date_str = (now.replace(day=1)).strftime('%Y-%m-%d')
    to_date_str = now.strftime('%Y-%m-%d')
    articles = fetch_news(entity_name if entity_type != 'league' else 'MLB', from_date_str, to_date_str)
    scored_articles = analyze_sentiment(articles)
    store_articles(scored_articles)

    # Display results
    display_results(entity_name, stats, scored_articles)

def display_results(entity_name, stats, articles):
    print(f"\n--- Results for {entity_name} ---")
    if stats:
        print("\n--- Stats ---")
        # Here, you can add code to display stats nicely
    if articles:
        print(f"\nFetched {len(articles)} articles with sentiment scores.")
        # Optionally, display sentiment summary or plot
        sentiments = [a['sentiment_score'] for a in articles]
        plt.hist(sentiments, bins=20)
        plt.title("Sentiment Score Distribution")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Number of Articles")
        plt.show
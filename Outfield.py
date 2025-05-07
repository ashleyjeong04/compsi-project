'''
SECTION 1
Setup: Imports and Globals 
'''

import os
from dotenv import load_dotenv # To load variables from a .env file
import time # For time-related functions
import json # To read from and write to JSON files
import requests # To make HTTP requests to APIs
import sqlite3 # To store articles in a local database
from datetime import datetime # Imports a class to handle date and time operations
import nltk # Natural Language Toolkit for sentiment analysis
import textwrap # To format and wrap long text for disdplay
nltk.download('vader_lexicon') # Download lexicon used for sentiment scoring with NTLK
from nltk.sentiment import SentimentIntensityAnalyzer # Imports VADER to score sentiment
import matplotlib.pyplot as plt # Imports plotting library for data vis

load_dotenv()

MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"  # MLB stats API base URL
VALID_ENTRIES_FILE = "valid_mlb_entities.json"          # Local file to store verified players/team names
TRADE_DEADLINE = datetime(2024, 7, 30)                  # Trade deadline timestamp to know when local data may be outdated
NEWS_API_URL = "https://gnews.io/api/v4/search"         # GNews base URL
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not NEWS_API_KEY:
    raise RuntimeError("Missing required environment variable: NEWS_API_KEY")

'''
SECTION 2
Entity Verification and Initialization
'''

# Check if local file with valid player/team data exists and is up-to-date
def is_file_up_to_date(file_path, deadline): # Deadline is date to compare against
    if not os.path.exists(file_path): 
        return False # If file doesn't exist
    try:
        with open(file_path, 'r') as f: # Attempts to open and read file as JSON
            data = json.load(f)
    except json.JSONDecodeError: 
        return False # If file content isn't valid JSON
    last_updated_str = data.get('timestamp', '1900-01-01')
    try:
        last_updated = datetime.strptime(last_updated_str, '%Y-%m-%d') # Converts timestamp string to datetime
        return last_updated > deadline # Compare timestamp with deadline 
    except ValueError:
        return False # Invalid date format 

# If file is missing/outdated, fetch player and team data from the API and save to local file
def fetch_and_cache_valid_entries():
    """
    Fetches all MLB teams, then for each team fetches its active roster,
    and caches the combined list of teams and players into a JSON file.
    """
    print("Fetching teams from MLB StatsAPI…")
    try:
        # 1. Get the list of all MLB teams
        teams_resp = requests.get(f"{MLB_STATS_API_BASE}/teams?sportId=1")  # Send HTTP GET request to fetch all MLB teams
        teams_resp.raise_for_status() # Check for HTTP errors
        teams_list = teams_resp.json().get("teams", []) # Extract list of teams 
    except requests.exceptions.RequestException as e:
        print(f"Error fetching teams: {e}")
        teams_list = [] # Print empty list if error

    current_season = datetime.now().year
    valid_data = { # Initializes data to store valid entries and set a current timestamp.
        "teams": [],
        "players": [],
        "timestamp": datetime.now().strftime("%Y-%m-%d")
    }

    # 2. For each team, record its info and fetch its active roster
    for team in teams_list: # Loops through teams 
        team_id = team.get("id")
        team_name = team.get("name")
        if not team_id or not team_name:
            continue # Skips teams with missing ID or name

        valid_data["teams"].append({ # Appends the team to the teams list 
            "id": team_id,
            "name": team_name
        })

        print(f"  • Fetching roster for {team_name} (ID {team_id})…")  # Prints the team being processed
        try:
            roster_resp = requests.get( # Requests the team's active roster
                f"{MLB_STATS_API_BASE}/teams/{team_id}/roster",
                params={
                    "season": current_season,
                    "rosterType": "Active"
                }
            )
            roster_resp.raise_for_status()
            roster_entries = roster_resp.json().get("roster", [])
        except requests.exceptions.RequestException as e: # Handles request errors
            print(f"    Error fetching roster for {team_name}: {e}")
            roster_entries = []

        # 3. Collect each player's ID and full name
        for entry in roster_entries:
            person = entry.get("person") or {}
            pid = person.get("id")
            pname = person.get("fullName")
            if pid and pname:
                valid_data["players"].append({
                    "id": pid,
                    "name": pname
                })

    # 4. Cache to local file for later use
    try:
        with open(VALID_ENTRIES_FILE, "w") as f:
            json.dump(valid_data, f, indent=2)
        print(f"Wrote {len(valid_data['teams'])} teams and {len(valid_data['players'])} players to {VALID_ENTRIES_FILE}")
    except OSError as e:
        print(f"Error writing {VALID_ENTRIES_FILE}: {e}")

def load_or_update_valid_entries(): # Load from file or refetch if needed
    if is_file_up_to_date(VALID_ENTRIES_FILE, TRADE_DEADLINE): # Checks if the file is up to date
        try: 
            with open(VALID_ENTRIES_FILE, 'r') as f:
                data = json.load(f) # Returns the loaded file if valid
                print(f"\nCurrent roster JSON found! Identified {len(data['players'])} players on {len(data['teams'])} teams.")
                return data
        except (json.JSONDecodeError, OSError) as e: # If there's an error, fetch again and then load
            print(f"Error reading or decoding {VALID_ENTRIES_FILE}: {e}. Re-fetching.")
            fetch_and_cache_valid_entries()
            with open(VALID_ENTRIES_FILE, 'r') as f:
                return json.load(f)
    else: # If outdated, fetch and retry. If errors persist, return an empty structure
        fetch_and_cache_valid_entries()
        try:
            with open(VALID_ENTRIES_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error reading {VALID_ENTRIES_FILE} after fetching: {e}")
            return {"players": [], "teams": []}

'''
SECTION 3
News Retrieval and Sentiment Analysis
'''

def fetch_news(query, from_date, to_date): # To get news articles related to a query in a specific time period
    total_articles = 10 # Limits the number of articles to 10
    url = ( # Constructs the URL for the API request, inserting the query (search term), date range, and API key
        f"{NEWS_API_URL}"
        f"?q={query}"
        f"&from={from_date}"
        f"&to={to_date}"
        f"&max={total_articles}"  
        f"&token={NEWS_API_KEY}"
    )
    try:
        response = requests.get(url) # Sends request and raises an error if the status is not ok
        response.raise_for_status()
        time.sleep(1)  # Pauses for 1 second to respect GNews rate limits
        data = response.json() # Parses JSON response and returns list of articles
        return data.get("articles", []) # If list not found, returns empty list
    except json.JSONDecodeError: # If JSON parsing errors, returns empty list
        print("Error decoding JSON from News API") 
        return []
    except requests.exceptions.RequestException as e: # If network-related errors, returns empty list
        print(f"Error fetching news: {e}")
        return []

def analyze_sentiment(articles): # VADER for scoring sentiment of each article  
    sia = SentimentIntensityAnalyzer() 
    for article in articles: # Loop through each article in the list 
        content = article.get('description', '') or '' # Gets the article description/text to analyze
        sentiment = sia.polarity_scores(content) # Calculates sentiment scores 
        article['sentiment_score'] = sentiment['compound'] # Compound score (sentiment from -1 to 1)
    return articles # Return modified articles with sentiment scores

# Converts numeric sentiment score into labels 
def categorize_score(score): 
    if score <= -0.667: # Defines ranges for each label, from strongly negative to strongly positive
        return "Strongly negative"
    elif score <= -0.334:
        return "Moderately negative"
    elif score < 0.0:
        return "Slightly negative"
    elif score == 0.0:
        return "Neutral"
    elif score <= 0.333:
        return "Slightly positive"
    elif score <= 0.666:
        return "Moderately positive"
    else:
        return "Strongly positive"

# Stores articles to a local SQLite database file
def store_articles(articles, db_path="news_articles.db"): 
    try:
        conn = sqlite3.connect(db_path) # Connects to the SQLite database 
        c = conn.cursor() # Creates cursor object to execute SQL statements
        c.execute('''CREATE TABLE IF NOT EXISTS articles ( 
            title TEXT,
            date TEXT,
            author TEXT,
            url TEXT,
            source TEXT,
            sentiment REAL
        )''') # Creates table named "articles"
        for article in articles: # Loops through each article 
            c.execute("INSERT INTO articles VALUES (?,?,?,?,?,?)", (
                article.get('title'),
                article.get('publishedAt'),
                article.get('author'),
                article.get('url'),
                article.get('source', {}).get('name'),
                article.get('sentiment_score')
            )) # After inserting each article into the SQLite database, extracts relevant fields to add to table
        conn.commit()
        conn.close() # Saves changes and closes SQLite database connection
    except sqlite3.Error as e:
        print(f"Database error: {e}") # Handles and prints database errors 

'''
SECTION 4
Statistical Lookup
'''

# Fetch player stats from the MLB API 
def fetch_player_stats(player_id):
    url = f"{MLB_STATS_API_BASE}/people/{player_id}/stats?stats=season" # Construct URL for player stats using player_id
    try:
        response = requests.get(url) # GET request to the API 
        response.raise_for_status() 
        return response.json() # Return JSON response
    except requests.exceptions.RequestException as e:  # Handle network or HTTP errors 
        print(f"Error fetching player stats: {e}") # Handle JSON parsing errors
        return {}
    except json.JSONDecodeError:
        print("Error decoding player stats JSON")
        return {}

# Get a team's seasonal performance stats 
def fetch_team_stats(team_id):
    url = f"{MLB_STATS_API_BASE}/teams/{team_id}/stats" # Construct URL for team stats 
    params = {
        "stats":     "statsSingleSeason",           # the stat type
        "season":    str(datetime.now().year),      # must be a string
        "group":     "team",                        
        "sportIds":  "1",                           # MLB only
        "gameType":  "R"                            # Regular‐season only
    }
    try: # Make request with parameters
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json() # Return parsed JSON
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team stats: {e}")
        return {}
    except json.JSONDecodeError:
        print("Error decoding team stats JSON")
        return {}

'''
SECTION 5
Output Generation
'''

def display_player_stats(entity_name, stats_json): # Show formatted player stats based on their position
    
    stats_list = stats_json.get('stats', [])  # Extract the "stats" list from the response
    if not stats_list:
        print("No stats available.")
        return

    # The first stats entry contains the seasonal data
    splits = stats_list[0].get('splits', [])
    if not splits:
        print("No seasonal splits found.")
        return
        
    team_name = splits[0].get('team', {}).get('name', 'Unknown Team') # Get the team name or default to Unknown Team

    # Print player header
    print("\n" + "="*80)
    print(f" Stats for {entity_name} ({team_name}):")
    print("="*80)

     # Get the stat dictionary
    stat = splits[0].get('stat', {})

    # Determine if hitting or pitching based on keys present
    if 'avg' in stat:
        # Hitting stats
        avg = stat.get('avg', 'N/A')
        hr = stat.get('homeRuns', 'N/A')
        rbi = stat.get('rbi', 'N/A')
        hits = stat.get('hits', 'N/A')
        ops = stat.get('ops', 'N/A')
        print(f"Batting Average: {avg}")
        print(f"Hits: {hits}")
        print(f"Home Runs: {hr}")
        print(f"RBIs: {rbi}")
        print(f"OPS: {ops}")
    elif 'era' in stat:
        # Pitching stats
        era = stat.get('era', 'N/A')
        wins = stat.get('wins', 'N/A')
        losses = stat.get('losses', 'N/A')
        so = stat.get('strikeOuts', 'N/A')
        whip = stat.get('whip', 'N/A')
        print(f"ERA: {era}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Strikeouts: {so}")
        print(f"WHIP: {whip}")
    else:
        # Other stats
        for key, value in stat.items():
            print(f"{key}: {value}")
        
def display_team_stats(team_name, stats_json):
    print(f"\n--- Stats for {team_name} (Team) ---")  # Print header for team stats
    stats_list = stats_json.get('stats', []) # Get 'stats' list from the API response 
    if not stats_list:
        print("No stats available.")
        return

    # The season‐level stats are in the first stats entry's splits
    splits = stats_list[0].get('splits', [])
    if not splits:
        print("No seasonal splits found.")
        return

    stat = splits[0].get('stat', {}) # Get the actual stats dictionary

    # Common team stats you might want to show:
    wins          = stat.get('wins', 'N/A') # Extract specific team stats for display
    losses        = stat.get('losses', 'N/A')
    win_pct       = stat.get('winPct', 'N/A')
    runs_scored   = stat.get('runsScored', 'N/A')
    runs_against  = stat.get('runsAgainst', 'N/A')
    home_wins     = stat.get('homeWins', 'N/A')
    away_wins     = stat.get('awayWins', 'N/A')
    streak        = stat.get('currentStreak', {}).get('streakCode', 'N/A')

    # Print team performance metrics
    print(f"Record        : {wins}-{losses} ({win_pct})")
    print(f"Runs Scored   : {runs_scored}")
    print(f"Runs Against  : {runs_against}")
    print(f"Home Wins     : {home_wins}")
    print(f"Away Wins     : {away_wins}")
    print(f"Streak        : {streak}")

# Display stats based on entity type 
def display_results(entity_name, entity_type, stats, articles): 
    if entity_type == "player":
        display_player_stats(entity_name, stats) # Show player stats
    elif entity_type == "team":
        print("\nStatistical data for teams coming soon!") # TO DO: display_team_stats(entity_name, stats) /// API is much more complicated for team stats than player stats, needs more fine-tuning before it's ready
    else:
        print("\nNo statistical data available.")

    if articles:
        dates = [a.get('publishedAt', '')[:10] for a in articles if a.get('publishedAt')]
        if dates:
            start_date = min(dates)
            end_date = max(dates)
        else:
            start_date = end_date = 'N/A'

        # Calculate average sentiment score across all articles 
        avg_sent = (sum(a.get('sentiment_score', 0.0) for a in articles)/ len(articles)) if articles else 0.0
        
        print("\n" + "="*80)
        print(f" News Articles for {entity_name} ({len(articles)} found):")
        print(f"   Date Range             : {start_date} to {end_date}")
        print(f"   Average Sentiment Score: {avg_sent:+.3f} ({categorize_score(avg_sent)})")
        print("="*80)

        # Iterate through and display each article 
        for idx, article in enumerate(articles, start=1):
            date  = article.get('publishedAt', '')[:10] or 'N/A'
            title = article.get('title', 'No Title')
            score = article.get('sentiment_score', 0.0)

            # Build snippet from article description or content 
            content = article.get('description') or article.get('content') or ''
            snippet = " ".join(content.split()[:30]) + "..."
            url   = article.get('url', '')

            # Print article metadata
            print(f"\n{idx}. {title}")
            print(f"   Date: {date}    Sentiment Score: {score:+.3f} ({categorize_score(score)})")
            print("   Snippet:")
            for line in textwrap.wrap(snippet, width=76): # Wrap long lines 
                print(f"     {line}")
            if url:
                print(f"   Read more: {url}") # Include article link 

            print("-"*80)
            
        # Plot histogram of sentiment Distribution
        sentiments = [a['sentiment_score'] for a in articles]
        plt.figure()
        plt.hist(sentiments, bins=20)
        plt.title("Sentiment Score Distribution")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Number of Articles")
        plt.show()

        #  Plot sentiment over time
        try:
            dates = [datetime.strptime(a.get('publishedAt', '')[:10], '%Y-%m-%d') for a in articles]
            plt.figure()
            plt.plot(dates, sentiments, marker='o')
            plt.title("Sentiment Over Time")
            plt.xlabel("Date")
            plt.ylabel("Sentiment Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot sentiment over time: {e}")

'''
SECTION 6
Execution
'''

def main():
    valid = load_or_update_valid_entries() # Calls function to load from cache or retrieve fresh list of valid teams/players 

    # Creates two dictionaries that map lowercase taem/player names to their full data dictionaries
    team_map   = { t['name'].lower(): t for t in valid['teams'] }
    player_map = { p['name'].lower(): p for p in valid['players'] }

    while True:
        user_input = input("\nEnter full player or team name (type EXIT to quit): ").strip().lower() # Loop prompting user to input team/player name or type EXIT

        if user_input == "exit":
            print("Goodbye!")
            return # If user types EXIT, prints goodbye message ane exits

        if user_input in team_map: # Check if user input matches a known team or player 
            entity_type, entity_id, entity_name = 'team',   team_map[user_input]['id'],   team_map[user_input]['name']
            break
        elif user_input in player_map:
            entity_type, entity_id, entity_name = 'player', player_map[user_input]['id'], player_map[user_input]['name']
            break
        elif user_input == 'mlb':
            entity_type, entity_id, entity_name = 'league', None, 'MLB'
            break
        else:
            print("Invalid input. Please enter a valid player or team name.")

    stats = {} # Initialize empty dictionary to store performance data
    if entity_type == 'player': # If user inputed player, fetches stats using player ID 
        stats = fetch_player_stats(entity_id)
    elif entity_type == 'team':
        stats == {} # TO DO: display_team_stats(entity_name, stats) /// API is much more complicated for team stats than player stats, needs more fine-tuning before it's ready
    elif entity_type == 'league':
        stats = {}

    now = datetime.now()
    from_date_str = (now.replace(day=1)).strftime('%Y-%m-%d')
    to_date_str = now.strftime('%Y-%m-%d')

    # Fetch news articles using selected name within date range
    articles = fetch_news(entity_name if entity_type != 'league' else 'MLB', from_date_str, to_date_str)
    scored_articles = analyze_sentiment(articles) # Store scored articles in local database
    store_articles(scored_articles)

    display_results(entity_name, entity_type, stats, scored_articles)
            
if __name__ == "__main__":
    main()

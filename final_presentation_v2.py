# 1. Setup: imports and globals

import os
import time
import json
import requests
import sqlite3
from datetime import datetime
import nltk
import textwrap
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"  # MLB stats API base URL
VALID_ENTRIES_FILE = "valid_mlb_entities.json"          # Local file to store verified players/team names
TRADE_DEADLINE = datetime(2024, 7, 30)                  # Trade deadline timestamp to know when local data may be outdated
NEWS_API_KEY = "5fce54ce82e2cc4b47d46a5022583dd8"       # Replace with ENV if time permits
NEWS_API_URL = "https://gnews.io/api/v4/search"         # GNews base URL


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
    """
    Fetches all MLB teams, then for each team fetches its active roster,
    and caches the combined list of teams and players into a JSON file.
    """
    print("Fetching teams from MLB StatsAPI…")
    try:
        # 1. Get the list of all MLB teams
        teams_resp = requests.get(f"{MLB_STATS_API_BASE}/teams?sportId=1")
        teams_resp.raise_for_status()
        teams_list = teams_resp.json().get("teams", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching teams: {e}")
        teams_list = []

    current_season = datetime.now().year
    valid_data = {
        "teams": [],
        "players": [],
        "timestamp": datetime.now().strftime("%Y-%m-%d")
    }

    # 2. For each team, record its info and fetch its active roster
    for team in teams_list:
        team_id = team.get("id")
        team_name = team.get("name")
        if not team_id or not team_name:
            continue

        valid_data["teams"].append({
            "id": team_id,
            "name": team_name
        })

        print(f"  • Fetching roster for {team_name} (ID {team_id})…")
        try:
            roster_resp = requests.get(
                f"{MLB_STATS_API_BASE}/teams/{team_id}/roster",
                params={
                    "season": current_season,
                    "rosterType": "Active"
                }
            )
            roster_resp.raise_for_status()
            roster_entries = roster_resp.json().get("roster", [])
        except requests.exceptions.RequestException as e:
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
    total_articles = 10
    url = (
        f"{NEWS_API_URL}"
        f"?q={query}"
        f"&from={from_date}"
        f"&to={to_date}"
        f"&max={total_articles}"  
        f"&token={NEWS_API_KEY}"
    )
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

def categorize_score(score):
    if score <= -0.667:
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
        
def fetch_team_stats(team_id):
    url = f"{MLB_STATS_API_BASE}/teams/{team_id}/stats"
    params = {
        "stats":     "statsSingleSeason",          # the statType
        "season":    str(datetime.now().year),  # must be a string
        "group":     "team",         # statGroup is required
        "sportIds":  "1",               # MLB only
        "gameType":  "R"                # Regular‐season only
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team stats: {e}")
        return {}
    except json.JSONDecodeError:
        print("Error decoding team stats JSON")
        return {}
        
def display_team_stats(team_name, stats_json):
    print(f"\n--- Stats for {team_name} (Team) ---")
    stats_list = stats_json.get('stats', [])
    if not stats_list:
        print("No stats available.")
        return

    # The season‐level stats are in the first stats entry's splits
    splits = stats_list[0].get('splits', [])
    if not splits:
        print("No seasonal splits found.")
        return

    stat = splits[0].get('stat', {})

    # Common team stats you might want to show:
    wins          = stat.get('wins', 'N/A')
    losses        = stat.get('losses', 'N/A')
    win_pct       = stat.get('winPct', 'N/A')
    runs_scored   = stat.get('runsScored', 'N/A')
    runs_against  = stat.get('runsAgainst', 'N/A')
    home_wins     = stat.get('homeWins', 'N/A')
    away_wins     = stat.get('awayWins', 'N/A')
    streak        = stat.get('currentStreak', {}).get('streakCode', 'N/A')

    print(f"Record        : {wins}-{losses} ({win_pct})")
    print(f"Runs Scored   : {runs_scored}")
    print(f"Runs Against  : {runs_against}")
    print(f"Home Wins     : {home_wins}")
    print(f"Away Wins     : {away_wins}")
    print(f"Streak        : {streak}")

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

def display_player_stats(entity_name, stats_json):
    
    stats_list = stats_json.get('stats', [])
    if not stats_list:
        print("No stats available.")
        return

    # The first stats entry contains the seasonal data
    splits = stats_list[0].get('splits', [])
    if not splits:
        print("No seasonal splits found.")
        return
        
    team_name = splits[0].get('team', {}).get('name', 'Unknown Team')
    
    print("\n" + "="*80)
    print(f" Stats for {entity_name} ({team_name}):")
    print("="*80)

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
            
def display_results(entity_name, entity_type, stats, articles):
    if entity_type == "team":
        print("\nStatistical data for teams coming soon!") # TO DO: display_team_stats(entity_name, stats) /// API is much more complicated for team stats than player stats, needs more fine-tuning before it's ready
    elif entity_type == "player":
        display_player_stats(entity_name, stats)
    else:
        print("\nNo statistical data available.")

    # Display articles with sentiment scores, snippets, and links
    if articles:
        dates = [a.get('publishedAt', '')[:10] for a in articles if a.get('publishedAt')]
        if dates:
            start_date = min(dates)
            end_date = max(dates)
        else:
            start_date = end_date = 'N/A'
            
        avg_sent = (sum(a.get('sentiment_score', 0.0) for a in articles)/ len(articles)) if articles else 0.0
        
        
        print("\n" + "="*80)
        print(f" News Articles for {entity_name} ({len(articles)} found):")
        print(f"   Date Range             : {start_date} to {end_date}")
        print(f"   Average Sentiment Score: {avg_sent:+.3f} ({categorize_score(avg_sent)})")
        print("="*80)

        for idx, article in enumerate(articles, start=1):
            date  = article.get('publishedAt', '')[:10] or 'N/A'
            title = article.get('title', 'No Title')
            score = article.get('sentiment_score', 0.0)
            content = article.get('description') or article.get('content') or ''
            snippet = " ".join(content.split()[:30]) + "..."
            url   = article.get('url', '')

            # Header
            print(f"\n{idx}. {title}")
            print(f"   Date: {date}    Sentiment Score: {score:+.3f} ({categorize_score(score)})")

            # Wrapped snippet
            print("   Snippet:")
            for line in textwrap.wrap(snippet, width=76):
                print(f"     {line}")

            # Link
            if url:
                print(f"   Read more: {url}")

            print("-"*80)
            
        # Plot histogram of sentiment scores   
        sentiments = [a['sentiment_score'] for a in articles]
        plt.figure()
        plt.hist(sentiments, bins=20)
        plt.title("Sentiment Score Distribution")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Number of Articles")
        plt.show()

        # Plot sentiment over time
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

# 6. Putting it together

def main():
    valid = load_or_update_valid_entries()
    # build fast, exact-match maps
    team_map   = { t['name'].lower(): t for t in valid['teams'] }
    player_map = { p['name'].lower(): p for p in valid['players'] }

    # keep asking until we get either a valid key or the user types EXIT
    while True:
        user_input = input("\nEnter full player or team name (type EXIT to quit): ").strip().lower()

        if user_input == "exit":
            print("Goodbye!")
            return

        if user_input in team_map:
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
            # loop back to prompt

    # at this point, entity_type/name/id are set
    stats = {}
    if entity_type == 'player':
        stats = fetch_player_stats(entity_id)
    elif entity_type == 'team':
        stats == {} # TO DO: display_team_stats(entity_name, stats) /// API is much more complicated for team stats than player stats, needs more fine-tuning before it's ready
    elif entity_type == 'league':
        stats = {}

    # Fetch news
    now = datetime.now()
    from_date_str = (now.replace(day=1)).strftime('%Y-%m-%d')
    to_date_str = now.strftime('%Y-%m-%d')
    articles = fetch_news(entity_name if entity_type != 'league' else 'MLB', from_date_str, to_date_str)
    scored_articles = analyze_sentiment(articles)
    store_articles(scored_articles)

    # Display results
    display_results(entity_name, entity_type, stats, scored_articles)
            
if __name__ == "__main__":
    main()
"""
Project: Beat the House - MLB Sports Betting Model (Streamlit app)

This app runs two parallel "betting models" against real MLB games and
tracks how much money each would have made or lost:
  - The Poisson model: predicts win probability from team offense/defense
    stats, and bets whenever it disagrees with the market by enough to be
    worth it (see EDGE_THRESHOLD).
  - The Coin-Flip model: a random-pick control group, bet alongside the
    Poisson model on the same games, so we can tell whether the Poisson
    model is actually smarter than chance.

Both scores paths below hit MLB's free, keyless Stats API - The Odds API is
only used elsewhere in this file for pre-game moneyline odds, never for
scores. They must never overlap on the same calendar dates or the running
balance gets double-counted:
  - The LIVE path (evaluate_and_record_past_games) fetches the last few
    calendar days (today included), one date at a time via
    fetch_historical_scores_for_date.
  - The HISTORICAL/BACKFILL path (run_full_season_backfill ->
    fetch_historical_scores_for_date) fetches by exact date, going all the
    way back to SEASON_START_DATE.

fetch_historical_scores_for_date caches its result to disk per-date, but
ONLY once a date is strictly in the past (Eastern calendar day) - a
still-live day is never cached, so the live path can safely re-poll today's
date on every page load without ever locking in a partial result.
"""
import json
import math
import os
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import polars as pl
import requests
import streamlit as st

from ledger import get_current_balance, load_ledger, record_completed_bet

# Configuration Constants
SEASON_START_DATE = datetime(2026, 3, 25).date()  # Earliest date the backfill will ever reach
DEFAULT_SLATE_DATE = datetime(2026, 8, 11).date()
INITIAL_BANKROLL = 1000000.0  # Default starting bankroll ($1M) for both models
EDGE_THRESHOLD = 0.03   # Minimum gap between model prob and fair market prob before we call it a "bet"
WAGER_SIZE = 5000.0     # Fixed flat wager per recommended bet, for both models
API_KEY = "2e518d0f1afe718f409a16e7dd6d9a98"  # The Odds API key (used for live/historical odds)

HISTORICAL_CACHE_DIR = "historical_cache"
os.makedirs(HISTORICAL_CACHE_DIR, exist_ok=True)

TEAM_ABBREVIATION_MAP = {
    "NYY": "New York Yankees",
    "LAD": "Los Angeles Dodgers",
    "CHC": "Chicago Cubs",
    "TOR": "Toronto Blue Jays",
    "MIL": "Milwaukee Brewers",
    "ARI": "Arizona Diamondbacks",
    "SEA": "Seattle Mariners",
    "BOS": "Boston Red Sox",
    "NYM": "New York Mets",
    "PHI": "Philadelphia Phillies",
    "HOU": "Houston Astros",
    "ATH": "Oakland Athletics",
    "DET": "Detroit Tigers",
    "STL": "St. Louis Cardinals",
    "ATL": "Atlanta Braves",
    "SDP": "San Diego Padres",
    "TBR": "Tampa Bay Rays",
    "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "MIA": "Miami Marlins",
    "MIN": "Minnesota Twins",
    "SFG": "San Francisco Giants",
    "SF": "San Francisco Giants",
    "BAL": "Baltimore Orioles",
    "KCR": "Kansas City Royals",
    "KC": "Kansas City Royals",
    "WSN": "Washington Nationals",
    "WSH": "Washington Nationals",
    "CLE": "Cleveland Guardians",
    "PIT": "Pittsburgh Pirates",
    "CHW": "Chicago White Sox",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "COL": "Colorado Rockies",
    "LAA": "Los Angeles Angels"
}

# Reverse lookup map: Full Name -> Abbreviation
REVERSE_TEAM_MAP = {v: k for k, v in TEAM_ABBREVIATION_MAP.items()}

def get_team_abbrev(name: str) -> str:
    """
    Converts a team name or abbreviation to its standard 3-letter abbreviation,
    for compact display in tables (e.g. ledgers, matchup lists).

    Args:
        name: A team name in any form - full name ("New York Yankees"),
              abbreviation ("NYY"), or empty/None.

    Returns:
        The 3-letter abbreviation if a match is found in REVERSE_TEAM_MAP,
        otherwise the input string unchanged (stripped of whitespace).
        Returns "" if name is empty/None.

    Example:
        get_team_abbrev("New York Yankees") -> "NYY"
        get_team_abbrev("NYY")              -> "NYY"  (already an abbreviation, passes through)
        get_team_abbrev("Some Unknown Team") -> "Some Unknown Team"  (no match, returned as-is)
    """
    if not name:
        return ""
    clean_name = name.strip()
    return REVERSE_TEAM_MAP.get(clean_name, clean_name)

def normalize_team_name(name: str) -> str:
    """
    Converts an abbreviation OR full team name to the standard full team name.
    This is the reverse direction of get_team_abbrev, and is what lets code
    compare team names coming from different data sources (The Odds API,
    MLB Stats API, local CSVs) even when they use different naming
    conventions for the same team.

    Args:
        name: A team name in any form - full name or abbreviation.

    Returns:
        The standardized full team name (e.g. "New York Yankees") if a match
        is found, otherwise the input string unchanged.

    Example:
        normalize_team_name("NYY")               -> "New York Yankees"
        normalize_team_name("New York Yankees")   -> "New York Yankees"  (already standard, passes through)
    """
    clean_name = name.strip()
    return TEAM_ABBREVIATION_MAP.get(clean_name, clean_name)

def fmt_pct(val) -> str:
    """
    Safely formats a float as a percentage string, handling None/NaN inputs
    without raising an exception (e.g. when a model probability couldn't be
    computed for a matchup).

    Args:
        val: A number between 0 and 1 (or anything convertible to float),
             or None/NaN.

    Returns:
        A percentage string like "62.50%", or "N/A" if val is missing/invalid.

    Example:
        fmt_pct(0.625)  -> "62.50%"
        fmt_pct(None)   -> "N/A"
    """
    if val is None or pd.isna(val):
        return "N/A"
    try:
        return f"{float(val):.2%}"
    except (ValueError, TypeError):
        return "N/A"

def safe_balance(val, default: float = INITIAL_BANKROLL) -> float:
    """
    Safely converts a value to a float bankroll amount, handling None/NaN
    without raising an exception, falling back to a default (normally the
    $1,000,000 starting bankroll) when the value is missing or invalid.

    Args:
        val: The raw value to convert (e.g. a cell read from the ledger CSV).
        default: What to return if val is missing/invalid. Defaults to INITIAL_BANKROLL.

    Returns:
        A float dollar amount.

    Example:
        safe_balance(998500.25) -> 998500.25
        safe_balance(None)      -> 1000000.0  (the default)
    """
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def latest_balance(df: pl.DataFrame, default: float = INITIAL_BANKROLL) -> float:
    """
    Returns the current_balance of the chronologically latest row in df, sorted
    by date_str (with an explicit row-index tiebreaker to keep same-day rows in
    their original, already-chronological order).

    This exists because a ledger can have many rows sharing the same date_str
    (one per game settled that day) - to get the TRUE latest balance, we need
    to sort by date first, and then within a date fall back to the original
    file order (which reflects the true order bets were recorded in), rather
    than relying on whatever arbitrary row order the DataFrame happens to be
    in after filtering.

    Args:
        df: A ledger DataFrame (or a date-filtered subset of one) that must
            contain "current_balance" and "date_str" columns.
        default: What to return if df is empty or missing required columns.
                 Defaults to INITIAL_BANKROLL.

    Returns:
        The current_balance (float) of the last game settled, in true
        chronological order, within df. Returns `default` if df is empty
        or missing the required columns.

    Example:
        Given a df with rows (in this file order):
            date_str="2026-04-01", current_balance=1005000
            date_str="2026-04-02", current_balance=1002000
            date_str="2026-04-01", current_balance=1004000   <- written later,
                                                                 but an EARLIER date
        latest_balance(df) sorts by (date_str, original row order) first, so
        the "2026-04-02" row correctly ends up last, and returns 1002000 -
        NOT 1004000, which is simply the last row in the raw file.
    """
    if df.is_empty() or "current_balance" not in df.columns or "date_str" not in df.columns:
        return default
    indexed = df.with_row_index("_orig_idx")  # remember original file order before sorting
    sorted_df = indexed.sort(["date_str", "_orig_idx"])  # sort by date, tiebreak by original order
    return safe_balance(sorted_df["current_balance"].to_list()[-1], default=default)

def attach_date_str(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a "date_str" column (YYYY-MM-DD) derived from the "timestamp" column,
    so ledger rows can be filtered/sorted/grouped by calendar day regardless
    of whatever format the original timestamp was stored in. Need this because 
    ledger timestamps can be in different formats depending on whether they came 
    from The Odds API, MLB Stats API, or were manually entered.

    Args:
        df: A ledger DataFrame that should contain a "timestamp" column.

    Returns:
        The same DataFrame with a new "date_str" column appended. If df is
        empty or has no "timestamp" column, returns df unchanged.

    Example:
        Input row:  timestamp = "2026-04-01T18:05:00-04:00"
        Output row: timestamp = "2026-04-01T18:05:00-04:00", date_str = "2026-04-01"
    """
    if df.is_empty() or "timestamp" not in df.columns:
        return df
    dates_raw = df["timestamp"].to_list()
    formatted_dates = [clean_date_str(d) for d in dates_raw]
    return df.with_columns(pl.Series("date_str", formatted_dates))

def clean_date_str(val) -> str:
    """
    Extracts just the YYYY-MM-DD date portion from an arbitrary date/timestamp
    value, whether it's a full ISO datetime string, a plain date string, or
    something else entirely (falls back to a sentinel date if unparseable).
    Need this because ledger timestamps can be in different formats depending on
    whether they came from The Odds API, MLB Stats API, or were manually entered.

    Args:
        val: Any value representing a date/timestamp - typically a string,
             but could be None/NaN or another object.

    Returns:
        A "YYYY-MM-DD" string. Returns "1970-01-01" as a safe sentinel value
        if val is None/NaN or doesn't look like a valid 10-character date
        once the time portion is stripped off.

    Example:
        clean_date_str("2026-04-01T18:05:00-04:00") -> "2026-04-01"
        clean_date_str("2026-04-01 18:05:00")        -> "2026-04-01"
        clean_date_str("2026-04-01")                 -> "2026-04-01"
        clean_date_str(None)                          -> "1970-01-01"
    """
    if val is None or pd.isna(val):
        return "1970-01-01"
    s = str(val).strip()
    s = s.split("T")[0].split(" ")[0]
    return s if len(s) == 10 else "1970-01-01"

def parse_utc_iso_to_eastern(iso_str: str) -> datetime:
    """
    Parses a UTC ISO-8601 timestamp (as returned by both The Odds API and MLB
    Stats API, e.g. "2026-04-01T22:05:00Z") and converts it to Eastern time -
    the de facto standard timezone MLB schedules/calendar days are read in.

    Raises whatever exception fromisoformat/astimezone would raise (e.g.
    ValueError, AttributeError) on unparseable input - callers are expected
    to wrap this in their own try/except with whatever fallback makes sense
    for that call site (a display placeholder, a naive date-string fallback,
    etc), since the right fallback differs by caller.

    Args:
        iso_str: A UTC ISO-8601 timestamp string, e.g. "2026-04-01T22:05:00Z".

    Returns:
        A timezone-aware datetime in America/New_York.

    Example:
        parse_utc_iso_to_eastern("2026-04-01T22:05:00Z") -> 2026-04-01 18:05:00-04:00
    """
    dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt_utc.astimezone(ZoneInfo("America/New_York"))

def filter_ledger_by_date_range(ledger_df: pl.DataFrame, start_str: str, end_str: str) -> pl.DataFrame:
    """
    Filters a date-stamped ledger DataFrame down to rows whose date_str falls
    within [start_str, end_str] (inclusive). Shared by both the Poisson and
    Coin-Flip ledgers in the Bankroll Tracker tab, which otherwise applied
    this identical filter separately to each.

    Args:
        ledger_df: A ledger DataFrame that has already been through attach_date_str.
        start_str / end_str: Inclusive date bounds as "YYYY-MM-DD" strings.

    Returns:
        The filtered DataFrame, or ledger_df unchanged if it's empty or has
        no "date_str" column.
    """
    if ledger_df.is_empty() or "date_str" not in ledger_df.columns:
        return ledger_df
    return ledger_df.filter(
        (pl.col("date_str") >= start_str) &
        (pl.col("date_str") <= end_str)
    )

def build_balance_chart_series(ledger_df: pl.DataFrame, model_label: str) -> pl.DataFrame:
    """
    Builds one model's end-of-day balance series for the Bankroll Tracker
    chart: one row per calendar date, holding that date's final balance.
    Shared by both the Poisson and Coin-Flip ledgers, which otherwise built
    this identical pipeline separately for each.

    IMPORTANT: maintain_order=True is required in the group_by. A single
    calendar date can have ~10-15 rows (one per MLB game settled that day),
    and Polars' group_by does NOT preserve row order by default - without
    maintain_order, .last() can grab an arbitrary mid-day row instead of the
    true end-of-day balance, making the chart disagree with the correct
    final balance shown above it.

    Args:
        ledger_df: A date-stamped, already date-range-filtered ledger DataFrame.
        model_label: The label to tag this series with, e.g. "Poisson Model Balance".

    Returns:
        A DataFrame with columns [Date, Balance, Model], one row per calendar date.
    """
    return (
        ledger_df
        .select([
            pl.col("date_str").alias("Date"),
            pl.col("current_balance").cast(pl.Float64).alias("Balance")
        ])
        .group_by("Date", maintain_order=True)
        .last()
        .with_columns(pl.lit(model_label).alias("Model"))
    )

# --- DATA PIPELINE FUNCTIONS ---

def generate_team_baseline_matrix():
    """
    Builds the season-long baseline stats table used by the Poisson model:
    one row per MLB team, with runs-per-game (offense) and ERA (defense),
    read from local CSV files and joined together on team name.

    Returns:
        A Polars DataFrame with columns: Team, runs_scored, total_games_played,
        runs_per_game, team_era, standard_team_name. Returns an empty
        DataFrame (and shows a Streamlit error) if the source CSVs can't be
        found.
    """
    try:
        batting_df = pl.read_csv("data/Team batting stats.csv")
        pitching_df = pl.read_csv("data/Team pitching stats.csv")
        
        offense = batting_df.select([
            pl.col("Team"),
            pl.col("R").alias("runs_scored"),
            pl.col("TG").alias("total_games_played"),
            (pl.col("R") / pl.col("TG")).alias("runs_per_game")  # season average runs scored per game
        ])
        
        defense = pitching_df.select([
            pl.col("Team"),
            pl.col("ERA").alias("team_era")  # earned run average = runs allowed per 9 innings (lower is better defense)
        ])
        
        merged_metrics = offense.join(defense, on="Team", how="inner")
        
        final_matrix = merged_metrics.with_columns([
            # Convert each row's abbreviation (e.g. "NYY") to the standard full
            # name so this matrix can be joined/compared against API data later.
            pl.col("Team").replace(TEAM_ABBREVIATION_MAP, default=pl.col("Team")).alias("standard_team_name")
        ])
        
        return final_matrix

    except FileNotFoundError as e:
        st.error(f"PIPELINE FILE ERROR: Could not locate CSV files. Details: {e}")
        return pl.DataFrame()

def fetch_live_api_odds_and_matchups():
    """
    Fetches today's live MLB matchups and moneyline (h2h) odds from The Odds
    API. Only reads the FIRST bookmaker's FIRST market for simplicity - this
    is not an average across books, just whichever book the API lists first.

    Returns:
        A list of dicts, one per game, each with keys: away_team, home_team,
        away_odds, home_odds, game_time. Returns an empty list on any
        request failure or non-200 response.
    """
    SPORT = "baseball_mlb"
    REGIONS = "us"
    MARKETS = "h2h"  # h2h = moneyline (who wins outright), no spreads/totals
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}&oddsFormat=american"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
        
        game_data = response.json()
        parsed_games = []

        for game in game_data:
            away_team = game.get("away_team")
            home_team = game.get("home_team")
            game_time = game.get("commence_time", "N/A")
            away_odds, home_odds = "N/A", "N/A"

            bookmakers = game.get("bookmakers", [])
            if bookmakers:
                markets = bookmakers[0].get("markets", [])
                if markets:
                    for outcome in markets[0].get("outcomes", []):
                        if outcome.get("name") == away_team:
                            away_odds = outcome.get("price")
                        elif outcome.get("name") == home_team:
                            home_odds = outcome.get("price")
                
                parsed_games.append({
                    "away_team": away_team,
                    "home_team": home_team,
                    "away_odds": away_odds,
                    "home_odds": home_odds,
                    "game_time": game_time
                })

        return parsed_games
    except Exception as e:
        print(f"Error fetching live odds: {e}")
        return []

def fetch_historical_scores_for_date(target_date_str: str, api_key: str = None):
    """
    Fetches completed MLB game scores for a specific date (YYYY-MM-DD) using
    the free MLB Stats API (statsapi.mlb.com).

    Caches JSON locally in historical_cache/ to avoid redundant network calls
    on subsequent runs for the same date - but ONLY for dates that are
    guaranteed fully complete (strictly before today's Eastern calendar
    date). Today's date (and any date that isn't in the past yet) is never
    cached, since games may still be in progress - caching a partial or
    empty result for a live day would mean any games that finish later never
    get re-checked. Everything from yesterday and earlier is a genuine
    finished day (or a genuine off-day/rainout), so it's safe and correct to
    cache permanently.

    Args:
        target_date_str: The date to fetch, as "YYYY-MM-DD".
        api_key: Unused - kept for call-site backward compatibility.

    Returns:
        A list of dicts already in the app's normalized format:
        away_team, home_team, winner, away_score, home_score, commence_time.
        Returns an empty list for off-days/rainouts or on request failure.
    """
    eastern_today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    is_finalized_date = target_date_str < eastern_today_str

    cache_file = os.path.join(HISTORICAL_CACHE_DIR, f"scores_{target_date_str}.json")

    if is_finalized_date and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={target_date_str}"
    parsed_games = []

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            payload = response.json()

            for date_block in payload.get("dates", []):
                for game in date_block.get("games", []):
                    # Only settle games that have actually finished
                    if game.get("status", {}).get("abstractGameState") != "Final":
                        continue

                    teams = game.get("teams", {})
                    away = teams.get("away", {})
                    home = teams.get("home", {})

                    away_team = away.get("team", {}).get("name")
                    home_team = home.get("team", {}).get("name")
                    away_score = away.get("score")
                    home_score = home.get("score")

                    if not away_team or not home_team:
                        continue
                    if away_score is None or home_score is None:
                        continue
                    if away_score == home_score:
                        continue

                    winner = away_team if away_score > home_score else home_team
                    commence_time = game.get("gameDate", f"{target_date_str}T12:00:00Z")

                    parsed_games.append({
                        "away_team": away_team,
                        "home_team": home_team,
                        "winner": winner,
                        "away_score": away_score,
                        "home_score": home_score,
                        "commence_time": commence_time
                    })

            print(f"MLB Stats API OK for {target_date_str}: games_found={len(parsed_games)}")
        else:
            print(f"MLB Stats API request failed for {target_date_str}: "
                  f"HTTP {response.status_code} - {response.text[:300]}")
    except Exception as e:
        print(f"Error fetching MLB Stats API scores for {target_date_str}: {e}")

    # Cache the result (even an empty list, for genuine off-days/rainouts) so
    # we don't re-hit the network for this date again - but only once the
    # date is fully in the past. Caching a live/today result here would risk
    # locking in an incomplete list of games forever.
    if is_finalized_date:
        with open(cache_file, "w") as f:
            json.dump(parsed_games, f)

    return parsed_games

def fetch_historical_odds_for_date(target_date_str: str) -> dict:
    """
    Fetches real historical H2H (moneyline) odds for MLB games on a given
    date, from The Odds API's historical *odds* endpoint (a snapshot taken
    at a fixed time each day, since The Odds API charges per-snapshot for
    historical odds and doesn't offer full-day coverage).

    Caches JSON locally in historical_cache/ to avoid re-spending API
    quota/credits on repeated backfill runs for the same date.

    Args:
        target_date_str: The date to fetch, as "YYYY-MM-DD". Internally
            requests a snapshot at 22:00 UTC (~6pm Eastern), which is late
            enough to catch odds for most MLB games that day.

    Returns:
        A dict keyed by "{away full name} @ {home full name}", each value a
        dict with "away_odds" and "home_odds" (American odds as floats, or
        None if unavailable for that matchup). Returns an empty dict on
        failure.
    """
    cache_file = os.path.join(HISTORICAL_CACHE_DIR, f"odds_{target_date_str}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            raw = json.load(f)
    else:
        SPORT = "baseball_mlb"
        snapshot_iso = f"{target_date_str}T22:00:00Z"
        url = (
            f"https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds/"
            f"?apiKey={API_KEY}&bookmakers=draftkings&markets=h2h&oddsFormat=american&date={snapshot_iso}"
        )
        raw = {}
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                raw = response.json()
                with open(cache_file, "w") as f:
                    json.dump(raw, f)
            else:
                print(f"Historical odds request failed for {target_date_str}: "
                      f"HTTP {response.status_code} - {response.text[:300]}")
        except Exception as e:
            print(f"Error fetching historical odds for {target_date_str}: {e}")


    odds_by_matchup = {}
    for game in raw.get("data", []) if isinstance(raw, dict) else []:
        away_team = game.get("away_team")
        home_team = game.get("home_team")
        if not away_team or not home_team:
            continue

        bookmakers = game.get("bookmakers", [])
        away_odds = _select_representative_price(bookmakers, away_team)
        home_odds = _select_representative_price(bookmakers, home_team)

        key = f"{normalize_team_name(away_team)} @ {normalize_team_name(home_team)}"
        odds_by_matchup[key] = {"away_odds": away_odds, "home_odds": home_odds}

    return odds_by_matchup

PLAUSIBLE_ODDS_CAP = 1000.0  # Sanity ceiling: American odds beyond +/-1000 are almost certainly bad/junk data

def _select_representative_price(bookmakers: list, team_name: str):
    """
    Picks team_name's moneyline price specifically from DraftKings (chosen
    as a single consistent reference book rather than averaging across many,
    which would mix different books' vig and timing).

    Args:
        bookmakers: The list of bookmaker dicts from an Odds API game object.
        team_name: The exact team name string to look up the price for.

    Returns:
        The American odds price (float/int) if found and within
        PLAUSIBLE_ODDS_CAP, otherwise None (e.g. DraftKings didn't list this
        game, or the price looks like bad data).
    """
    by_key = {b.get("key"): b for b in bookmakers}
    draftkings = by_key.get("draftkings")
    if not draftkings:
        return None

    markets = draftkings.get("markets", [])
    if not markets:
        return None

    for outcome in markets[0].get("outcomes", []):
        if outcome.get("name") == team_name:
            price = outcome.get("price")
            if price is not None and abs(price) <= PLAUSIBLE_ODDS_CAP:
                return price
            return None

    return None

def get_odds_for_matchup(odds_cache_by_date: dict, date_str: str, away_team: str, home_team: str) -> tuple:
    """
    Looks up real DraftKings moneyline odds for a specific matchup from a
    per-date odds cache (fetching and populating that cache on first use for
    a given date, so each date only triggers one network call no matter how
    many games on it need odds looked up).

    Args:
        odds_cache_by_date: A dict of {date_str: odds_by_matchup}, mutated
            in place as new dates are looked up (typically a fresh dict
            created once per backfill run and passed through).
        date_str: The date the game took place, as "YYYY-MM-DD".
        away_team: Away team name (any form - will be normalized internally).
        home_team: Home team name (any form - will be normalized internally).

    Returns:
        A (away_odds, home_odds) tuple. Each side is a float if real
        DraftKings odds were found for that matchup/date, otherwise None.
        There is NO fabricated fallback (e.g. -110) - callers are expected
        to skip recording a bet for any game where either side comes back
        None, since we'd otherwise be settling a bet at a price nobody
        could have actually gotten.

    Example:
        get_odds_for_matchup({}, "2026-04-01", "NYY", "BOS")
        -> (-135.0, 115.0)  # if real DraftKings odds were found
        -> (None, None)     # if no odds data existed for that matchup/date
    """
    if date_str not in odds_cache_by_date:
        odds_cache_by_date[date_str] = fetch_historical_odds_for_date(date_str)

    key = f"{normalize_team_name(away_team)} @ {normalize_team_name(home_team)}"
    matchup_odds = odds_cache_by_date[date_str].get(key, {})
    away_odds = matchup_odds.get("away_odds")
    home_odds = matchup_odds.get("home_odds")

    return (
        float(away_odds) if away_odds is not None else None,
        float(home_odds) if home_odds is not None else None,
    )

# --- MODEL MATH ENGINE ---

def convert_market_odds_to_implied_prob_and_remove_juice(market_away_odds, market_home_odds):
    """
    Converts American odds for both sides of a game into "fair" (de-vigged)
    win probabilities.

    Sportsbooks price both sides so their implied probabilities sum to
    slightly MORE than 100% - that extra bit is the "vig" or "juice", which
    is how the book guarantees itself a profit margin regardless of outcome.
    This function reverses that: it converts each side's odds to a raw
    implied probability, then rescales both so they sum to exactly 100%,
    giving a "fair" probability estimate with the book's edge removed.

    Args:
        market_away_odds: American odds for the away team (e.g. -150, +130).
        market_home_odds: American odds for the home team.

    Returns:
        A (fair_away_prob, fair_home_prob) tuple of floats that sum to 1.0.

    Step-by-step example:
        Away team is a favorite at -150, home team is an underdog at +130.

        1. Convert away odds (-150, negative = favorite) to raw implied prob:
           raw_away = 150 / (150 + 100) = 150 / 250 = 0.600  (60.0%)

        2. Convert home odds (+130, positive = underdog) to raw implied prob:
           raw_home = 100 / (130 + 100) = 100 / 230 = 0.4348  (43.48%)

        3. Sum the two raw probabilities (this is > 1.0, which is the "vig"):
           total = 0.600 + 0.4348 = 1.0348  (103.48%)

        4. Rescale each side by dividing by that total, so they sum to
           exactly 1.0:
           fair_away = 0.600 / 1.0348 = 0.5798  (57.98%)
           fair_home = 0.4348 / 1.0348 = 0.4202  (42.02%)

        Result: convert_market_odds_to_implied_prob_and_remove_juice(-150, 130)
                -> (0.5798, 0.4202)
        Notice 0.5798 + 0.4202 = 1.0 exactly, unlike the raw 103.48% total -
        the ~3.5% gap was the book's built-in edge, now removed.
    """
    if market_away_odds < 0:
        # Negative odds (favorite): implied prob = |odds| / (|odds| + 100)
        juiced_market_away_prob = abs(market_away_odds) / (abs(market_away_odds) + 100) 
    else:
        # Positive odds (underdog): implied prob = 100 / (odds + 100)
        juiced_market_away_prob = 100 / (market_away_odds + 100)

    if market_home_odds < 0:
        juiced_market_home_prob = abs(market_home_odds) / (abs(market_home_odds) + 100)
    else:
        juiced_market_home_prob = 100 / (market_home_odds + 100)

    total_market_prob = juiced_market_away_prob + juiced_market_home_prob 
    return juiced_market_away_prob / total_market_prob, juiced_market_home_prob / total_market_prob

def estimate_matchup_probabilities(away_team, home_team, matrix_df):
    """
    THE CORE POISSON MODEL. Estimates each team's expected runs scored
    (lambda) against this specific opponent by scaling team offense/defense
    relative to league average, then uses a normal approximation of the
    Skellam distribution (the distribution of the DIFFERENCE between two
    independent Poisson-distributed variables - here, away runs minus home
    runs) to convert that into a win probability for the away team.

    Args:
        away_team: Away team name (any form - normalized internally).
        home_team: Home team name (any form - normalized internally).
        matrix_df: The baseline stats matrix from generate_team_baseline_matrix(),
            with a "standard_team_name" column plus "runs_per_game" and
            "team_era" per team.

    Returns:
        A (away_win_prob, home_win_prob) tuple of floats summing to 1.0, or
        (None, None) if matrix_df is empty or either team isn't found in it
        (e.g. a name normalization miss).

    Step-by-step example (illustrative numbers):
        Say league averages are: avg runs/game = 4.5, avg ERA = 4.00.
        Away team (a good offense) scores 5.4 runs/game on average.
        Home team (a good pitching staff) has a team ERA of 3.20.
        Home team's own offense scores 4.2 runs/game.
        Away team's own pitching has a team ERA of 4.50.

        1. Expected AWAY runs this game (lambda_away) = away team's offense
           strength x home team's defense weakness x league average:
             lambda_away = (5.4 / 4.5) * (3.20 / 4.00) * 4.5
                          = 1.20 * 0.80 * 4.5
                          = 4.32 expected runs for the away team

        2. Expected HOME runs this game (lambda_home) = home offense x away
           defense x league average:
             lambda_home = (4.2 / 4.5) * (4.50 / 4.00) * 4.5
                          = 0.9333 * 1.125 * 4.5
                          = 4.725 expected runs for the home team

        3. Treat each team's run total as an independent Poisson variable
           with those means. The DIFFERENCE (away runs - home runs) then has:
             mean_diff     = lambda_away - lambda_home = 4.32 - 4.725 = -0.405
             variance_diff = lambda_away + lambda_home = 4.32 + 4.725 = 9.045
           (variance adds for independent variables, even under subtraction)

        4. Convert to a z-score (standardized distance from zero):
             z = mean_diff / sqrt(variance_diff) = -0.405 / sqrt(9.045)
               = -0.405 / 3.0075 = -0.1347

        5. Convert the z-score to a probability using the standard normal
           CDF (erf form): P(away wins) = P(diff > 0) under this normal
           approximation:
             away_win_prob = 0.5 * (1 + erf(z / sqrt(2)))
                            = 0.5 * (1 + erf(-0.1347 / 1.4142))
                            = 0.5 * (1 + erf(-0.0953))
                            = 0.5 * (1 + (-0.1072))
                            = 0.5 * 0.8928
                            = 0.4464  (44.64% chance the away team wins)

        Result: away_win_prob ~= 0.4464, home_win_prob = 1 - 0.4464 = 0.5536
        This makes sense: even though the away team's raw offense is
        stronger, the home team's excellent pitching (low ERA) drags the
        away team's expected runs down enough that the home team is still
        favored overall.
    """
    if matrix_df is None or matrix_df.is_empty():
        return None, None

    norm_away = normalize_team_name(away_team)
    norm_home = normalize_team_name(home_team)

    valid_teams = matrix_df["standard_team_name"].to_list()
    if norm_away not in valid_teams or norm_home not in valid_teams:
        return None, None

    league_avg_rpg = matrix_df["runs_per_game"].mean()
    league_avg_era = matrix_df["team_era"].mean()

    away_offense = matrix_df.filter(pl.col("standard_team_name") == norm_away)["runs_per_game"].item()
    away_defense = matrix_df.filter(pl.col("standard_team_name") == norm_away)["team_era"].item()
    home_offense = matrix_df.filter(pl.col("standard_team_name") == norm_home)["runs_per_game"].item()
    home_defense = matrix_df.filter(pl.col("standard_team_name") == norm_home)["team_era"].item()

    # Expected runs = (team's offense relative to league avg) x (opponent's
    # defense relative to league avg) x league avg runs/game
    lambda_away = (away_offense / league_avg_rpg) * (home_defense / league_avg_era) * league_avg_rpg
    lambda_home = (home_offense / league_avg_rpg) * (away_defense / league_avg_era) * league_avg_rpg

    # Normal approximation to Skellam: mean = difference of the two lambdas,
    # variance = sum of the two lambdas (since Var(X-Y) = Var(X)+Var(Y) for
    # independent Poisson variables).
    mean_diff = lambda_away - lambda_home 
    variance_diff = lambda_away + lambda_home 
    z_score = mean_diff / math.sqrt(variance_diff)

    # Convert z-score to a probability via the standard normal CDF (erf form)
    away_win_prob = 0.5 * (1 + (math.erf(z_score / math.sqrt(2))))
    return away_win_prob, 1 - away_win_prob

def get_game_data_dataframe(matrix_df, live_games=None):
    """
    Builds the "today's odds" table shown in Tab 1: for each live matchup,
    computes the model's win probability, the de-vigged market probability,
    and flags a bet recommendation whenever the model disagrees with the
    market by more than EDGE_THRESHOLD.

    Args:
        matrix_df: The baseline stats matrix from generate_team_baseline_matrix().
        live_games: Optionally pass in a pre-fetched list of live games
            (skips the network call); if None, fetches fresh via
            fetch_live_api_odds_and_matchups().

    Returns:
        A Polars DataFrame sorted by game time, with one row per matchup
        that has usable odds, including columns for the model's projected
        probabilities, the fair (de-vigged) market probabilities, and a
        RECOMMENDATION column. Returns an empty DataFrame if there are no
        games with usable odds.
    """
    if live_games is None:
        live_games = fetch_live_api_odds_and_matchups()

    away_teams, home_teams, market_away_odds, market_home_odds, game_times = [], [], [], [], []

    for game in live_games:
        if game["away_odds"] == "N/A" or game["home_odds"] == "N/A": 
            continue
        game_times.append(game["game_time"])
        away_teams.append(game["away_team"])
        home_teams.append(game["home_team"])
        market_away_odds.append(game["away_odds"])
        market_home_odds.append(game["home_odds"])

    if not away_teams:
        return pl.DataFrame()

    model_away_probabilities, model_home_probabilities = [], []
    juiceless_market_away_odds, juiceless_market_home_odds = [], []
    formatted_times, bet_recommendations = [], []

    for away, home, away_odds, home_odds, raw_time in zip(away_teams, home_teams, market_away_odds, market_home_odds, game_times):
        if raw_time != "N/A":
            try:
                # Odds API times are UTC - convert to Eastern for display since that's
                # the de facto standard timezone MLB schedules are usually read in.
                time_str = parse_utc_iso_to_eastern(raw_time).strftime("%I:%M %p")
            except Exception:
                time_str = "N/A"
        else:
            time_str = "N/A"
        formatted_times.append(time_str)

        juiceless_away_prob, juiceless_home_prob = convert_market_odds_to_implied_prob_and_remove_juice(away_odds, home_odds)
        juiceless_market_away_odds.append(juiceless_away_prob)
        juiceless_market_home_odds.append(juiceless_home_prob)

        market_away, market_home = estimate_matchup_probabilities(away, home, matrix_df)
        model_away_probabilities.append(market_away)
        model_home_probabilities.append(market_home)

        # "Value" bet = model thinks a side is meaningfully more likely to win
        # than the de-vigged market price suggests.
        if market_away is None or market_home is None:
            bet_recommendations.append("NO CLEAR VALUE")
        elif market_away - juiceless_away_prob > EDGE_THRESHOLD:
            bet_recommendations.append(f"BET ON {get_team_abbrev(away)}")
        elif market_home - juiceless_home_prob > EDGE_THRESHOLD:
            bet_recommendations.append(f"BET ON {get_team_abbrev(home)}")
        else:
            bet_recommendations.append("NO CLEAR VALUE")

    return pl.DataFrame({
        "RECOMMENDATION": bet_recommendations,
        "Game Time (EST)": formatted_times,
        "Away Team": away_teams,
        "Home Team": home_teams,
        "My Personal Away Projection": model_away_probabilities,
        "My Personal Home Projection": model_home_probabilities,
        "Away Market Odds": market_away_odds,
        "Home Market Odds": market_home_odds,
        "Fair Market Away Prob %": juiceless_market_away_odds,
        "Fair Market Home Prob %": juiceless_market_home_odds,
    }).sort("Game Time (EST)")

def build_already_recorded_keys(poisson_ledger: pl.DataFrame) -> set:
    """
    Builds the "already settled" key set from an existing Poisson ledger, so
    a settlement pass never records the same game twice. Shared by both the
    live path (evaluate_and_record_past_games) and the backfill path
    (run_full_season_backfill), which otherwise duplicated this exact logic.

    Key = normalized team names + the calendar date (Eastern time) the game
    belongs to - NOT a game ID, since neither data source gives us a stable
    cross-source ID to match on.

    Args:
        poisson_ledger: The Poisson model's ledger, as returned by load_ledger("poisson").

    Returns:
        A set of "AwayTeam @ HomeTeam | YYYY-MM-DD" strings (normalized full
        team names). Empty set if the ledger is empty or has no "game" column.
    """
    keys = set()
    if poisson_ledger.is_empty() or "game" not in poisson_ledger.columns:
        return keys
    for row in poisson_ledger.iter_rows(named=True):
        g_str = row.get("game", "")
        if " @ " not in g_str:
            continue
        ts_str = clean_date_str(row.get("timestamp"))
        a_team, h_team = g_str.split(" @ ")
        keys.add(f"{normalize_team_name(a_team)} @ {normalize_team_name(h_team)} | {ts_str}")
    return keys

def record_settled_game(
    game_key: str,
    away_abbrev: str,
    home_abbrev: str,
    poisson_pick: str,
    away_odds: float,
    home_odds: float,
    actual_winner: str,
    away_score,
    home_score,
    timestamp_str: str,
) -> str:
    """
    Records one settled game's bet for BOTH models against the shared
    ledger: the Poisson model's pick (passed in, since evaluate_and_record_past_games
    and run_full_season_backfill each derive it differently - edge-threshold
    logic vs plain-favorite logic) and the Coin-Flip model's pick (a fresh
    random choice made here). Shared by both callers to avoid duplicating
    this record-both-models bookkeeping in each.

    Args:
        game_key: Display string for the matchup, e.g. "NYY @ BOS".
        away_abbrev / home_abbrev: 3-letter team abbreviations for this game.
        poisson_pick: The Poisson model's chosen team abbreviation.
        away_odds / home_odds: Historical moneyline odds for each side.
        actual_winner: The actual winning team (normalized internally for comparison).
        away_score / home_score: Final score for each side, for ledger display.
        timestamp_str: The calendar date (YYYY-MM-DD) to record both bets under.

    Returns:
        The Coin-Flip model's randomly chosen team abbreviation, in case the
        caller wants it for its own bookkeeping.
    """
    record_completed_bet(
        model_type="poisson",
        game_name=game_key,
        team_bet=poisson_pick,
        amount=WAGER_SIZE,
        odds=away_odds if poisson_pick == away_abbrev else home_odds,
        won=(normalize_team_name(poisson_pick) == normalize_team_name(actual_winner)),
        extra_data={"score": f"{away_score}-{home_score}", "timestamp": timestamp_str}
    )

    random_pick = random.choice([away_abbrev, home_abbrev])
    record_completed_bet(
        model_type="random",
        game_name=game_key,
        team_bet=random_pick,
        amount=WAGER_SIZE,
        odds=away_odds if random_pick == away_abbrev else home_odds,
        won=(normalize_team_name(random_pick) == normalize_team_name(actual_winner)),
        extra_data={"score": f"{away_score}-{home_score}", "timestamp": timestamp_str}
    )

    return random_pick

def evaluate_and_record_past_games(matrix_df, days_back: int = 3):
    """
    THE "LIVE" SETTLEMENT PATH - runs automatically on every page load (see
    the APP INITIALIZATION section below), and also manually via the "Run
    Daily Betting Pipeline" button. Pulls the last few days of completed
    games from MLB Stats API (via fetch_historical_scores_for_date, once per
    day) and records both models' picks for any game not already in the
    ledger. This is the ONLY path that should ever touch the trailing few
    days - see run_full_season_backfill's end_date buffer for why.

    Args:
        matrix_df: The baseline stats matrix from generate_team_baseline_matrix().
        days_back: How many calendar days back to check for completed games,
            including today. Must be at least 1.

    Returns:
        The number of newly settled games recorded (int). Games already
        present in the ledger are silently skipped, so calling this
        repeatedly is safe/idempotent.
    """
    valid_days = max(1, days_back)
    eastern_today = datetime.now(ZoneInfo("America/New_York")).date()

    actual_results = []
    for offset in range(valid_days):
        target_date = eastern_today - timedelta(days=offset)
        actual_results.extend(fetch_historical_scores_for_date(target_date.strftime("%Y-%m-%d")))

    if not actual_results:
        return 0

    existing_poisson = load_ledger("poisson")
    already_recorded_normalized = build_already_recorded_keys(existing_poisson)

    settled_count = 0
    odds_cache_by_date = {}

    for score_info in actual_results:
        away_team_api = score_info["away_team"]
        home_team_api = score_info["home_team"]

        away_abbrev = get_team_abbrev(away_team_api)
        home_abbrev = get_team_abbrev(home_team_api)
        game_key = f"{away_abbrev} @ {home_abbrev}"
        game_timestamp = score_info.get("commence_time") or datetime.now().isoformat()

        try:
            # MLB Stats API gives commence_time in UTC - convert to Eastern so
            # "which calendar day" a late-night game belongs to matches how
            # humans (and MLB's own schedule) would describe it.
            local_date_str = parse_utc_iso_to_eastern(game_timestamp).strftime("%Y-%m-%d")
        except Exception:
            local_date_str = clean_date_str(game_timestamp)

        norm_api_key = f"{normalize_team_name(away_team_api)} @ {normalize_team_name(home_team_api)} | {local_date_str}"
        
        if norm_api_key in already_recorded_normalized:
            continue
            
        actual_winner = score_info["winner"]
        away_s = score_info.get("away_score", 0)
        home_s = score_info.get("home_score", 0)

        away_prob, home_prob = estimate_matchup_probabilities(away_team_api, home_team_api, matrix_df)
        if away_prob is None:
            continue

        # Bet recommendation logic: prefer whichever side clears the edge
        # threshold against a flat 50/50 baseline; if neither clears it,
        # just take whichever side the model likes better (no "pass" option
        # here - unlike the live-odds tab, every settled game gets a pick).
        standard_implied_prob = 0.50 
        poisson_pick = None
        
        if away_prob - standard_implied_prob > EDGE_THRESHOLD:
            poisson_pick = away_abbrev
        elif home_prob - standard_implied_prob > EDGE_THRESHOLD:
            poisson_pick = home_abbrev
        else:
            poisson_pick = away_abbrev if away_prob > home_prob else home_abbrev

        # Look up the real historical moneyline odds for this matchup/date so
        # payouts reflect what a real bettor would actually have gotten. If
        # DraftKings didn't have a price for this game, skip it entirely
        # rather than settling it at a fabricated/default price.
        away_odds, home_odds = get_odds_for_matchup(
            odds_cache_by_date, local_date_str, away_team_api, home_team_api
        )
        if away_odds is None or home_odds is None:
            continue

        record_settled_game(
            game_key=game_key,
            away_abbrev=away_abbrev,
            home_abbrev=home_abbrev,
            poisson_pick=poisson_pick,
            away_odds=away_odds,
            home_odds=home_odds,
            actual_winner=actual_winner,
            away_score=away_s,
            home_score=home_s,
            timestamp_str=local_date_str,
        )

        settled_count += 1

    return settled_count

def run_full_season_backfill(matrix_df, api_key: str):
    """
    THE HISTORICAL BACKFILL PATH - triggered manually via the "Backfill
    Season" button. Iterates day-by-day from SEASON_START_DATE up to (today
    minus a buffer) and settles any completed games not already in the
    ledger, using MLB Stats API as the score source (see
    fetch_historical_scores_for_date) and The Odds API's historical odds
    snapshots for pricing (see fetch_historical_odds_for_date).

    Args:
        matrix_df: The baseline stats matrix from generate_team_baseline_matrix().
        api_key: Unused - kept for call-site backward compatibility, since
            MLB Stats API needs no key (The Odds API key is read from the
            global API_KEY constant instead).

    Returns:
        The total number of newly settled games across the whole backfill
        range (int). Also updates a Streamlit progress bar/status text as
        it runs, since this can take a while for a full season.
    """
    start_date = SEASON_START_DATE
    # Stop 4 days short of today: the daily live pipeline (evaluate_and_record_past_games,
    # which runs on every page load with days_back=3) already exclusively owns that trailing
    # window. Using a 4-day buffer (rather than exactly 3) keeps a one-day margin of safety
    # at the boundary rather than cutting it exactly at the live path's edge.
    end_date = datetime.now().date() - timedelta(days=4)
    
    current_eval_date = start_date
    total_settled = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_days = (end_date - start_date).days + 1
    day_count = 0
    odds_cache_by_date = {}

    while current_eval_date <= end_date:
        date_str = current_eval_date.strftime("%Y-%m-%d")
        status_text.text(f"Backfilling games for date: {date_str}...")

        # Re-load and re-scan the ledger on every single day of the loop
        # (rather than once up front) so that games recorded earlier in THIS
        # SAME backfill run are also picked up as "already recorded" -
        # otherwise a bug or a re-run partway through could double-record.
        existing_poisson = load_ledger("poisson")
        already_recorded = build_already_recorded_keys(existing_poisson)

        parsed_games = fetch_historical_scores_for_date(date_str, api_key)

        for score_info in parsed_games:
            away_team_api = score_info["away_team"]
            home_team_api = score_info["home_team"]

            away_abbrev = get_team_abbrev(away_team_api)
            home_abbrev = get_team_abbrev(home_team_api)
            game_key = f"{away_abbrev} @ {home_abbrev}"
            
            norm_key = f"{normalize_team_name(away_team_api)} @ {normalize_team_name(home_team_api)} | {date_str}"
            if norm_key in already_recorded:
                continue

            actual_winner = score_info["winner"]
            away_s = score_info["away_score"]
            home_s = score_info["home_score"]

            away_prob, home_prob = estimate_matchup_probabilities(away_team_api, home_team_api, matrix_df)
            if away_prob is None:
                continue

            # Note: unlike evaluate_and_record_past_games, the backfill path
            # has no edge-threshold check here - it simply takes whichever
            # side the model favors, every time.
            poisson_pick = away_abbrev if away_prob > home_prob else home_abbrev

            # Skip this game entirely if DraftKings didn't have a price for
            # it on this date - we don't want to settle a bet at a
            # fabricated/default odds price.
            away_odds, home_odds = get_odds_for_matchup(
                odds_cache_by_date, date_str, away_team_api, home_team_api
            )
            if away_odds is None or home_odds is None:
                continue

            record_settled_game(
                game_key=game_key,
                away_abbrev=away_abbrev,
                home_abbrev=home_abbrev,
                poisson_pick=poisson_pick,
                away_odds=away_odds,
                home_odds=home_odds,
                actual_winner=actual_winner,
                away_score=away_s,
                home_score=home_s,
                timestamp_str=date_str,
            )

            already_recorded.add(norm_key)  # keep this loop's in-memory set current too
            total_settled += 1

        day_count += 1
        progress_bar.progress(min(day_count / total_days, 1.0))
        current_eval_date += timedelta(days=1)

    status_text.text(f"Backfill Complete! Settled {total_settled} historical games.")

    return total_settled

# --- APP INITIALIZATION ---
# Everything below this point runs top-to-bottom on EVERY Streamlit rerun
# (i.e. every time the page loads or the user interacts with any widget).

st.set_page_config(page_title="Beat the House", layout="wide")

matrix_df = generate_team_baseline_matrix()

today_date = datetime.now().date()
# Auto-sync: settle the last 3 days of completed games every time the page loads,
# so the ledger stays current without the user needing to click anything.
evaluate_and_record_past_games(matrix_df, days_back=3)

today_games_df = get_game_data_dataframe(matrix_df)

# These are the absolute-latest (unfiltered) balances, used for the header
# display at the top of the page - NOT the same as the date-filtered balances
# shown inside the Bankroll Tracker tab further down.
current_bankroll = latest_balance(attach_date_str(load_ledger("poisson")), default=INITIAL_BANKROLL)
random_current_bankroll = latest_balance(attach_date_str(load_ledger("random")), default=INITIAL_BANKROLL)

st.title("Project: Beat the House ⚾")
st.header("MLB Sports Betting Model - 2026 Season")


st.caption("NOTE: GAME DATA AVAILABLE UNTIL AUGUST 11 2026")
st.markdown(f"# **${current_bankroll:,.2f}**")

def style_custom_dataframe(df_to_style):
    """
    Applies Streamlit/pandas Styler formatting to a matchup or ledger table:
    picks a curated subset/order of display columns, formats probability
    columns as percentages, and color-highlights rows for wins (green),
    losses (red), and "BET ON" recommendations (green).

    Args:
        df_to_style: A Polars DataFrame with some combination of the
            recognized display columns (RECOMMENDATION, Game Time (EST),
            Game Date, Away Team, Home Team, Game Score, Bet Result, and the
            two projection columns).

    Returns:
        A pandas Styler object (from df.style.apply(...)) ready to be passed
        to st.dataframe().
    """
    pd_display = df_to_style.to_pandas().copy()

    # Only these columns are ever shown, and in this fixed order - both the
    # Tab 1 (today's odds) and Tab 2 (past matchups) tables pass through
    # this same function, so it accepts a mixed superset of both their
    # columns and just keeps whichever ones are actually present.
    display_columns = [
        "RECOMMENDATION",
        "Game Time (EST)",
        "Game Date",
        "Away Team",
        "Home Team",
        "Game Score",
        "Bet Result",
        "My Personal Away Projection",
        "My Personal Home Projection"
    ]
    
    pd_display = pd_display[[c for c in display_columns if c in pd_display.columns]]

    for col in ["My Personal Away Projection", "My Personal Home Projection"]:
        if col in pd_display.columns:
            pd_display[col] = pd_display[col].apply(fmt_pct)

    def highlight_rows_or_cells(row):
        """Row-styling callback for pandas Styler: red for a settled loss,
        green for a settled win OR a live 'BET ON' recommendation, otherwise
        no special styling."""
        styles = [''] * len(row)
        rec_str = str(row.get("RECOMMENDATION", ""))
        bet_res = str(row.get("Bet Result", ""))

        if "LOSS" in bet_res:
            return ['background-color: #3a1e1e; color: #ff4d4f; font-weight: bold'] * len(row)
        if "WIN" in bet_res:
            return ['background-color: #1e3a29; color: #52c41a; font-weight: bold'] * len(row)

        if "BET ON" in rec_str:
            return ['background-color: #1e3a29; color: #52c41a; font-weight: bold'] * len(row)

        return styles

    return pd_display.style.apply(highlight_rows_or_cells, axis=1)

def render_date_range_picker(state_prefix: str, label: str, today_date, season_start_date=SEASON_START_DATE):
    """
    Renders a "Filter ... Date Range" widget with a "Reset to full range"
    fallback, namespacing its session_state keys under state_prefix so
    multiple independent pickers (Tab 2's Past Matchups filter, Tab 3's
    Bankroll Tracker filter) don't collide. Both tabs had this exact
    boilerplate duplicated before; this centralizes it.

    Handles two Streamlit quirks:
    - A `key`'d date_input only honors its default on first render, so
      "keep the end date pinned to today" has to be re-synced via
      session_state directly, tracked against a `_{state_prefix}_last_today` flag.
    - The "Reset to full range" button can't overwrite the date_input's
      session_state value and rerun in the same script pass, so it sets a
      `_{state_prefix}_reset_flag` and reruns; the reset is applied on the
      NEXT pass, before the widget is (re)created.

    Args:
        state_prefix: Unique namespace for this picker's session_state keys
            (e.g. "past_matchups", "bankroll_tracker").
        label: The label shown above the date_input widget.
        today_date: Today's date, used as the default end date and max_value.
        season_start_date: Earliest selectable date (default: SEASON_START_DATE).

    Returns:
        (start_date, end_date) as date objects - always a valid pair, even
        mid-selection (a single picked date is treated as both start and end).
    """
    range_key = f"{state_prefix}_date_range"
    last_today_key = f"_{state_prefix}_last_today"
    reset_flag_key = f"_{state_prefix}_reset_flag"

    if st.session_state.get(last_today_key) != today_date:
        st.session_state[range_key] = (season_start_date, today_date)
        st.session_state[last_today_key] = today_date

    if st.session_state.pop(reset_flag_key, False):
        st.session_state[range_key] = (season_start_date, today_date)
        st.session_state[last_today_key] = today_date

    selected_range = st.date_input(
        label,
        min_value=season_start_date,
        max_value=today_date,
        key=range_key
    )

    # date_input can momentarily return a single date (not a 2-tuple) while
    # the user is mid-selection (e.g. they've only clicked a start date so
    # far) - offer a reset button in that state so they're not stuck.
    if not (isinstance(selected_range, tuple) and len(selected_range) == 2):
        if st.button("Reset to full range", key=f"{state_prefix}_reset_range"):
            st.session_state[reset_flag_key] = True
            st.rerun()

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        return selected_range
    picked = selected_range[0] if isinstance(selected_range, tuple) else selected_range
    return picked, picked

def fetch_past_slate(date_str: str) -> list:
    """
    Rebuilds a past day's slate in the same shape that
    fetch_live_api_odds_and_matchups() returns, using the historical odds
    snapshot for that date (see fetch_historical_odds_for_date). This lets the
    Today's Matchup tab show what a past days with games looks like.
 
    Only games that start on date_str (Eastern time) are kept, and a game is
    skipped if DraftKings had no price for either side.
 
    Args:
        date_str: The day to show, as "YYYY-MM-DD".
 
    Returns:
        A list of dicts with keys: away_team, home_team, away_odds,
        home_odds, game_time. Empty if no odds data exists for that date.
    """
    fetch_historical_odds_for_date(date_str)  # makes sure the odds file for this date is cached
    cache_file = os.path.join(HISTORICAL_CACHE_DIR, f"odds_{date_str}.json")
    if not os.path.exists(cache_file):
        return []
 
    with open(cache_file, "r") as f:
        raw = json.load(f)
 
    games = []
    for game in raw.get("data", []) if isinstance(raw, dict) else []:
        away_team = game.get("away_team")
        home_team = game.get("home_team")
        commence_time = game.get("commence_time", "N/A")
        if not away_team or not home_team:
            continue
 
        try:
            if parse_utc_iso_to_eastern(commence_time).strftime("%Y-%m-%d") != date_str:
                continue
        except Exception:
            pass
 
        bookmakers = game.get("bookmakers", [])
        away_odds = _select_representative_price(bookmakers, away_team)
        home_odds = _select_representative_price(bookmakers, home_team)
        if away_odds is None or home_odds is None:
            continue
 
        games.append({
            "away_team": away_team,
            "home_team": home_team,
            "away_odds": away_odds,
            "home_odds": home_odds,
            "game_time": commence_time,
        })
    return games

tab3, tab2, tab1 = st.tabs(["Bankroll Tracker", "Past Matchups", "Today's Matchup (offseason)"])

st.markdown(
    """
    <style>
    /* Gray out and disable the 3rd tab (Today's Matchup) during the offseason */
    button[data-baseweb="tab"]:nth-of-type(3) {
        opacity: 0.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# TAB 1: TODAY'S MATCHUP
# ==========================================
with tab1:
    slate_date = st.date_input(
        "Filter Slate Date (Min: March 25, 2026)",
        value=DEFAULT_SLATE_DATE,
        min_value=SEASON_START_DATE,
        max_value=max(today_date, DEFAULT_SLATE_DATE),
        key="slate_date",
    )
 
    if slate_date == today_date:
        slate_df = today_games_df
        st.write(f"### Live Slate ({slate_date.strftime('%Y-%m-%d')})")
    else:
        slate_df = get_game_data_dataframe(
            matrix_df, live_games=fetch_past_slate(slate_date.strftime("%Y-%m-%d"))
        )
        st.write(f"### Slate for {slate_date.strftime('%Y-%m-%d')}")
        st.caption("Odds are DraftKings prices from an evening snapshot, so a few games may be missing.")
 
    if not slate_df.is_empty():
        styled_df = style_custom_dataframe(slate_df)
        st.dataframe(styled_df, use_container_width=True)
        st.write("---")
        st.markdown("## Detailed Matchup Analysis")
 
        # Render one expanded write-up per game below the summary table
        for row in slate_df.iter_rows(named=True):
            away_team = row["Away Team"]
            home_team = row["Home Team"]
            game_time = row["Game Time (EST)"]
 
            away_win = row["My Personal Away Projection"]
            home_win = row["My Personal Home Projection"]
            away_fair = row["Fair Market Away Prob %"]
            home_fair = row["Fair Market Home Prob %"]
            bet_rec = row["RECOMMENDATION"]
 
            st.markdown(f"### **{away_team} at {home_team}** ({game_time})")
 
            if away_win is None or home_win is None:
                st.write("Model probabilities are not available for this matchup.")
            else:
                favored_team = away_team if away_win >= home_win else home_team
                favored_prob = max(away_win, home_win)
 
                st.markdown("#### **Model Projection**")
                st.markdown(f"* **Favored Winner:** {favored_team} ({fmt_pct(favored_prob)})")
                st.markdown(f"* **Full Breakdown:** {away_team} ({fmt_pct(away_win)}) vs. {home_team} ({fmt_pct(home_win)})")
 
                st.markdown("#### **Market Context**")
                st.markdown(f"* **Fair Implied Odds:** {away_team} {fmt_pct(away_fair)} | {home_team} {fmt_pct(home_fair)}")
 
                st.markdown("#### **Recommendation**")
                st.markdown(f"* **Action:** {bet_rec}")
 
            st.divider()
    else:
        st.info("No games with odds found for this date.")

# ==========================================
# TAB 2: PAST MATCHUPS
# ==========================================
with tab2:
    st.write("### Past Matchups & Recommended Bets")

    start_date, end_date = render_date_range_picker(
        "past_matchups", "Filter Past Matchups Range (Min: March 25, 2026)", today_date
    )

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    poisson_ledger = load_ledger("poisson")
    
    if not poisson_ledger.is_empty() and "game" in poisson_ledger.columns:
        poisson_ledger = attach_date_str(poisson_ledger)

        date_bounded_ledger = filter_ledger_by_date_range(poisson_ledger, start_str, end_str)

        # Only rows that look like a real "Away @ Home" game entry (defensive
        # filter in case any malformed rows ever end up in the ledger)
        filtered_ledger = date_bounded_ledger.filter(pl.col("game").str.contains(" @ "))

        tab2_p_bal = latest_balance(date_bounded_ledger, default=INITIAL_BANKROLL)
        
        st.markdown(f"# **${tab2_p_bal:,.2f}**")

        if not filtered_ledger.is_empty():
            away_teams_list, home_teams_list = [], []
            away_projs, home_projs = [], []
            recommendations_list = []
            game_dates_list = []
            game_scores_list = []
            bet_results_list = []

            # Build one display row per settled game, re-computing the model's
            # projection fresh (rather than reading it from the ledger, which
            # doesn't store it) so this table always reflects the CURRENT
            # model math even for older bets.
            for row in filtered_ledger.iter_rows(named=True):
                game_str = row.get("game", "")
                parts = game_str.split(" @ ")
                if len(parts) != 2:
                    continue

                away_t, home_t = parts[0], parts[1]
                a_prob, h_prob = estimate_matchup_probabilities(away_t, home_t, matrix_df)

                team_bet = get_team_abbrev(row.get("team_bet", ""))
                rec_label = f"BET ON {team_bet.upper()}" if team_bet else "NO CLEAR VALUE"

                game_date_fmt = row.get("date_str", "N/A")

                won = row.get("won")
                # $0.0 default matches the payout to highlight a missing/corrupt payout for a lost bet
                payout = safe_balance(row.get("payout"), default=0.0)
                score = row.get("score", "N/A")

                if won is True:
                    bet_res = f"WIN (+${payout:,.2f})"
                elif won is False:
                    bet_res = f"LOSS (-${WAGER_SIZE:,.2f})"
                else:
                    bet_res = "PENDING"

                away_teams_list.append(away_t)
                home_teams_list.append(home_t)
                away_projs.append(a_prob)
                home_projs.append(h_prob)
                game_dates_list.append(game_date_fmt)
                game_scores_list.append(score)
                bet_results_list.append(bet_res)
                recommendations_list.append(rec_label)

            if away_teams_list:
                past_display_df = pl.DataFrame({
                    "RECOMMENDATION": recommendations_list,
                    "Game Date": game_dates_list,
                    "Away Team": away_teams_list,
                    "Home Team": home_teams_list,
                    "Game Score": game_scores_list,
                    "Bet Result": bet_results_list,
                    "My Personal Away Projection": away_projs,
                    "My Personal Home Projection": home_projs,
                }).sort("Game Date", descending=True)

                st.write(f"#### Recorded Matchups ({start_str} to {end_str})")
                styled_past_df = style_custom_dataframe(past_display_df)
                st.dataframe(styled_past_df, use_container_width=True)
            else:
                st.info("No recorded matchups found for this date range.")
        else:
            st.info("No recorded matchups found within the selected date range.")
    else:
        st.info("No past games logged yet. Click 'Backfill Season' to sync historical games.")

# ==========================================
# TAB 3: BANKROLL TRACKER
# ==========================================
with tab3:
    st.markdown("## **Bankroll Tracker & Detailed Ledgers**")

    p_ledger_all = load_ledger("poisson")
    r_ledger_all = load_ledger("random")

    b_start_date, b_end_date = render_date_range_picker(
        "bankroll_tracker", "Filter Bankroll Date Range (Min: March 25, 2026)", today_date
    )

    b_start_str = b_start_date.strftime("%Y-%m-%d")
    b_end_str = b_end_date.strftime("%Y-%m-%d")

    p_ledger_all = attach_date_str(p_ledger_all)
    r_ledger_all = attach_date_str(r_ledger_all)

    # Filter each ledger down to the selected date range - everything below
    # (balance numbers, chart, detailed tables) reads from these filtered
    # versions, so it all stays consistent with whatever range is picked.
    p_ledger = filter_ledger_by_date_range(p_ledger_all, b_start_str, b_end_str)
    r_ledger = filter_ledger_by_date_range(r_ledger_all, b_start_str, b_end_str)

    # Headline balance numbers - always derived from the date-FILTERED
    # ledger (via latest_balance), so they match whatever end date is
    # selected above rather than always showing today's absolute latest.
    col1, col2 = st.columns(2)
    with col1:
        p_bal = latest_balance(p_ledger, default=INITIAL_BANKROLL)
        st.caption("Poisson Model Balance")
        st.markdown(f"# **${p_bal:,.2f}**")
    with col2:
        r_bal = latest_balance(r_ledger, default=INITIAL_BANKROLL)
        st.caption("Coin-Flip Model Balance")
        st.markdown(f"# **${r_bal:,.2f}**")

    if not p_ledger.is_empty() and "date_str" in p_ledger.columns:
        p_chart = build_balance_chart_series(p_ledger, "Poisson Model Balance")

        if not r_ledger.is_empty() and "date_str" in r_ledger.columns:
            r_chart = build_balance_chart_series(r_ledger, "Coin-Flip Model Balance")
            combined_chart = pl.concat([p_chart, r_chart]).sort("Date")
        else:
            combined_chart = p_chart.sort("Date")

        chart_data = combined_chart.to_pandas()

        # Base chart: shared x/y/color encoding for both model lines.
        # scale=alt.Scale(zero=False) lets the y-axis start near the actual
        # balance range instead of forcing it down to $0, so swings are
        # easier to read.
        base = alt.Chart(chart_data).encode(
            x=alt.X("Date:N", title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Balance:Q", title="Bankroll Balance ($)", scale=alt.Scale(zero=False), axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=["Poisson Model Balance", "Coin-Flip Model Balance"],
                    range=["#52c41a", "#ff4d4f"]  # green = Poisson, red = Coin-Flip
                ),
                legend=alt.Legend(title="Model", orient="right")
            )
        )

        # Draw the Coin-Flip line dashed and the Poisson line solid, so the
        # two are distinguishable even without relying on color alone.
        lines = base.mark_line(strokeWidth=2.5).encode(
            strokeDash=alt.condition(
                alt.datum.Model == "Coin-Flip Model Balance",
                alt.value([4, 4]),
                alt.value([0])
            )
        )

        st.altair_chart(lines.properties(height=380), use_container_width=True)

    st.write("---")
    st.markdown("## **Detailed Betting Ledgers**")
    st.info("**Note:** All wagers across both models are fixed at **$5,000** per recommended bet.")
    st.info("**Note:** All wagers moneyline odds are historical DraftKings prices for the matchup/date, not fabricated or default values.")

    ledger_tab1, ledger_tab2 = st.tabs(["Poisson Model Ledger", "Coin-Flip Model Ledger"])

    def format_ledger_df(raw_df):
        """
        Prepares a raw ledger DataFrame for display as a readable table:
        abbreviates team names, builds a human-friendly "result" column from
        the raw won/payout data if one doesn't already exist, and picks a
        clean subset/order of columns to show.

        Args:
            raw_df: A (typically date-filtered) ledger Polars DataFrame,
                as loaded/filtered earlier in this tab.

        Returns:
            A pandas DataFrame with columns limited to whichever of
            [timestamp, game, result, team_bet, odds_taken, current_balance]
            are actually present, ready to pass to st.dataframe(). Returns
            an empty pandas DataFrame if raw_df is empty.
        """
        if raw_df.is_empty():
            return raw_df.to_pandas()
        
        df_pd = raw_df.to_pandas()

        # Convert full team names to 3-letter abbreviations in the ledger UI tables
        if "game" in df_pd.columns:
            def abbrev_game(game_str):
                if " @ " in str(game_str):
                    a, h = game_str.split(" @ ")
                    return f"{get_team_abbrev(a)} @ {get_team_abbrev(h)}"
                return game_str
            df_pd["game"] = df_pd["game"].apply(abbrev_game)

        if "team_bet" in df_pd.columns:
            df_pd["team_bet"] = df_pd["team_bet"].apply(get_team_abbrev)

        # Build a human-readable "result" column if the ledger doesn't
        # already have one (older ledger schemas store won/payout separately)
        if "result" not in df_pd.columns and "won" in df_pd.columns:
            results = []
            for idx, row in df_pd.iterrows():
                if row.get("won") is True:
                    # $4545.45 default matches a standard -110 win on a $5,000
                    # wager, used only if payout is somehow missing for a win.
                    payout = safe_balance(row.get("payout"), default=4545.45)
                    results.append(f"Win (+${payout:.2f})")
                elif row.get("won") is False:
                    results.append(f"Loss (-${WAGER_SIZE:.2f})")
                else:
                    results.append("N/A")
            df_pd["result"] = results

        if "odds_taken" not in df_pd.columns and "odds" in df_pd.columns:
            df_pd["odds_taken"] = df_pd["odds"]

        display_cols = ["timestamp", "game", "result", "team_bet", "odds_taken", "current_balance"]
        final_cols = [c for c in display_cols if c in df_pd.columns]
        return df_pd[final_cols]

    with ledger_tab1:
        formatted_p_df = format_ledger_df(p_ledger)
        st.dataframe(formatted_p_df, use_container_width=True)

    with ledger_tab2:
        formatted_r_df = format_ledger_df(r_ledger)
        st.dataframe(formatted_r_df, use_container_width=True)
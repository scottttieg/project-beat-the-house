import random
import polars as pl
from ledger import get_current_balance, load_ledger


def generate_random_selections(games_df: pl.DataFrame) -> list:
    """
    Simulates coin-flip selections for today's slate.
    Returns list of team names selected (or None if no valid matchup).
    """
    random_selections = []
    for row in games_df.iter_rows(named=True):
        if row["Away Team"] == "N/A" or row["Home Team"] == "N/A":
            random_selections.append(None)
            continue
            
        flip = random.randint(0, 1)
        selected_team = row["Away Team"] if flip == 0 else row["Home Team"]
        random_selections.append(selected_team)
        
    return random_selections
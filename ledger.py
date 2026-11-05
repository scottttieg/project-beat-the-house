from datetime import datetime
import os
import polars as pl

# Configuration Constants
INITIAL_BANKROLL = 1000000.0  # Default starting bankroll ($1M)
POISSON_CSV = "poisson_ledger.csv"
RANDOM_CSV = "random_ledger.csv"

def get_csv_filename(model_type: str = "poisson") -> str:
    """Returns the corresponding CSV filename based on model type."""
    if model_type.lower() == "random":
        return RANDOM_CSV
    return POISSON_CSV

def load_ledger(model_type: str = "poisson") -> pl.DataFrame:
    """Loads the ledger CSV as a Polars DataFrame. Returns an empty DataFrame if non-existent."""
    filename = get_csv_filename(model_type)
    if not os.path.exists(filename):
        return pl.DataFrame()
    try:
        return pl.read_csv(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return pl.DataFrame()

def get_current_balance(model_type: str = "poisson") -> float:
    """
    Retrieves the most recent bankroll balance by loading the ledger, sorting by
    date (in case rows aren't already in chronological order), and selecting the
    current_balance of the latest-dated entry. If empty, defaults to $1,000,000.00.
    """
    df = load_ledger(model_type)
    if df.is_empty() or "current_balance" not in df.columns:
        return INITIAL_BANKROLL

    if "timestamp" in df.columns:
        sort_keys = [_parse_sort_key(t) for t in df["timestamp"].to_list()]
        df = df.with_columns(pl.Series("_sort_key", sort_keys)).sort("_sort_key")

    balances = df["current_balance"].to_list()
    if not balances or balances[-1] is None:
        return INITIAL_BANKROLL
        
    try:
        return float(balances[-1])
    except (ValueError, TypeError):
        return INITIAL_BANKROLL

def calculate_payout(amount: float, odds: float) -> float:
    """Calculates net profit/payout based on wager amount and American odds."""
    if odds > 0:
        return amount * (odds / 100.0)
    elif odds < 0:
        return amount * (100.0 / abs(odds))
    return 0.0

def _parse_sort_key(ts) -> datetime:
    """
    Parses a timestamp value into a sortable datetime, tolerant of both full
    ISO timestamps (e.g. '2026-08-01T18:10:00+00:00') and bare date strings
    (e.g. '2026-08-01'). Falls back to datetime.min for anything unparseable
    so bad/missing timestamps sort first rather than crashing the sort.
    """
    s = str(ts).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return datetime.min


def record_completed_bet(
    model_type: str,
    game_name: str,
    team_bet: str,
    amount: float,
    odds: float,
    won: bool,
    extra_data: dict = None
) -> float:
    """
    Records a settled bet to the specified model ledger and updates the bankroll balance.
    Uses the historical game commence_time if passed in extra_data, otherwise defaults to now().

    current_balance is recomputed as a chronological cumulative sum across the
    whole ledger on every write (rather than just appended onto whatever the
    last row happened to be) because games can arrive out of date order - e.g.
    the daily pipeline settling a batch that spans several days, or a backfill
    run filling in an earlier date after later ones were already recorded. Each
    row's payout/loss only depends on its own amount/odds/won, never on prior
    balance, so re-deriving the running total after sorting by date is always
    correct regardless of insertion order.
    """
    filename = get_csv_filename(model_type)

    payout = calculate_payout(amount, odds) if won else 0.0

    # Prefer game commence_time if provided in extra_data, else default to current timestamp
    timestamp = extra_data.get("timestamp") if extra_data and extra_data.get("timestamp") else datetime.now().isoformat()
    score = extra_data.get("score", "N/A") if extra_data else "N/A"

    new_row = {
        "timestamp": [timestamp],
        "game": [game_name],
        "team_bet": [team_bet],
        "amount": [amount],
        "odds_taken": [odds],
        "won": [won],
        "payout": [round(payout, 2)],
        "current_balance": [0.0],  # placeholder - recomputed chronologically below
        "score": [score]
    }

    new_df = pl.DataFrame(new_row)
    existing_df = load_ledger(model_type)

    if existing_df.is_empty():
        combined_df = new_df
    else:
        combined_df = pl.concat([existing_df, new_df], how="diagonal")

    sort_keys = [_parse_sort_key(t) for t in combined_df["timestamp"].to_list()]
    combined_df = combined_df.with_columns(pl.Series("_sort_key", sort_keys)).sort("_sort_key")

    running_balance = INITIAL_BANKROLL
    recomputed_balances = []
    for row in combined_df.iter_rows(named=True):
        if row.get("won"):
            running_balance += row.get("payout") or 0.0
        else:
            running_balance -= row.get("amount") or 0.0
        recomputed_balances.append(round(running_balance, 2))

    combined_df = combined_df.with_columns(
        pl.Series("current_balance", recomputed_balances)
    ).drop("_sort_key")

    combined_df.write_csv(filename)
    return recomputed_balances[-1] if recomputed_balances else INITIAL_BANKROLL
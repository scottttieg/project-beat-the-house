# Beat the House ⚾

A Streamlit app that tests whether a simple statistical model can beat MLB moneyline odds, and compares it against a random coin-flip control so you can see whether the model is actually smarter than chance.

> Educational project only. This is not betting advice.

## What it does

Two models "bet" on every settled MLB game, a flat **$5,000 per bet**, starting from a **$1,000,000 bankroll** each:

- **Poisson model:** estimates each team's expected runs against a specific opponent, then converts that into a win probability. It recommends a bet when its probability beats the market's by more than a set edge (3 percentage points).
- **Coin-flip model:** picks a side at random on the same games. It is the control group.

Comparing the two bankrolls over the season shows whether the Poisson model's picks are better than luck.

The edge threshold applies to the live recommendations in the Today's Matchup tab. When settling past games, the Poisson model takes whichever side it favors.

## How the Poisson model works

1. From team batting stats (runs per game) and pitching stats (ERA), compute each team's offense and defense relative to the league average.
2. Expected runs for each side = (team offense / league avg) x (opponent defense / league avg) x league avg runs per game.
3. Treat each team's runs as an independent Poisson variable. The difference between the two has mean `lambda_away - lambda_home` and variance `lambda_away + lambda_home`, and a normal approximation of that (Skellam) distribution gives the away team's win probability.
4. Convert the sportsbook's American odds into implied probabilities and remove the "vig" (the book's built-in margin), so both sides sum to 100%.
5. If the model's probability beats the de-vigged market probability by more than the edge threshold, recommend the bet.

## The app

- **Today's Matchup:** moneyline odds, the model's projection, the fair market probability, and a bet recommendation for each game. It shows the live slate during the season but in the offseason, it displays one sample day.
- **Past Matchups:** settled games by date range with results.
- **Bankroll Tracker:** balance over time for both models, plus detailed ledgers.

## Data sources

- Live and historical moneyline odds: [The Odds API](https://the-odds-api.com/) (historical odds are DraftKings prices; games without a price are skipped rather than given a made-up one).
- Final scores: MLB's free Stats API (no key needed).
- Team batting and pitching stats: CSV files in `data/`.

## Project layout

```
app.py                      Streamlit app: data pipeline, model, and UI
ledger.py                   Reads and writes the bet ledgers (Polars + CSV)
poisson_ledger.csv          Poisson model's settled bets and running balance
random_ledger.csv           Coin-flip model's settled bets and running balance
data/                       Team batting and pitching stats CSVs
requirements.txt            Python dependencies
```

`historical_cache/` is created automatically at runtime to cache scores and odds by date.

## Run it locally

1. Python 3.9 or newer is required.
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Get a free API key from [The Odds API](https://the-odds-api.com/) and paste it into the `API_KEY` line near the top of `app.py`.
   
4. Start the app from the project folder:
   ```
   streamlit run app.py
   ```
   It opens at http://localhost:8501.

On load, the app settles the last few days of completed games and appends them to the ledgers, so the CSV files change as you use it.

## Built with

Python, Streamlit, Polars, pandas, Altair

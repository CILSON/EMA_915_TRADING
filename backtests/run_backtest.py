import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.EMA_915 import EMATradingAlgorithm
import yfinance as yf
import pandas as pd

# -----------------------------
# Load configuration
# -----------------------------

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# Download 5-minute data
df = yf.download("ETH", start="2025-11-03", interval="5m")

# FIX: flatten MultiIndex columns
df.columns = df.columns.get_level_values(0)

# Reset index and normalize columns
df = df.reset_index()
df.columns = df.columns.str.lower()

# Initialize strategy
algo = EMATradingAlgorithm(
    initial_capital=config["initial_capital"],
    risk_per_trade=config["risk_per_trade"],
    atr_stop_multiplier=config["atr_stop_multiplier"],
    max_bars_in_trade=config["max_bars_in_trade"],
    min_atr_percentile=config["min_atr_percentile"],
    atr_percentile_lookback=config["atr_percentile_lookback"],
    min_stop_distance_pct=config["min_stop_distance_pct"],
)

# Run backtest
result = algo.backtest(df)
stats = algo.calculate_statistics(result)

# Print stats
for key, value in stats.items():
    print(f"{key:.<30} {value}")

# Convert stats dict to DataFrame
stats_df = pd.DataFrame(
    list(stats.items()),
    columns=["Metric", "Value"]
)

# Save to CSV
stats_df.to_csv("btc_5m_summary_stats.csv", index=False)

# Plot
algo.plot_results(result)

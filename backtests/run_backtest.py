import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.EMA_915 import EMATradingAlgorithm
import yfinance as yf

# Download 5-minute data
df = yf.download("ETH", start="2025-11-03", interval="5m")

# FIX: flatten MultiIndex columns
df.columns = df.columns.get_level_values(0)

# Reset index and normalize columns
df = df.reset_index()
df.columns = df.columns.str.lower()

# Initialize strategy
algo = EMATradingAlgorithm(
    initial_capital=10000,
    risk_per_trade=0.01,
    atr_stop_multiplier=2.0,
    max_bars_in_trade=78,          # 1 trading day for 5m bars
    min_atr_percentile=50,
    atr_percentile_lookback=100,
    min_stop_distance_pct=0.5
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

# EMA 9/15 Trading Strategy

A rule-based intraday trend-following strategy built around **EMA 9/15**, momentum, volatility filtering, ATR-based risk management, and next-bar execution.

The project is designed to evaluate a systematic trading strategy with explicit position sizing, stop-loss logic, transaction costs, time-based exits, and trade-level performance analysis.

> **Research project — not financial advice.** Backtest results are historical simulations and do not imply future performance.

---

## Strategy Overview

The strategy combines several components rather than relying on an EMA crossover alone:

```text
Market Data
    ↓
EMA 9 / EMA 15
    ↓
Normalized Price Slope
    ↓
ATR Volatility Filter
    ↓
Signal Generation
    ↓
Next-Bar Execution
    ↓
ATR Stop Loss
    ↓
Risk-Based Position Sizing
    ↓
Time Stop / Stop Loss
    ↓
Trade & Equity Analysis
```

### Entry Logic

The current implementation generates:

- **Long signals** when:
  - normalized price slope is positive
  - current price is above EMA 9
  - ATR percentile passes the configured volatility threshold

- **Short signals** when:
  - normalized price slope is negative
  - current price is below EMA 9
  - ATR percentile passes the configured volatility threshold

Signals are generated on one bar and executed at the **next bar's open**.

### Risk Management

Position size is calculated from:

```text
Risk Amount = Current Capital × Risk Per Trade

Position Size = Risk Amount / Price Distance to Stop
```

The strategy also applies:

- ATR-based stop distance
- Minimum stop-distance constraint
- Fixed-fractional risk
- Maximum holding period
- Entry and exit transaction fees

---

## Core Components

| Component | Implementation |
|---|---|
| Trend | EMA 9 / EMA 15 |
| Momentum | 3-bar normalized price slope |
| Volatility | ATR(14) |
| Volatility Filter | Rolling ATR percentile |
| Stop Loss | ATR × multiplier |
| Position Sizing | Fixed fractional risk |
| Execution | Next-bar open |
| Time Exit | Maximum bars in trade |
| Transaction Costs | Entry + exit fees |
| Position Types | Long + Short |
| Evaluation | Trade statistics + equity curve + drawdown + Sharpe |

The ATR percentile calculation uses only historical ATR observations before the current bar, avoiding look-ahead from the percentile calculation.

---

## Backtest Configuration

The strategy supports parameters including:

```yaml
initial_capital: 10000
risk_per_trade: 0.005
atr_stop_multiplier: 2.0
max_bars_in_trade: 78
min_atr_percentile: 30
atr_percentile_lookback: 100
min_stop_distance_pct: 0.5
fee_rate: 0.0005
```

These parameters control:

- Starting capital
- Risk allocated to each trade
- ATR stop distance
- Maximum holding period
- Minimum volatility regime
- ATR percentile lookback
- Minimum stop distance
- Transaction fee rate

---

## Performance Metrics

The backtester calculates:

- Total trades
- Winning / losing trades
- Win rate
- Average win / loss
- Largest win / loss
- Maximum consecutive wins / losses
- Profit factor
- Gross P&L
- Total fees
- Initial / final capital
- Strategy return
- Annualized Sharpe ratio
- Maximum drawdown
- Average bars held
- Stop-loss exits
- Time-stop exits

---

## Example Backtest Result

One recorded backtest produced the following result:

| Metric | Value |
|---|---:|
| Risk Per Trade | 1.00% |
| ATR Stop Multiplier | 2.0x |
| Minimum Stop Distance | 0.5% |
| Maximum Bars in Trade | 78 |
| Minimum ATR Percentile | 50th |
| ATR Percentile Lookback | 100 |
| Total Trades | 58 |
| Winning Trades | 19 |
| Losing Trades | 39 |
| Win Rate | 32.76% |
| Stop Loss Exits | 37 |
| Time Stop Exits | 21 |
| Average Bars Held | 39.6 |
| Average Win | $400.89 |
| Average Loss | -$131.01 |
| Largest Win | $1,074.42 |
| Largest Loss | -$166.41 |
| Max Consecutive Wins | 3 |
| Max Consecutive Losses | 8 |
| Profit Factor | 1.49 |
| Gross P&L | $2,507.38 |
| Fees | $1,141.11 |
| Initial Capital | $10,000.00 |
| Final Capital | $12,507.38 |
| Total P&L | $2,507.38 |
| Strategy Return | 25.07% |
| Max Drawdown | -12.36% |
| Annualized Sharpe Ratio | 8.65 |

### Interpretation

The recorded run produced a **32.76% win rate** but a substantially larger average winning trade than average losing trade:

```text
Average Win  ≈ $400.89
Average Loss ≈ -$131.01
Profit Factor = 1.49
```

This is consistent with a trend-following profile where profitability does not depend on having a high percentage of winning trades.

The reported Sharpe ratio is based on the annualization methodology implemented in the current backtester and should therefore be interpreted in the context of that methodology rather than as a universal risk-adjusted performance figure.

---

## Project Structure

```text
EMA_915_TRADING/
│
├── src/
│   └── EMA_915.py
│
├── backtests/
│   └── run_backtest.py
│
├── config.yaml
│
├── paper_trades_eth_5m.csv
│
└── README.md
```

---

## Usage

Clone the repository and install the required dependencies.

```bash
pip install pandas numpy matplotlib yfinance
```

Run the backtest:

```bash
python backtests/run_backtest.py
```

The runner:

1. Downloads 5-minute market data.
2. Normalizes the downloaded data.
3. Initializes the trading algorithm.
4. Runs the backtest.
5. Calculates performance statistics.
6. Saves trade/statistical output.
7. Displays the strategy plots.

---

## Visual Analysis

The backtester produces a multi-panel analysis containing:

1. Price action with EMA 9 / EMA 15
2. Entry and exit markers
3. Capital evolution
4. Strategy vs. buy-and-hold returns
5. Strategy equity curve
6. Normalized price slope / momentum

This allows the strategy to be evaluated from both **trade-level** and **portfolio-level** perspectives.

---

## Important Implementation Details

### Next-Bar Execution

Signals are generated on the current bar and stored as pending signals. Entries are executed at the following bar's open rather than at the signal-generation price.

### Historical ATR Percentile

The ATR volatility filter calculates the percentile using historical ATR values before the current bar. This is intended to prevent the percentile calculation from using future information.

### Risk-Based Position Sizing

Position size changes with account capital and stop distance. A wider stop therefore reduces position size for the same percentage risk allocation.

### Transaction Costs

Fees are applied on both entry and exit when a position is closed.

### Time Stops

Positions can be closed after reaching the configured maximum number of bars in the trade.

---

## Current Limitations / Future Work

Potential extensions to the strategy and framework include:

- Multi-strategy support
- Walk-forward optimization
- Out-of-sample evaluation
- Parameter sensitivity analysis
- More robust benchmark comparison
- Slippage modeling
- More realistic execution assumptions
- Portfolio-level risk constraints
- Additional market/timeframe testing
- Automated experiment tracking

---

## Disclaimer

This repository is a **research and backtesting project**.

Backtests are simulations based on historical data and assumptions about execution, fees, liquidity, and market behavior. Results can be affected by data quality, model assumptions, parameter selection, transaction costs, and other factors.

Past simulated performance does not guarantee future results.

---

## Author

**Siddharth Cilson**

IIT (BHU), Varanasi

Focus areas:

- Quantitative Finance
- Algorithmic Trading
- Machine Learning
- AI Systems

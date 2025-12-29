import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class EMATradingAlgorithm:
    def __init__(self, initial_capital=10000, risk_per_trade=0.01, atr_stop_multiplier=2.0, 
                 max_bars_in_trade=20, min_atr_percentile=30, atr_percentile_lookback=100,
                 min_stop_distance_pct=0.5):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.atr_stop_multiplier = atr_stop_multiplier
        self.max_bars_in_trade = max_bars_in_trade
        self.min_atr_percentile = min_atr_percentile
        self.atr_percentile_lookback = atr_percentile_lookback
        self.min_stop_distance_pct = min_stop_distance_pct
        self.position = None
        self.entry_price = 0
        self.stop_loss = 0
        self.position_size = 0
        self.entry_bar = 0
        self.pending_signal = None
        self.pending_stop = None
        self.pending_atr = None
        self.trades = []
        self.equity_curve = []
        
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_slope(self, data, lookback=3):
        """Calculate normalized price slope (momentum)"""
        price_change = data['close'].diff(lookback)
        normalized_slope = (price_change / data['close'].shift(lookback)) * 100
        return normalized_slope
    
    def calculate_rolling_atr_percentile(self, data, i):
        """Calculate ATR percentile using only historical data (no lookahead)"""
        if i < self.atr_percentile_lookback:
            return 0
        
        # Only use historical ATR values
        historical_atr = data['atr'].iloc[i - self.atr_percentile_lookback:i]
        current_atr = data['atr'].iloc[i]
        
        if pd.isna(current_atr) or len(historical_atr) == 0:
            return 0
        
        percentile = (historical_atr < current_atr).sum() / len(historical_atr) * 100
        return percentile
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on 1% risk"""
        risk_amount = self.capital * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        
        max_position_size = self.capital / entry_price
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def backtest(self, data):
        """Run the trading algorithm backtest"""
        data = data.copy()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if 'date' not in data.columns:
            if isinstance(data.index, pd.DatetimeIndex):
                data['date'] = data.index
            else:
                data['date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
        
        data = data.reset_index(drop=True)
        
        data['ema9'] = self.calculate_ema(data, 9)
        data['ema15'] = self.calculate_ema(data, 15)
        data['slope'] = self.calculate_slope(data, 3)
        data['atr'] = self.calculate_atr(data, 14)
        
        data['buy_hold_returns'] = (data['close'] / data['close'].iloc[0] - 1) * 100
        
        data['signal'] = 0
        data['position'] = None
        data['strategy_equity'] = self.initial_capital
        data['capital'] = self.initial_capital
        
        start_bar = max(20, self.atr_percentile_lookback)
        
        for i in range(start_bar, len(data)):
            current_price = data.iloc[i]['close']
            current_open = data.iloc[i]['open']
            current_high = data.iloc[i]['high']
            current_low = data.iloc[i]['low']
            ema9 = data.iloc[i]['ema9']
            ema15 = data.iloc[i]['ema15']
            slope = data.iloc[i]['slope']
            atr = data.iloc[i]['atr']
            
            # Rolling ATR percentile (no lookahead)
            atr_percentile = self.calculate_rolling_atr_percentile(data, i)
            atr_filter_pass = atr_percentile >= self.min_atr_percentile
            
            # Execute pending signal at next bar open
            if self.pending_signal is not None and self.position is None:
                entry_price = current_open
                stop_loss = self.pending_stop
                atr_value = self.pending_atr
                
                self.open_position(self.pending_signal, entry_price, stop_loss, i, 
                                 data.iloc[i]['date'], atr_value)
                data.at[data.index[i], 'signal'] = 1 if self.pending_signal == 'long' else -2
                
                self.pending_signal = None
                self.pending_stop = None
                self.pending_atr = None
            
            bars_in_trade = i - self.entry_bar if self.position is not None else 0
            
            # Time stop
            if self.position is not None and bars_in_trade >= self.max_bars_in_trade:
                self.close_position(current_price, i, data.iloc[i]['date'], 
                                  exit_reason='time_stop')
                data.at[data.index[i], 'signal'] = -3 if self.position == 'long' else 3
                
            # Intrabar stop loss using high/low
            elif self.position == 'long' and current_low <= self.stop_loss:
                exit_price = self.stop_loss
                self.close_position(exit_price, i, data.iloc[i]['date'], 
                                  exit_reason='stop_loss')
                data.at[data.index[i], 'signal'] = -1
                
            elif self.position == 'short' and current_high >= self.stop_loss:
                exit_price = self.stop_loss
                self.close_position(exit_price, i, data.iloc[i]['date'], 
                                  exit_reason='stop_loss')
                data.at[data.index[i], 'signal'] = 2
            
            # Generate signals for next-bar execution
            elif self.position is None and self.pending_signal is None and atr_filter_pass and i < len(data) - 1:
                if slope > 0 and current_price > ema9:
                    # Enforce minimum stop distance for long
                    atr_stop_distance = atr * self.atr_stop_multiplier
                    min_stop_distance = current_price * (self.min_stop_distance_pct / 100)
                    stop_distance = max(atr_stop_distance, min_stop_distance)
                    
                    proposed_stop = current_price - stop_distance
                    
                    self.pending_signal = 'long'
                    self.pending_stop = proposed_stop
                    self.pending_atr = atr
                    
                elif slope < 0 and current_price < ema9:
                    # Enforce minimum stop distance for short
                    atr_stop_distance = atr * self.atr_stop_multiplier
                    min_stop_distance = current_price * (self.min_stop_distance_pct / 100)
                    stop_distance = max(atr_stop_distance, min_stop_distance)
                    
                    proposed_stop = current_price + stop_distance
                    
                    self.pending_signal = 'short'
                    self.pending_stop = proposed_stop
                    self.pending_atr = atr
            
            # Update equity curve
            if self.position == 'long':
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
                current_equity = self.capital + unrealized_pnl
            elif self.position == 'short':
                unrealized_pnl = (self.entry_price - current_price) * self.position_size
                current_equity = self.capital + unrealized_pnl
            else:
                current_equity = self.capital
            
            data.at[data.index[i], 'strategy_equity'] = current_equity
            data.at[data.index[i], 'capital'] = self.capital
            data.at[data.index[i], 'position'] = self.position
        
        data['strategy_returns'] = (data['strategy_equity'] / self.initial_capital - 1) * 100
        
        return data
    
    def open_position(self, position_type, price, stop_loss, index, date, atr):
        """Open a trading position with 1% risk management"""
        self.position = position_type
        self.entry_price = price
        self.stop_loss = stop_loss
        self.entry_bar = index
        
        self.position_size = self.calculate_position_size(price, stop_loss)
        
        position_value = self.position_size * price
        risk_amount = self.capital * self.risk_per_trade
        stop_distance = abs(price - stop_loss)
        stop_distance_pct = (stop_distance / price) * 100
        
        print(f"\n{date.strftime('%Y-%m-%d')} - Opening {position_type.upper()} position (NEXT-BAR ENTRY)")
        print(f"  Entry Price: ${price:.2f} (at open)")
        print(f"  ATR: ${atr:.2f}")
        print(f"  Stop Loss: ${stop_loss:.2f}")
        print(f"  Stop Distance: ${stop_distance:.2f} ({stop_distance_pct:.2f}%)")
        print(f"  Min Stop Distance Enforced: {self.min_stop_distance_pct}%")
        print(f"  Current Capital: ${self.capital:.2f}")
        print(f"  Risk Amount (1%): ${risk_amount:.2f}")
        print(f"  Position Size: {self.position_size:.4f} units")
        print(f"  Position Value: ${position_value:.2f}")
    
    def close_position(self, price, index, date, exit_reason='normal'):
        """Close the current trading position and update capital"""
        if self.position == 'long':
            pnl = (price - self.entry_price) * self.position_size
            pnl_pct = (price / self.entry_price - 1) * 100
        else:
            pnl = (self.entry_price - price) * self.position_size
            pnl_pct = (1 - price / self.entry_price) * 100
        
        bars_in_trade = index - self.entry_bar
        
        old_capital = self.capital
        self.capital += pnl
        
        trade = {
            'type': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'position_size': self.position_size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'capital_before': old_capital,
            'capital_after': self.capital,
            'exit_reason': exit_reason,
            'bars_held': bars_in_trade,
            'date': date
        }
        self.trades.append(trade)
        
        exit_type_map = {
            'stop_loss': 'STOPPED OUT',
            'time_stop': 'TIME STOP',
            'normal': 'CLOSED'
        }
        exit_type = exit_type_map.get(exit_reason, 'CLOSED')
        
        print(f"\n{date.strftime('%Y-%m-%d')} - {exit_type} {self.position.upper()} position")
        print(f"  Exit Price: ${price:.2f}")
        print(f"  Bars Held: {bars_in_trade}")
        print(f"  Position Size: {self.position_size:.4f} units")
        print(f"  P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        print(f"  Capital Before: ${old_capital:.2f}")
        print(f"  Capital After: ${self.capital:.2f}")
        
        self.position = None
        self.entry_price = 0
        self.stop_loss = 0
        self.position_size = 0
        self.entry_bar = 0
    
    def calculate_statistics(self, data):
        """Calculate trading statistics and benchmark comparison"""
        if not self.trades:
            return None
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        largest_win = trades_df['pnl'].max()
        largest_loss = trades_df['pnl'].min()
        
        stop_loss_exits = len(trades_df[trades_df['exit_reason'] == 'stop_loss'])
        time_stop_exits = len(trades_df[trades_df['exit_reason'] == 'time_stop'])
        normal_exits = len(trades_df[trades_df['exit_reason'] == 'normal'])
        
        avg_bars_held = trades_df['bars_held'].mean() if len(trades_df) > 0 else 0
        
        strategy_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        buy_hold_return = data['buy_hold_returns'].iloc[-1]
        
        equity_returns = data['strategy_equity'].pct_change().dropna()
        
        if len(equity_returns) > 0 and equity_returns.std() > 0:
            avg_daily_return = equity_returns.mean()
            std_daily_return = equity_returns.std()
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        max_equity = data['strategy_equity'].expanding().max()
        drawdown = (data['strategy_equity'] - max_equity) / max_equity * 100
        max_drawdown = drawdown.min()
        
        trades_df['win'] = trades_df['pnl'] > 0
        trades_df['streak'] = (trades_df['win'] != trades_df['win'].shift()).cumsum()
        win_streaks = trades_df[trades_df['win']].groupby('streak').size()
        loss_streaks = trades_df[~trades_df['win']].groupby('streak').size()
        max_consecutive_wins = win_streaks.max() if len(win_streaks) > 0 else 0
        max_consecutive_losses = loss_streaks.max() if len(loss_streaks) > 0 else 0
        
        stats = {
            'Risk Per Trade': f"{self.risk_per_trade * 100:.2f}%",
            'ATR Stop Multiplier': f"{self.atr_stop_multiplier}x",
            'Min Stop Distance': f"{self.min_stop_distance_pct}%",
            'Max Bars in Trade': self.max_bars_in_trade,
            'Min ATR Percentile': f"{self.min_atr_percentile}th",
            'ATR Percentile Lookback': self.atr_percentile_lookback,
            '': '',
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': f"{win_rate:.2f}%",
            ' ': '',
            'Stop Loss Exits': stop_loss_exits,
            'Time Stop Exits': time_stop_exits,
            'Normal Exits': normal_exits,
            'Avg Bars Held': f"{avg_bars_held:.1f}",
            '  ': '',
            'Average Win': f"${avg_win:.2f}",
            'Average Loss': f"${avg_loss:.2f}",
            'Largest Win': f"${largest_win:.2f}",
            'Largest Loss': f"${largest_loss:.2f}",
            'Max Consecutive Wins': int(max_consecutive_wins),
            'Max Consecutive Losses': int(max_consecutive_losses),
            'Profit Factor': f"{abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 'N/A'}",
            '   ': '',
            'Initial Capital': f"${self.initial_capital:.2f}",
            'Final Capital': f"${self.capital:.2f}",
            'Total P&L': f"${total_pnl:.2f}",
            '    ': '',
            'Strategy Return': f"{strategy_return:.2f}%",
            'Buy & Hold Return': f"{buy_hold_return:.2f}%",
            'Alpha (vs Buy & Hold)': f"{(strategy_return - buy_hold_return):.2f}%",
            '     ': '',
            'Sharpe Ratio (Annualized)': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2f}%"
        }
        
        return stats
    
    def plot_results(self, data):
        """Plot the trading results with benchmark comparison"""
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(5, 1, hspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(data['date'], data['close'], label='Price', linewidth=2, color='blue')
        ax1.plot(data['date'], data['ema9'], label='EMA 9', linewidth=1.5, color='green', alpha=0.7)
        ax1.plot(data['date'], data['ema15'], label='EMA 15', linewidth=1.5, color='orange', alpha=0.7)
        
        buy_signals = data[data['signal'] == 1]
        sell_signals = data[data['signal'] == -1]
        short_signals = data[data['signal'] == -2]
        cover_signals = data[data['signal'] == 2]
        
        ax1.scatter(buy_signals['date'], buy_signals['close'], color='green', marker='^', 
                   s=100, label='Long Entry', zorder=5)
        ax1.scatter(sell_signals['date'], sell_signals['close'], color='red', marker='v', 
                   s=100, label='Long Exit', zorder=5)
        ax1.scatter(short_signals['date'], short_signals['close'], color='red', marker='v', 
                   s=100, label='Short Entry', zorder=5, alpha=0.7)
        ax1.scatter(cover_signals['date'], cover_signals['close'], color='green', marker='^', 
                   s=100, label='Short Exit', zorder=5, alpha=0.7)
        
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title('EMA 9/15 Trading Algorithm - Price Action (1% Risk Per Trade)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', ncol=3)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(data['date'], data['capital'], label='Available Capital', 
                linewidth=2, color='purple')
        ax2.axhline(y=self.initial_capital, color='black', linestyle='--', 
                   linewidth=1, alpha=0.5, label='Initial Capital')
        ax2.set_ylabel('Capital ($)', fontsize=12)
        ax2.set_title('Capital Evolution (Updates After Each Trade)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(data['date'], data['strategy_returns'], label='Strategy Returns', 
                linewidth=2, color='green')
        ax3.plot(data['date'], data['buy_hold_returns'], label='Buy & Hold Returns', 
                linewidth=2, color='blue', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.fill_between(data['date'], data['strategy_returns'], data['buy_hold_returns'],
                        where=(data['strategy_returns'] >= data['buy_hold_returns']),
                        color='green', alpha=0.2, label='Outperformance')
        ax3.fill_between(data['date'], data['strategy_returns'], data['buy_hold_returns'],
                        where=(data['strategy_returns'] < data['buy_hold_returns']),
                        color='red', alpha=0.2, label='Underperformance')
        
        ax3.set_ylabel('Returns (%)', fontsize=12)
        ax3.set_title('Strategy vs Buy & Hold Returns', fontsize=14, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(data['date'], data['strategy_equity'], label='Strategy Equity', 
                linewidth=2, color='green')
        ax4.axhline(y=self.initial_capital, color='black', linestyle='--', 
                   linewidth=1, alpha=0.5, label='Initial Capital')
        ax4.set_ylabel('Equity ($)', fontsize=12)
        ax4.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[4, 0])
        ax5.plot(data['date'], data['slope'], label='Price Slope', linewidth=1.5, color='purple')
        ax5.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax5.fill_between(data['date'], data['slope'], 0, where=(data['slope'] > 0), 
                        color='green', alpha=0.3, label='Positive Momentum')
        ax5.fill_between(data['date'], data['slope'], 0, where=(data['slope'] < 0), 
                        color='red', alpha=0.3, label='Negative Momentum')
        
        ax5.set_xlabel('Date', fontsize=12)
        ax5.set_ylabel('Slope (%)', fontsize=12)
        ax5.set_title('Normalized Price Slope (Momentum Indicator)', fontsize=14, fontweight='bold')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("EMA 9/15 ALGORITHMIC TRADING STRATEGY WITH RISK MANAGEMENT")
    print("=" * 70)
    print("\nStrategy Rules:")
    print("  LONG:  Entry when slope > 0 AND close > EMA9 | Exit at ATR-based stop")
    print("  SHORT: Entry when slope < 0 AND close < EMA9 | Exit at ATR-based stop")
    print("  RISK:  1% of capital per trade with position sizing")
    print("  STOP:  ATR-based (default 2x ATR) + Time stop (max 20 bars)")
    print("        + Minimum stop distance enforcement")
    print("  FILTER: Rolling ATR percentile (no lookahead)")
    print("  EXECUTION: Next-bar entry at open")
    print("=" * 70)
    print()
    
    print("USAGE EXAMPLE:")
    print("-" * 70)
    print("import yfinance as yf")
    print("df = yf.download('SPY', start='2023-01-01', end='2024-01-01')")
    print()
    print("algo = EMATradingAlgorithm(")
    print("    initial_capital=10000,")
    print("    risk_per_trade=0.01,")
    print("    atr_stop_multiplier=2.0,")
    print("    max_bars_in_trade=20,")
    print("    min_atr_percentile=30,")
    print("    atr_percentile_lookback=100,")
    print("    min_stop_distance_pct=0.5")
    print(")")
    print("result = algo.backtest(df)")
    print("stats = algo.calculate_statistics(result)")
    print("for key, value in stats.items():")
    print("    print(f'{key:.<30} {value}')")
    print("algo.plot_results(result)")
    print("-" * 70)
    print()
    print("NOTE: Columns should be lowercase.")
    print("If not: df.columns = df.columns.str.lower()")
    print("=" * 70)
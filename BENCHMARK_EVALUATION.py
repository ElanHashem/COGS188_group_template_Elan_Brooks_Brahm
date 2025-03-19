import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("stock_details_5_years.csv")

# Convert Date to datetime format and sort
df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values(["Company", "Date"])

# Compute daily returns
df["Return"] = df.groupby("Company")["Close"].pct_change()

# Drop NA values from returns (first row per company will be NaN)
df = df.dropna(subset=["Return"])

### 1️⃣ Simple Moving Average (SMA) ###
def compute_SMA_signals(df, short_window=5, long_window=100):
    df = df.copy()
    df["SMA_Short"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=short_window, min_periods=1).mean())
    df["SMA_Long"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=long_window, min_periods=1).mean())

    df["SMA_Signal"] = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "SMA_Signal"] = 1  
    df.loc[df["SMA_Short"] < df["SMA_Long"], "SMA_Signal"] = -1  

    return df

df = compute_SMA_signals(df)
df["SMA_Strategy_Return"] = df["Return"] * df["SMA_Signal"].shift(1)
df = df.dropna(subset=["SMA_Strategy_Return"])

### 2️⃣ Exponential Moving Average (EMA) ###
def compute_EMA_signals(df, short_span=5, long_span=20):
    df = df.copy()
    df["EMA_Short"] = df.groupby("Company")["Close"].transform(lambda x: x.ewm(span=short_span, adjust=False).mean())
    df["EMA_Long"] = df.groupby("Company")["Close"].transform(lambda x: x.ewm(span=long_span, adjust=False).mean())

    df["EMA_Signal"] = 0
    df.loc[df["EMA_Short"] > df["EMA_Long"], "EMA_Signal"] = 1  
    df.loc[df["EMA_Short"] < df["EMA_Long"], "EMA_Signal"] = -1  

    return df

df = compute_EMA_signals(df)
df["EMA_Strategy_Return"] = df["Return"] * df["EMA_Signal"].shift(1)
df = df.dropna(subset=["EMA_Strategy_Return"])

### 3️⃣ Mean Reversion Strategy ###
def compute_mean_reversion_signals(df, window=50, threshold=0.05):
    df = df.copy()
    df["SMA"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    df["Deviation"] = (df["Close"] - df["SMA"]) / df["SMA"]

    df["MeanReversion_Signal"] = 0
    df.loc[df["Deviation"] < -threshold, "MeanReversion_Signal"] = 1  
    df.loc[df["Deviation"] > threshold, "MeanReversion_Signal"] = -1  

    df["MeanReversion_Strategy_Return"] = df["Return"] * df["MeanReversion_Signal"].shift(1)
    return df.dropna(subset=["MeanReversion_Strategy_Return"])

df = compute_mean_reversion_signals(df)

### 4️⃣ Momentum Strategy ###
def compute_momentum_signals(df, lookback=10):
    df = df.copy()
    df["Momentum"] = df.groupby("Company")["Return"].transform(lambda x: x.rolling(window=lookback, min_periods=1).sum())

    df["Momentum_Signal"] = 0
    df.loc[df["Momentum"] > 0, "Momentum_Signal"] = 1  
    df.loc[df["Momentum"] < 0, "Momentum_Signal"] = -1  

    df["Momentum_Strategy_Return"] = df["Return"] * df["Momentum_Signal"].shift(1)
    return df.dropna(subset=["Momentum_Strategy_Return"])

df = compute_momentum_signals(df)

### 5️⃣ Moving Average Crossover with Momentum (MAC) ###
def compute_mac_signals(df, short_window=10, long_window=50, momentum_window=5):
    df = df.copy()
    df["SMA_Short"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=short_window, min_periods=1).mean())
    df["SMA_Long"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=long_window, min_periods=1).mean())
    df["Momentum"] = df.groupby("Company")["Return"].transform(lambda x: x.rolling(window=momentum_window, min_periods=1).sum())

    df["MAC_Signal"] = 0
    df.loc[(df["SMA_Short"] > df["SMA_Long"]) & (df["Momentum"] > 0), "MAC_Signal"] = 1  
    df.loc[(df["SMA_Short"] < df["SMA_Long"]) & (df["Momentum"] < 0), "MAC_Signal"] = -1  

    df["MAC_Strategy_Return"] = df["Return"] * df["MAC_Signal"].shift(1)
    return df.dropna(subset=["MAC_Strategy_Return"])

df = compute_mac_signals(df)

### 6️⃣ Scalping Strategy ###
def compute_scalping_signals(df, threshold=0.002):
    df = df.copy()
    df["Scalping_Signal"] = 0

    df.loc[df["Return"] > threshold, "Scalping_Signal"] = 1  # Buy signal
    df.loc[df["Return"] < -threshold, "Scalping_Signal"] = -1  # Sell signal

    df["Scalping_Strategy_Return"] = df["Return"] * df["Scalping_Signal"].shift(1)  # Apply yesterday’s signal
    return df.dropna(subset=["Scalping_Strategy_Return"])

df = compute_scalping_signals(df)

### 7️⃣ Swing Trading Strategy ###
def compute_swing_trading_signals(df, short_window=10, long_window=50):
    df = df.copy()
    df["SMA_Short"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=short_window, min_periods=1).mean())
    df["SMA_Long"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=long_window, min_periods=1).mean())

    df["Swing_Signal"] = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Swing_Signal"] = 1  # Buy signal
    df.loc[df["SMA_Short"] < df["SMA_Long"], "Swing_Signal"] = -1  # Sell signal

    df["Swing_Strategy_Return"] = df["Return"] * df["Swing_Signal"].shift(1)  # Apply yesterday’s signal
    return df.dropna(subset=["Swing_Strategy_Return"])

df = compute_swing_trading_signals(df)

### 8️⃣ Buy-and-Hold Strategy ###
def compute_buy_and_hold_returns(df):
    df = df.copy()
    first_close = df.groupby("Company")["Close"].transform("first")
    df["Buy_Hold_Return"] = (df["Close"] - first_close) / first_close
    return df

df = compute_buy_and_hold_returns(df)

### 📊 Sharpe Ratio Calculation ###
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate  
    std_dev = np.std(excess_returns, ddof=1)  

    return np.inf if std_dev == 0 else np.mean(excess_returns) / std_dev

# Compute Sharpe Ratios
sma_sharpe = calculate_sharpe_ratio(df["SMA_Strategy_Return"])
ema_sharpe = calculate_sharpe_ratio(df["EMA_Strategy_Return"])
mean_reversion_sharpe = calculate_sharpe_ratio(df["MeanReversion_Strategy_Return"])
momentum_sharpe = calculate_sharpe_ratio(df["Momentum_Strategy_Return"])
mac_sharpe = calculate_sharpe_ratio(df["MAC_Strategy_Return"])
scalping_sharpe = calculate_sharpe_ratio(df["Scalping_Strategy_Return"].dropna())
swing_sharpe = calculate_sharpe_ratio(df["Swing_Strategy_Return"].dropna())
buy_hold_sharpe = calculate_sharpe_ratio(df["Buy_Hold_Return"].dropna())

### 📝 Display Results ###
print("\n📈 **Sharpe Ratios for Trading Strategies** 📉")
print(f"🔵 SMA Strategy Sharpe Ratio: {sma_sharpe:.4f}")
print(f"🟢 EMA Strategy Sharpe Ratio: {ema_sharpe:.4f}")
print(f"🔴 Mean Reversion Strategy Sharpe Ratio: {mean_reversion_sharpe:.4f}")
print(f"🟡 Momentum Strategy Sharpe Ratio: {momentum_sharpe:.4f}")
print(f"🔷 MAC Strategy Sharpe Ratio: {mac_sharpe:.4f}")
print(f"⚪ Scalping Strategy Sharpe Ratio: {scalping_sharpe:.4f}")
print(f"🔶 Swing Trading Strategy Sharpe Ratio: {swing_sharpe:.4f}")
print(f"⚫ Buy-and-Hold Strategy Sharpe Ratio: {buy_hold_sharpe:.4f}")
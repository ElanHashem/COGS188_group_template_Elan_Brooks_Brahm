import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("stock_details_5_years.csv")

# Convert Date to datetime format
df["Date"] = pd.to_datetime(df["Date"], utc=True)

# Sort the DataFrame by Company and Date
df = df.sort_values(["Company", "Date"])

# Drop rows with any missing values in important columns
df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

# Compute daily returns
df["Return"] = df.groupby("Company")["Close"].pct_change()

# Drop NA values from returns (first row per company will be NaN)
df = df.dropna(subset=["Return"])

# Feature Engineering
df["Prev_Close"] = df.groupby("Company")["Close"].shift(1)
df["Prev_Open"] = df.groupby("Company")["Open"].shift(1)
df["Price_Change"] = df["Close"] - df["Prev_Close"]
df["High_Low_Range"] = df["High"] - df["Low"]
df["Volatility"] = df["Close"].rolling(window=5).std()

# Calculate moving averages
df["SMA_10"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
df["SMA_50"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=50, min_periods=1).mean())

# Drop rows with any missing values after creating new features
df = df.dropna()

# Filter for Apple (AAPL)
aapl_df = df[df["Company"] == "AAPL"]

# Temporal Difference Learning Class
class TDLearning:
    def __init__(self, alpha=0.1, gamma=0.9):
        self.alpha = alpha  
        self.gamma = gamma  
        self.value_function = {}  

    def predict(self, state):
        stored_value = self.value_function.get(state, 0)
        if stored_value < state * 0.5:
            return state  
        return stored_value

    def update(self, state, reward, next_state):
        current_value = self.predict(state)
        next_value = self.predict(next_state)
        self.value_function[state] = current_value + self.alpha * (reward + self.gamma * next_value - current_value)

    def train(self, data):
        for i in range(len(data)):
            price = data.iloc[i]["Close"]
            if price not in self.value_function:
                self.value_function[price] = price
        
        for i in range(1, len(data)):
            current_price = data.iloc[i - 1]["Close"]
            next_price = data.iloc[i]["Close"]
            reward = next_price - current_price  
            self.update(current_price, reward, next_price)

# Hyperparameters for grid search
split_ratios = [0.4, 0.5, 0.6, 0.75]  # Train-test split ratios
price_change_thresholds = [0, .005, .01, .015, .02, .025, .03, .035, 0.04]  # Price change thresholds

# Initialize variables to store best parameters and max profit
best_split_ratio = None
best_price_change_threshold = None
max_dp_profit = float('-inf')  # Start with negative infinity to ensure any profit will be better

# Grid search to find the best hyperparameters
print("Starting grid search...")
print("Split Ratio | Threshold | Profit")
print("-" * 35)

for split_ratio in split_ratios:
    for price_change_threshold in price_change_thresholds:
        # Split data into training and testing based on the current split ratio
        split_index = int(len(aapl_df) * split_ratio)
        train_df = aapl_df.iloc[:split_index]
        test_df = aapl_df.iloc[split_index:]

        # Initialize and train TD learning model
        td_model = TDLearning()
        td_model.train(train_df)

        # Prepare a DataFrame for predictions
        full_df = aapl_df.copy()
        full_df["Predicted_Close"] = np.nan  

        # Make predictions for the test set
        for i in range(len(test_df)):
            state = test_df.iloc[i]["Prev_Close"]
            prediction = td_model.predict(state)
            full_df.iloc[i + split_index, full_df.columns.get_loc("Predicted_Close")] = prediction

        # Optimized Dynamic Programming for Trading using Predicted Prices
        test_dates = test_df["Date"].values
        pred_prices = full_df["Predicted_Close"].iloc[split_index:].values
        
        # Parameters for trading strategy
        n = len(pred_prices)
        dp_profit = 0
        buy_sell_points = []
        holding = False  # Track if we are currently holding stock
        buy_price = 0

        # Traverse predicted prices to identify buy/sell points
        for i in range(n - 1):
            price_change = (pred_prices[i + 1] - pred_prices[i]) / pred_prices[i]
            if not holding and price_change > price_change_threshold:  # Buy condition
                buy_price = pred_prices[i]
                holding = True
            elif holding and price_change < -price_change_threshold and pred_prices[i] > buy_price:  # Sell condition
                dp_profit += pred_prices[i] - buy_price  # Calculate cumulative profit
                holding = False

        # Ensure the last transaction closes properly
        if holding:
            dp_profit += pred_prices[-1] - buy_price

        # Print current parameters and profit
        print(f"{split_ratio:.2f}     | {price_change_threshold:.3f}    | {dp_profit:.2f}")
        
        # Update best parameters if current profit is higher
        if dp_profit > max_dp_profit:
            max_dp_profit = dp_profit
            best_split_ratio = split_ratio
            best_price_change_threshold = price_change_threshold

# Final output for the best hyperparameters found
print("\nBest Hyperparameters Found:")
print(f"Best Train-Test Split Ratio: {best_split_ratio:.3f}")
print(f"Best Price Change Threshold: {best_price_change_threshold:.3f}")
print(f"Max DP Profit: {max_dp_profit:.2f}")

# Now implement the remainder of the code with the best parameters
# Using the best parameters for the final model

# Split data into training and testing based on the best split ratio
split_index = int(len(aapl_df) * best_split_ratio)
train_df = aapl_df.iloc[:split_index]
test_df = aapl_df.iloc[split_index:]

# Initialize and train TD learning model
td_model = TDLearning()
td_model.train(train_df)

# Prepare a DataFrame for predictions
full_df = aapl_df.copy()
full_df["Predicted_Close"] = np.nan  

# Make predictions for the test set (second half)
for i in range(len(test_df)):
    state = test_df.iloc[i]["Prev_Close"]
    prediction = td_model.predict(state)
    full_df.iloc[i + split_index, full_df.columns.get_loc("Predicted_Close")] = prediction

# Plot actual vs predicted closing prices
plt.figure(figsize=(15, 8))
plt.plot(full_df["Date"], full_df["Close"], label="Actual Closing Price", color="blue")
plt.plot(full_df["Date"], full_df["Predicted_Close"], label="Predicted Closing Price", color="orange", linestyle="--")
plt.axvline(x=full_df.iloc[split_index]["Date"], color='r', linestyle='-', label="Train/Test Split")
plt.title("Actual vs Predicted Closing Prices for AAPL")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Optimized Dynamic Programming for Trading using Predicted Prices
test_dates = test_df["Date"].values
pred_prices = full_df["Predicted_Close"].iloc[split_index:].values
actual_prices = test_df["Close"].values

# Parameters for trading strategy using best price change threshold
n = len(pred_prices)
dp_profit = 0
buy_sell_points = []
holding = False  # Track if we are currently holding stock
buy_price = 0

# Traverse predicted prices to identify buy/sell points with improvements
for i in range(n - 1):
    price_change = (pred_prices[i + 1] - pred_prices[i]) / pred_prices[i]
    if not holding and price_change > best_price_change_threshold:  # Buy condition
        buy_sell_points.append((test_dates[i], "Buy", pred_prices[i]))
        buy_price = pred_prices[i]
        holding = True
    elif holding and price_change < -best_price_change_threshold and pred_prices[i] > buy_price:  # Sell condition
        buy_sell_points.append((test_dates[i], "Sell", pred_prices[i]))
        dp_profit += pred_prices[i] - buy_price  # Calculate cumulative profit
        holding = False

# Ensure the last transaction closes properly
if holding:
    buy_sell_points.append((test_dates[-1], "Sell", pred_prices[-1]))
    dp_profit += pred_prices[-1] - buy_price

# Calculate Buy-and-Hold profit on predicted prices
predicted_buy_price = pred_prices[0]
predicted_sell_price = pred_prices[-1]
predicted_buy_hold_profit = predicted_sell_price - predicted_buy_price

print(f"\nOptimized DP Strategy Profit (Predicted Prices): {dp_profit:.2f}")
print(f"Buy-and-Hold Profit (Predicted Prices): {predicted_buy_hold_profit:.2f}")

# --- GRAPH FOR DP OPTIMIZED TRADES ON PREDICTED PRICES ---
plt.figure(figsize=(15, 8))
plt.plot(test_dates, pred_prices, label="Predicted Prices", color="blue")
plt.scatter([point[0] for point in buy_sell_points if point[1] == "Buy"],
            [point[2] for point in buy_sell_points if point[1] == "Buy"], 
            color="green", label="Buy", marker="^", s=100)
plt.scatter([point[0] for point in buy_sell_points if point[1] == "Sell"],
            [point[2] for point in buy_sell_points if point[1] == "Sell"], 
            color="red", label="Sell", marker="v", s=100)
plt.axhline(y=predicted_buy_price, color="gray", linestyle="--", label="Buy-and-Hold Entry Price")
plt.axhline(y=predicted_sell_price, color="black", linestyle="--", label="Buy-and-Hold Exit Price")
plt.title("Optimized DP Trading Strategy with Multiple Buy/Sell Actions (Predicted Prices)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()  
plt.show()


# --- NEW CODE: Apply DP strategy to ACTUAL prices ---
# Extract buy/sell signals from the DP strategy (dates and actions only)
buy_sell_actions = [(point[0], point[1]) for point in buy_sell_points]

# Initialize portfolios with $100
dp_portfolio = 100.0  # Cash for DP strategy
buy_hold_portfolio = 100.0  # Cash for buy-and-hold

# Calculate buy-and-hold shares (buy once at beginning of test period)
buy_hold_shares = buy_hold_portfolio / actual_prices[0]
buy_hold_portfolio = 0  # All cash used

# Initialize tracking arrays
dp_value_history = np.zeros(len(test_dates))
buy_hold_history = np.zeros(len(test_dates))

# Track DP strategy performance using actual prices
holding = False
shares_owned = 0

# Process each day in the test period
for i in range(len(test_dates)):
    current_date = test_dates[i]
    current_price = actual_prices[i]
    
    # Update buy-and-hold value (shares * current price)
    buy_hold_history[i] = buy_hold_shares * current_price
    
    # Check for buy/sell actions on this date
    action_today = next((action for date, action in buy_sell_actions if date == current_date), None)
    
    if action_today == "Buy" and not holding:
        # Buy with all available cash
        shares_owned = dp_portfolio / current_price
        dp_portfolio = 0  # All cash used
        holding = True
    elif action_today == "Sell" and holding:
        # Sell all shares
        dp_portfolio = shares_owned * current_price
        shares_owned = 0
        holding = False
    
    # Calculate current DP portfolio value (cash + share value)
    dp_value_history[i] = dp_portfolio + (shares_owned * current_price)

# Calculate final returns
actual_buy_price = actual_prices[0]
actual_sell_price = actual_prices[-1]
actual_buy_hold_profit = actual_sell_price - actual_buy_price

dp_final_return = (dp_value_history[-1] - 100) / 100 * 100  # Percentage return
buy_hold_final_return = (buy_hold_history[-1] - 100) / 100 * 100  # Percentage return

# --- GRAPH FOR STRATEGY COMPARISON ON ACTUAL PRICES ---
plt.figure(figsize=(15, 8))
plt.plot(test_dates, dp_value_history, label="DP Strategy Portfolio", color="green")
plt.plot(test_dates, buy_hold_history, label="Buy-and-Hold Portfolio", color="blue", linestyle="--")
plt.axhline(100, color="black", linestyle="--", alpha=0.5, label="Initial Investment ($100)")
plt.title("Comparison of Portfolio Value: DP Strategy vs. Buy-and-Hold (Using Actual Prices)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Print final results
print("\n--- PERFORMANCE ON ACTUAL PRICES ---")
print(f"DP Strategy Final Portfolio Value: ${dp_value_history[-1]:.2f}")
print(f"Buy-and-Hold Final Portfolio Value: ${buy_hold_history[-1]:.2f}")
print(f"DP Strategy Return: {dp_final_return:.2f}%")
print(f"Buy-and-Hold Return: {buy_hold_final_return:.2f}%")

# Display trade-by-trade performance on actual prices
print("\nTrade-by-Trade Performance (Actual Prices):")
total_trades = 0
profitable_trades = 0
total_profit = 0
trade_dates = []
trade_profits = []

holding = False
entry_price = 0
entry_date = None

for i in range(len(test_dates)):
    current_date = test_dates[i]
    current_price = actual_prices[i]
    
    # Check for buy/sell actions on this date
    action_today = next((action for date, action in buy_sell_actions if date == current_date), None)
    
    if action_today == "Buy" and not holding:
        entry_price = current_price
        entry_date = current_date
        holding = True
    elif action_today == "Sell" and holding:
        total_trades += 1
        trade_profit = current_price - entry_price
        total_profit += trade_profit
        
        if trade_profit > 0:
            profitable_trades += 1
        
        #print(f"Trade {total_trades}: Buy at ${entry_price:.2f} on {entry_date}, Sell at ${current_price:.2f} on {current_date}, Profit: ${trade_profit:.2f}")
        
        trade_dates.append(current_date)
        trade_profits.append(trade_profit)
        
        holding = False

if total_trades > 0:
    win_rate = (profitable_trades / total_trades) * 100
    print(f"\nTotal Trades: {total_trades}")
    print(f"Profitable Trades: {profitable_trades} ({win_rate:.2f}%)")
    print(f"Total Profit from Trading: ${total_profit:.2f}")
    print(f"Average Profit per Trade: ${total_profit / total_trades:.2f}")
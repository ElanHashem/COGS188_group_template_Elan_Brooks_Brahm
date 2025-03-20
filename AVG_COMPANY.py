import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import kagglehub
import os

path = kagglehub.dataset_download("iveeaten3223times/massive-yahoo-finance-dataset")


csv_file = os.path.join(path, "stock_details_5_years.csv")
df = pd.read_csv(csv_file)
# Load dataset
#df = pd.read_csv("stock_details_5_years.csv")

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

### Temporal Difference Learning Class
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

# Get unique companies
companies = df["Company"].unique()

# Initialize dictionaries to store results
prediction_accuracy = {}
portfolio_performance = {}

# Parameters
split_ratio = 0.5  # Train/test split ratio #########HYPERPARAMTER 1#############
price_change_threshold = 0.015  # 1.5% threshold for trading #########HYPERPARAMTER 2#############
initial_investment = 100.0  # Starting with $100

# Create figures for plots
fig_accuracy = plt.figure(figsize=(15, 10))
fig_returns = plt.figure(figsize=(15, 10))

# Create dictionaries to store results by date
all_actual_prices = {}
all_predicted_prices = {}
all_dp_values = {}
all_buyhold_values = {}
company_weights = {}

# Process each company
for company in companies:
    print(f"Processing {company}...")
    
    # Filter data for the current company
    company_df = df[df["Company"] == company]
    
    # Skip if we have too few data points
    if len(company_df) < 100:
        print(f"Skipping {company} - insufficient data points")
        continue
    
    # Split data into training and testing
    split_index = int(len(company_df) * split_ratio)
    train_df = company_df.iloc[:split_index]
    test_df = company_df.iloc[split_index:]
    
    # Initialize and train TD learning model
    td_model = TDLearning()
    td_model.train(train_df)
    
    # Prepare a DataFrame for predictions
    full_df = company_df.copy()
    full_df["Predicted_Close"] = np.nan
    
    # Make predictions for the test set
    for i in range(len(test_df)):
        state = test_df.iloc[i]["Prev_Close"]
        prediction = td_model.predict(state)
        full_df.iloc[i + split_index, full_df.columns.get_loc("Predicted_Close")] = prediction
    
    # Extract test data
    test_dates = test_df["Date"].values
    pred_prices = full_df["Predicted_Close"].iloc[split_index:].values
    actual_prices = test_df["Close"].values
    
    # Calculate company weight based on market cap or volume
    # Here we use total trading volume as a simple proxy for importance
    company_weight = test_df["Volume"].sum()
    company_weights[company] = company_weight
    
    # Normalize starting prices to 1.0
    norm_actual = actual_prices / actual_prices[0]
    norm_pred = pred_prices / pred_prices[0]
    
    # Store normalized prices by date
    for i, date in enumerate(test_dates):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        
        if date_str not in all_actual_prices:
            all_actual_prices[date_str] = []
            all_predicted_prices[date_str] = []
            all_dp_values[date_str] = []
            all_buyhold_values[date_str] = []
        
        all_actual_prices[date_str].append((norm_actual[i], company_weight))
        all_predicted_prices[date_str].append((norm_pred[i], company_weight))
    
    # --- Apply DP strategy to actual prices ---
    # Parameters for trading strategy
    n = len(pred_prices)
    buy_sell_points = []
    holding = False
    buy_price = 0
    
    # Generate buy/sell signals based on predicted prices
    for i in range(n - 1):
        price_change = (pred_prices[i + 1] - pred_prices[i]) / pred_prices[i]
        if not holding and price_change > price_change_threshold:  # Buy condition
            buy_sell_points.append((test_dates[i], "Buy"))
            buy_price = pred_prices[i]
            holding = True
        elif holding and price_change < -price_change_threshold and pred_prices[i] > buy_price:  # Sell condition
            buy_sell_points.append((test_dates[i], "Sell"))
            holding = False
    
    # Ensure the last transaction closes properly
    if holding:
        buy_sell_points.append((test_dates[-1], "Sell"))
    
    # Extract buy/sell signals (dates and actions only)
    buy_sell_actions = [(point[0], point[1]) for point in buy_sell_points]
    
    # Initialize portfolios
    dp_portfolio = initial_investment  # Cash for DP strategy
    buy_hold_portfolio = initial_investment  # Cash for buy-and-hold
    
    # Calculate buy-and-hold shares
    buy_hold_shares = buy_hold_portfolio / actual_prices[0]
    buy_hold_portfolio = 0  # All cash used
    
    # Initialize tracking arrays
    dp_value_history = np.zeros(len(test_dates))
    buy_hold_history = np.zeros(len(test_dates))
    
    # Track DP strategy performance
    holding = False
    shares_owned = 0
    
    # Process each day in the test period
    for i in range(len(test_dates)):
        current_date = test_dates[i]
        current_price = actual_prices[i]
        
        # Update buy-and-hold value
        buy_hold_history[i] = buy_hold_shares * current_price
        
        # Check for buy/sell actions
        action_today = next((action for date, action in buy_sell_actions if date == current_date), None)
        
        if action_today == "Buy" and not holding:
            shares_owned = dp_portfolio / current_price
            dp_portfolio = 0
            holding = True
        elif action_today == "Sell" and holding:
            dp_portfolio = shares_owned * current_price
            shares_owned = 0
            holding = False
        
        # Calculate current DP portfolio value
        dp_value_history[i] = dp_portfolio + (shares_owned * current_price)
    
    # Normalize portfolio values
    norm_dp_values = dp_value_history / initial_investment
    norm_buyhold_values = buy_hold_history / initial_investment
    
    # Store normalized portfolio values by date
    for i, date in enumerate(test_dates):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        all_dp_values[date_str].append((norm_dp_values[i], company_weight))
        all_buyhold_values[date_str].append((norm_buyhold_values[i], company_weight))
    
    # Calculate final returns
    dp_final_return = (dp_value_history[-1] - initial_investment) / initial_investment * 100
    buy_hold_final_return = (buy_hold_history[-1] - initial_investment) / initial_investment * 100
    
    # Store results for this company
    prediction_accuracy[company] = np.mean(np.abs(norm_actual - norm_pred))
    portfolio_performance[company] = {
        "DP_Return": dp_final_return,
        "BuyHold_Return": buy_hold_final_return
    }
    
    print(f"{company} - Prediction Error: {prediction_accuracy[company]:.4f}")
    print(f"{company} - DP Return: {dp_final_return:.2f}%, Buy-Hold Return: {buy_hold_final_return:.2f}%")
    print("-" * 50)

# Calculate weighted averages by date
common_dates = sorted(all_actual_prices.keys())
weighted_avg_actual = []
weighted_avg_predicted = []
weighted_avg_dp = []
weighted_avg_buyhold = []
plot_dates = []

for date_str in common_dates:
    if (all_actual_prices[date_str] and all_predicted_prices[date_str] and 
        all_dp_values[date_str] and all_buyhold_values[date_str]):
        
        # Calculate weighted average for this date
        total_weight_actual = sum(weight for _, weight in all_actual_prices[date_str])
        total_weight_pred = sum(weight for _, weight in all_predicted_prices[date_str])
        total_weight_dp = sum(weight for _, weight in all_dp_values[date_str])
        total_weight_buyhold = sum(weight for _, weight in all_buyhold_values[date_str])
        
        if total_weight_actual > 0 and total_weight_pred > 0 and total_weight_dp > 0 and total_weight_buyhold > 0:
            w_actual = sum(val * weight for val, weight in all_actual_prices[date_str]) / total_weight_actual
            w_pred = sum(val * weight for val, weight in all_predicted_prices[date_str]) / total_weight_pred
            w_dp = sum(val * weight for val, weight in all_dp_values[date_str]) / total_weight_dp
            w_buyhold = sum(val * weight for val, weight in all_buyhold_values[date_str]) / total_weight_buyhold
            
            weighted_avg_actual.append(w_actual)
            weighted_avg_predicted.append(w_pred)
            weighted_avg_dp.append(w_dp)
            weighted_avg_buyhold.append(w_buyhold)
            plot_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))

# ----- PLOT 1: Average Prediction Accuracy -----
plt.figure(fig_accuracy.number)
plt.plot(plot_dates, weighted_avg_actual, label="Weighted Avg Actual Price", color="blue")
plt.plot(plot_dates, weighted_avg_predicted, label="Weighted Avg Predicted Price", color="orange", linestyle="--")
plt.title("Weighted Average Actual vs Predicted Normalized Prices Across All Companies")
plt.xlabel("Date")
plt.ylabel("Normalized Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ----- PLOT 2: Average Portfolio Performance -----
plt.figure(fig_returns.number)
plt.plot(plot_dates, weighted_avg_dp, label="Weighted Avg DP Strategy Returns", color="green")
plt.plot(plot_dates, weighted_avg_buyhold, label="Weighted Avg Buy-and-Hold Returns", color="blue", linestyle="--")
plt.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Initial Investment (Normalized)")
plt.title("Weighted Average Portfolio Performance: DP Strategy vs. Buy-and-Hold")
plt.xlabel("Date")
plt.ylabel("Normalized Portfolio Value")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Calculate ending performance
final_dp_return = (weighted_avg_dp[-1] - 1.0) * 100 if weighted_avg_dp else 0
final_buyhold_return = (weighted_avg_buyhold[-1] - 1.0) * 100 if weighted_avg_buyhold else 0

# Summary statistics
print("\n----- SUMMARY STATISTICS -----")
print(f"Number of companies analyzed: {len(prediction_accuracy)}")

# Average prediction accuracy
avg_prediction_error = sum(prediction_accuracy.values()) / len(prediction_accuracy) if prediction_accuracy else 0
print(f"Average prediction error: {avg_prediction_error:.4f}")

# Average returns
avg_dp_return = sum(p["DP_Return"] for p in portfolio_performance.values()) / len(portfolio_performance) if portfolio_performance else 0
avg_buyhold_return = sum(p["BuyHold_Return"] for p in portfolio_performance.values()) / len(portfolio_performance) if portfolio_performance else 0
print(f"Average DP Strategy Return: {avg_dp_return:.2f}%")
print(f"Average Buy-and-Hold Return: {avg_buyhold_return:.2f}%")

# Weighted average returns (from portfolio performance)
if portfolio_performance and sum(company_weights[company] for company in portfolio_performance) > 0:
    weighted_dp_return = sum(p["DP_Return"] * company_weights[company] for company, p in portfolio_performance.items()) / sum(company_weights[company] for company in portfolio_performance)
    weighted_buyhold_return = sum(p["BuyHold_Return"] * company_weights[company] for company, p in portfolio_performance.items()) / sum(company_weights[company] for company in portfolio_performance)
    print(f"Weighted Average DP Strategy Return: {weighted_dp_return:.2f}%")
    print(f"Weighted Average Buy-and-Hold Return: {weighted_buyhold_return:.2f}%")

# Final portfolio performance from time series
print(f"\nFinal Performance (from time series):")
print(f"Weighted Avg DP Strategy Final Return: {final_dp_return:.2f}%")
print(f"Weighted Avg Buy-and-Hold Final Return: {final_buyhold_return:.2f}%")

# Display the plots
plt.show(
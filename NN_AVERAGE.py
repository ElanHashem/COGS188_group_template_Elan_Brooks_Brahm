import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Load dataset
path = kagglehub.dataset_download("iveeaten3223times/massive-yahoo-finance-dataset")
csv_file = os.path.join(path, "stock_details_5_years.csv")
df = pd.read_csv(csv_file)

# Convert Date to datetime format
df["Date"] = pd.to_datetime(df["Date"], utc=True)

# Sort and clean data
df = df.sort_values(["Company", "Date"])
df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
df["Return"] = df.groupby("Company")["Close"].pct_change()
df = df.dropna(subset=["Return"])

# Feature Engineering
df["Prev_Close"] = df.groupby("Company")["Close"].shift(1)
df["Price_Change"] = df["Close"] - df["Prev_Close"]
df["High_Low_Range"] = df["High"] - df["Low"]
df["Volatility"] = df["Close"].rolling(window=5).std()
df["SMA_10"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
df["SMA_50"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
df = df.dropna()

# Get unique companies
companies = df["Company"].unique()

# Initialize dictionaries to store results
prediction_accuracy = {}
portfolio_performance = {}
company_weights = {}

# Parameters
split_ratio = 0.7  # Train/test split ratio
initial_investment = 100.0  # Starting investment

# Dictionaries to store weighted average returns
weighted_avg_nn = {}
weighted_avg_buyhold = {}

# Define PyTorch neural network class
class TradingNN(nn.Module):
    def __init__(self, input_size):
        super(TradingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Process each company
for company in companies:
    print(f"Processing {company}...")
    company_df = df[df["Company"] == company]
    if len(company_df) < 100:
        continue
    
    split_index = int(len(company_df) * split_ratio)
    train_df = company_df.iloc[:split_index]
    test_df = company_df.iloc[split_index:]
    
    features = ["Prev_Close", "Price_Change", "High_Low_Range", "Volatility", "SMA_10", "SMA_50"]
    X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
    X_test = torch.tensor(test_df[features].values, dtype=torch.float32)
    y_train = torch.tensor((train_df["Close"].pct_change() > 0).astype(int).values[1:], dtype=torch.long)
    
    model = TradingNN(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train[:-1])
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    y_pred = model(X_test[:-1]).detach().numpy()
    predicted_actions = np.argmax(y_pred, axis=1)
    test_df = test_df.iloc[1:]
    test_df["NN_Trade_Action"] = predicted_actions
    
    buy_hold_shares = initial_investment / test_df["Close"].iloc[0]
    buy_hold_values = test_df["Close"] * buy_hold_shares / initial_investment
    
    nn_portfolio = initial_investment
    holding = False
    shares_owned = 0
    nn_values = []
    
    for i in range(len(test_df)):
        if predicted_actions[i] == 1 and not holding:
            shares_owned = nn_portfolio / test_df["Close"].iloc[i]
            nn_portfolio = 0
            holding = True
        elif predicted_actions[i] == 0 and holding:
            nn_portfolio = shares_owned * test_df["Close"].iloc[i]
            shares_owned = 0
            holding = False
        nn_values.append((nn_portfolio + shares_owned * test_df["Close"].iloc[i]) / initial_investment)
    
    company_weights[company] = test_df["Volume"].sum()
    for i, date in enumerate(test_df["Date"]):
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in weighted_avg_nn:
            weighted_avg_nn[date_str] = []
            weighted_avg_buyhold[date_str] = []
        weighted_avg_nn[date_str].append((nn_values[i], company_weights[company]))
        weighted_avg_buyhold[date_str].append((buy_hold_values.iloc[i], company_weights[company]))

# Calculate weighted averages
common_dates = sorted(weighted_avg_nn.keys())
nn_avg = []
buyhold_avg = []
date_list = []

for date_str in common_dates:
    total_weight_nn = sum(weight for _, weight in weighted_avg_nn[date_str])
    total_weight_buyhold = sum(weight for _, weight in weighted_avg_buyhold[date_str])
    
    if total_weight_nn > 0 and total_weight_buyhold > 0:
        w_nn = sum(val * weight for val, weight in weighted_avg_nn[date_str]) / total_weight_nn
        w_buyhold = sum(val * weight for val, weight in weighted_avg_buyhold[date_str]) / total_weight_buyhold
        nn_avg.append(w_nn)
        buyhold_avg.append(w_buyhold)
        date_list.append(datetime.strptime(date_str, '%Y-%m-%d'))

# Plot results
plt.figure(figsize=(15, 8))
plt.plot(date_list, nn_avg, label="Weighted Avg NN Strategy Returns", color="green")
plt.plot(date_list, buyhold_avg, label="Weighted Avg Buy-and-Hold Returns", linestyle="--", color="blue")
plt.axhline(y=1, color='gray', linestyle='--', label="Initial Investment (Normalized)")
plt.axvline(x=date_list[len(date_list) // 2], color='red', linestyle='-', label="Train/Test Split")
plt.title("Weighted Average Portfolio Performance: NN Strategy vs. Buy-and-Hold")
plt.xlabel("Date")
plt.ylabel("Normalized Portfolio Value")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

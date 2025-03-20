import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import os
import torch
import torch.nn as nn
import torch.optim as optim

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

# Filter for Apple (AAPL)
aapl_df = df[df["Company"] == "AAPL"]
split_index = int(len(aapl_df) * 0.5)
train_df = aapl_df.iloc[:split_index]
test_df = aapl_df.iloc[split_index:]

# Convert data to PyTorch tensors
features = ["Prev_Close", "Price_Change", "High_Low_Range", "Volatility", "SMA_10", "SMA_50"]
X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
X_test = torch.tensor(test_df[features].values, dtype=torch.float32)

y_train = torch.tensor((train_df["Close"].pct_change() > 0).astype(int).values[1:], dtype=torch.long)
y_test = torch.tensor((test_df["Close"].pct_change() > 0).astype(int).values[1:], dtype=torch.long)

# Define PyTorch neural network class
class TradingNN(nn.Module):
    def __init__(self):
        super(TradingNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
model = TradingNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train[:-1])  # Avoid last NaN row
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Make predictions
y_pred = model(X_test[:-1]).detach().numpy()
predicted_actions = np.argmax(y_pred, axis=1)
test_df = test_df.iloc[1:]  # Align indices
test_df["NN_Trade_Action"] = predicted_actions

# Plot weighted average portfolio performance with training data included
plt.figure(figsize=(15, 8))
plt.plot(aapl_df["Date"], aapl_df["Close"] / aapl_df["Close"].iloc[0],
         label="Weighted Avg NN Strategy Returns", color="green")
plt.plot(aapl_df["Date"], aapl_df["Close"] / aapl_df["Close"].iloc[0],
         label="Weighted Avg Buy-and-Hold Returns", linestyle="--", color="blue")
plt.axhline(y=1, color='gray', linestyle='--', label="Initial Investment (Normalized)")
plt.axvline(x=aapl_df.iloc[split_index]["Date"], color='red', linestyle='-', label="Train/Test Split")
plt.title("Weighted Average Portfolio Performance: NN Strategy vs. Buy-and-Hold")
plt.xlabel("Date")
plt.ylabel("Normalized Portfolio Value")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

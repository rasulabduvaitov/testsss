import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Generate mock stock price data
data = {
    'date': pd.date_range(start='1/1/2020', periods=100),
    'close_price': [100 + i + (i % 10) for i in range(100)]  # Simulated stock prices
}

df = pd.DataFrame(data)
df['days'] = (df['date'] - df['date'].min()).dt.days  # Convert date to number of days

# Features and target variable
X = df[['days']]  # The number of days since the first date
y = df['close_price']  # Simulated stock prices

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a pickle file
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully to 'linear_model.pkl'")

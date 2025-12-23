import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta


model = load_model('G:\projects1\StockPredictionModel\Stock Preictions Model.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'HDFCBANK.NS')

ticker = yf.Ticker(stock)

# Get the earliest available trading date (acts as listing date)
hist = ticker.history(period="max")
earliest_date = hist.index.min().date()

# Get current date
current_date = datetime.now().date()

# Calculate 10 years ago from current date
ten_years_ago = current_date - timedelta(days=365 * 10)

# Choose start date (later of the two: listing date or 10 years ago)
start_date = max(earliest_date, ten_years_ago)

# Download final data from start_date to current date
data = yf.download(
    stock,
    start=start_date,
    end=current_date
)

st.subheader('Stock Data')
st.write(data)

# Check if there's enough data for predictions (minimum 100 days required)
if len(data) < 100:
    st.error(f" Insufficient data for predictions. This stock has only {len(data)} days of data, but the model requires at least 100 days of historical data.")
    st.info("Please try a different stock symbol with more historical data.")
    st.stop()

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(14,9))
plt.plot(ma_50_days, 'r', label='MA 50 Days')
plt.plot(data.Close, 'g', label='Close Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(14, 9))
plt.plot(ma_50_days, 'r', label='MA 50 Days')
plt.plot(ma_100_days, 'b', label='MA 100 Days')
plt.plot(data.Close, 'g', label='Close Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(14,9))
plt.plot(ma_100_days, 'r', label='MA 100 Days')
plt.plot(ma_200_days, 'b', label='MA 200 Days')
plt.plot(data.Close, 'g', label='Close Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig3)

x=[]
y=[]
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
    
x,y = np.array(x), np.array(y)

predict = model.predict(x)
scale = 1/scaler.scale_

predict = predict * scale

y=y*scale


future_days = 30  
last_100_days = data_test_scale[-100:]  # Get last 100 days of scaled data

future_predictions = []
for _ in range(future_days):
    input_data = last_100_days.reshape(1, 100, 1)
    next_pred = model.predict(input_data, verbose=0)
    future_predictions.append(next_pred[0, 0])
    # Update the last 100 days by removing first and adding predicted value
    last_100_days = np.append(last_100_days[1:], next_pred[0, 0])
    last_100_days = last_100_days.reshape(-1, 1)

# Scale back future predictions
future_predictions = np.array(future_predictions) * scale

st.subheader('Original Price vs Predicted Price')
# Get the corresponding dates for the test data
test_dates = data.index[int(len(data)*0.80):]

# Generate future dates
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

# Combine predictions
all_predictions = np.concatenate([predict.flatten(), future_predictions])
all_dates = test_dates.union(future_dates)

fig4 = plt.figure(figsize=(14,9))
plt.plot(test_dates, y, 'g', label='Original Price', linewidth=2)
plt.plot(all_dates, all_predictions, 'r', label='Predicted Price', linewidth=2)
plt.axvline(x=last_date, color='gray', linestyle='--', linewidth=1, label='Current Date')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
st.pyplot(fig4)
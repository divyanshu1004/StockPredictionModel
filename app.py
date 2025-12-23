import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta
import requests


model = load_model('Stock Preictions Model.keras')

st.header('Stock Market Predictor')

@st.cache_data(ttl=300)
def search_yahoo_finance(query):
    """Search Yahoo Finance for stock symbols and company names"""
    if not query or len(query) < 1:
        return []
    
    try:
        # Yahoo Finance autocomplete API
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            'q': query,
            'quotesCount': 10,
            'newsCount': 0,
            'enableFuzzyQuery': False,
            'quotesQueryId': 'tss_match_phrase_query'
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            if 'quotes' in data:
                for quote in data['quotes']:
                    symbol = quote.get('symbol', '')
                    name = quote.get('longname') or quote.get('shortname', '')
                    exchange = quote.get('exchDisp', '')
                    quote_type = quote.get('quoteType', '')
                    
                    # Filter for stocks and ETFs
                    if quote_type in ['EQUITY', 'ETF', 'MUTUALFUND']:
                        display_text = f"{name} ({symbol})"
                        if exchange:
                            display_text += f" - {exchange}"
                        results.append({
                            'display': display_text,
                            'symbol': symbol,
                            'name': name
                        })
            
            return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []
    
    return []

# Search Interface
st.subheader('🔍 Search Stock')

col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input(
        'Type company name or symbol',
        '',
        placeholder='e.g., Apple, Tesla, Microsoft, HDFC Bank...',
        label_visibility='collapsed'
    )

with col2:
    search_button = st.button('🔎 Search', use_container_width=True)

# Initialize stock variable
stock = None

# Perform search when button clicked or query entered
if search_query and (search_button or search_query):
    with st.spinner('Searching Yahoo Finance...'):
        results = search_yahoo_finance(search_query)
    
    if results:
        st.success(f'Found {len(results)} results')
        
        # Display results
        stock_options = [r['display'] for r in results]
        
        selected_display = st.selectbox(
            'Select a stock:',
            options=stock_options,
            key='stock_select'
        )
        
        # Get the selected symbol
        selected_index = stock_options.index(selected_display)
        stock = results[selected_index]['symbol']
        
        st.info(f'✅ Selected: **{selected_display}**')
        
    else:
        st.warning('⚠️ No results found. Try a different search term or enter symbol directly below.')
        stock = st.text_input('Enter stock symbol manually:', 'AAPL')
else:
    # Default stock if no search performed
    st.info('👆 Search for any stock or enter a symbol below')
    stock = st.text_input('Or enter stock symbol directly:', 'HDFCBANK.NS')

# Proceed only if stock is selected
if not stock:
    st.warning('⚠️ Please select or enter a stock symbol to continue')
    st.stop()

st.divider()

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

# Properly inverse transform to get actual prices
predict = scaler.inverse_transform(predict)
y = scaler.inverse_transform(y.reshape(-1, 1))


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

# Properly inverse transform future predictions to get actual prices
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

st.subheader('Original Price vs Predicted Price')
# Get the corresponding dates for the test data
test_dates = data.index[int(len(data)*0.80):]

# Generate future dates
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

# Combine predictions
all_predictions = np.concatenate([predict.flatten(), future_predictions.flatten()])
all_dates = test_dates.union(future_dates)

fig4 = plt.figure(figsize=(14,9))
plt.plot(test_dates, y.flatten(), 'g', label='Original Price', linewidth=2)
plt.plot(all_dates, all_predictions, 'r', label='Predicted Price', linewidth=2)
plt.axvline(x=last_date, color='gray', linestyle='--', linewidth=1, label='Current Date')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.title(f'{stock} - Stock Price Prediction', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
# Format y-axis to show prices with comma separators
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
plt.tight_layout()
plt.show()
st.pyplot(fig4)
import pandas as pd
import numpy as np

data = pd.read_csv('data/stock_data_2020-01-01.csv')
data = data.sort_values(by=['Symbol', 'Date'])

## Add 5 day percent change
data['Weekly %Change'] = data.groupby('Symbol')['Close'].pct_change(periods=5) * 100

## Feature Engineering
# 5-day and 10-day moving averages
data['5-day MA'] = data.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5).mean())
data['10-day MA'] = data.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=10).mean())

# Daily returns Stocks
data['Daily Return'] = data.groupby('Symbol')['Close'].pct_change()

# Daily returns SP500
data.set_index('Date', inplace=True)
sp500_daily_return = data[data['Symbol'] == '^GSPC']['Daily Return']
data['SP500 Daily Return'] = data.index.map(sp500_daily_return)
data.reset_index(inplace=True)

# Daily outpreformance of SP500
data['Daily Outperformed SP500'] = (data['Daily Return'] > data['SP500 Daily Return']).astype(int)

# Rolling 5 day outpreformance of SP500
data['Rolling 5-day Outperform'] = data.groupby('Symbol')['Daily Outperformed SP500'].rolling(window=5).sum().reset_index(level=0, drop=True)

# Price momentum (5 days)
data['5-day Momentum'] = data.groupby('Symbol')['Close'].transform(lambda x: x - x.shift(5))

# Volume changes (5 days)
data['5-day Volume Change'] = data.groupby('Symbol')['Volume'].transform(lambda x: x.pct_change(periods=5))

# Historical volatility (5 days)
data['5-day Volatility'] = data.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5).std())

# Price-to-Volume ratio
data['Price-to-Volume'] = data['Close'] / data['Volume']

# Moving Average Convergence Divergence (MACD)
data['12-day EMA'] = data.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
data['26-day EMA'] = data.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
data['MACD'] = data['12-day EMA'] - data['26-day EMA']
data['Signal Line'] = data.groupby('Symbol')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

# Rate of Change (ROC)
data['ROC'] = data.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(periods=10) * 100)

# Historical Highs/Lows
def compute_high_low(series):
    rolling_high = series.expanding(min_periods=1).max()
    rolling_low = series.expanding(min_periods=1).min()
    
    # After 52 days, set the values to the 52-day rolling high/low
    rolling_high[52:] = series.rolling(window=52).max()[52:]
    rolling_low[52:] = series.rolling(window=52).min()[52:]
    
    return rolling_high, rolling_low

data['52-week High'] = data.groupby('Symbol')['Close'].transform(lambda x: compute_high_low(x)[0])
data['52-week Low'] = data.groupby('Symbol')['Close'].transform(lambda x: compute_high_low(x)[1])

data['% Distance from 52-week High'] = ((data['Close'] - data['52-week High']) / data['52-week High']) * 100
data['% Distance from 52-week Low'] = ((data['Close'] - data['52-week Low']) / data['52-week Low']) * 100

## Create target variable
# Convert the 'Date' column to datetime type and extract the day of week
data['Date'] = pd.to_datetime(data['Date'])
data['Day_of_Week'] = data['Date'].dt.dayofweek

# Calculate the 5-day percentage change using Open of the current day and Close 5 days later
data['Close_5_Days_Later'] = data.groupby('Symbol')['Close'].shift(-4)
data['5_Day_%Change'] = ((data['Close_5_Days_Later'] - data['Open']) / data['Open']) * 100

# Extracting the 5-day percentage change for ^GSPC
sp500_5_day_change = data[data['Symbol'] == '^GSPC'][['Date', '5_Day_%Change']].rename(columns={'5_Day_%Change': 'SP500_5_Day_%Change'})

# Merging the 5-day percentage change for each stock with ^GSPC's 5-day percentage change
final_data_5_day = pd.merge(data, sp500_5_day_change, on='Date', how='left')

# Determine if the stock outperformed ^GSPC over 5 days
final_data_5_day['Outperformed'] = np.where(
    (final_data_5_day['5_Day_%Change'].isna()) | (final_data_5_day['SP500_5_Day_%Change'].isna()),
    np.nan,
    final_data_5_day['5_Day_%Change'] > final_data_5_day['SP500_5_Day_%Change']
)

# Shift the 'Outperformed' column back by one day to create 'Outperformed_Predicted_Next_Week' before filtering by Friday
final_data_5_day['Outperformed_Predicted_Next_Week'] = final_data_5_day.groupby('Symbol')['Outperformed'].shift(-1)

# Filter the data to only include rows corresponding to Fridays
friday_data_with_prediction = final_data_5_day[final_data_5_day['Day_of_Week'] == 4]
friday_data_with_prediction

# Elimnate stocks with Rolling 5-day Outperform<=3
perfomance_threshold = 3
latest_date = friday_data_with_prediction['Date'].max()
stocks_to_exclude_5day = friday_data_with_prediction[(friday_data_with_prediction['Date'] == latest_date) & (friday_data_with_prediction['Rolling 5-day Outperform'] <= perfomance_threshold)]['Symbol'].unique()
eliminated_df = friday_data_with_prediction[~friday_data_with_prediction['Symbol'].isin(stocks_to_exclude_5day)]

eliminated_df['Outperformed'] = eliminated_df['Outperformed'].astype(int, errors='ignore')
eliminated_df['Outperformed_Predicted_Next_Week'] = eliminated_df['Outperformed_Predicted_Next_Week'].astype(int,errors='ignore')

eliminated_df.to_csv('data/stock_data_cleaned.csv', index=False)

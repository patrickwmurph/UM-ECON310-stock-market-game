import yfinance as yf
import pandas as pd
import datetime as dt
historical_tickers_giturl = 'https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(08-01-2023).csv'
historical_tickers = pd.read_csv(historical_tickers_giturl)

data = pd.read_csv(historical_tickers_giturl)

def get_latest_tickers_for_date(target_date, data):
    print('Getting latest tickers for date...')
    filtered_data = data[data['date'] <= target_date]
    
    latest_date = filtered_data['date'].iloc[-1]
    tickers = filtered_data['tickers'].iloc[-1].split(',')
    
    return latest_date, tickers

def fetch_stock_data_for_tickers(tickers, start_date, end_date):
    print('Start Fetching Stock Data')
    dataframes = []

    for ticker in tickers:
        print(f'{ticker}')
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if not data.empty:
                data.reset_index(inplace=True)
                data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
                data['Symbol'] = ticker
                dataframes.append(data)
            else:
                print(f"No data found for {ticker} between {start_date} and {end_date}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return dataframes

start_date = '2020-01-01'
end_date = dt.date.today().strftime('%Y-%m-%d')

_, latest_tickers = get_latest_tickers_for_date(end_date, data)

stock_dataframes = fetch_stock_data_for_tickers(latest_tickers, start_date, end_date)
sp_dataframe = fetch_stock_data_for_tickers(['^GSPC'], start_date, end_date)
stock_dataframes.append(sp_dataframe[0])
concatenated_df = pd.concat(stock_dataframes)

concatenated_df.to_csv(f'data/stock_data_{start_date}.csv', index=False)
concatenated_df
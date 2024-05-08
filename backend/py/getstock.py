import os
import pandas as pd
import yfinance as yf
from talib import abstract

class StockData:
    def __init__(self, symbol):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)

    def fetch_current_data(self):
        # 獲取股票的當前詳細資訊
        info = self.stock.info

        # 篩選並整理出需要的股票資訊
        relevant_info = {
            'CurrentPrice': info.get('currentPrice'),  # 當前股價
            'PreviousClose': info.get('regularMarketPreviousClose'),  # 上一交易日的收盤價
            'MarketCap': info.get('marketCap'),  # 公司市值
            'Volume': info.get('volume'),  # 當日成交量
            'WeekHigh': info.get('fiftyTwoWeekHigh'),  # 過去52週的最高股價
            'WeekLow': info.get('fiftyTwoWeekLow')  # 過去52週的最低股價
        }
        
        # 返回整理後的股票資訊
        return relevant_info


    def fetch_historical_data(self, start_date, end_date):
        hist = self.stock.history(start=start_date, end=end_date)
        hist.columns = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
        return hist

    def fetch_fundamentals(self):
        info = self.stock.info
        fundamentals = {
            'Market Cap': info.get('marketCap'),
            'Enterprise Value': info.get('enterpriseValue'),
            'Trailing P/E': info.get('trailingPE'),
            'Forward P/E': info.get('forwardPE'),
            'PEG Ratio': info.get('pegRatio'),
            'Price/Sales': info.get('priceToSalesTrailing12Months'),
            'Price/Book': info.get('priceToBook'),
            'Enterprise to Revenue': info.get('enterpriseToRevenue'),
            'Enterprise to EBITDA': info.get('enterpriseToEbitda'),
            'Dividend Yield': info.get('dividendYield'),
            'Profit Margins': info.get('profitMargins'),
            'Operating Margins': info.get('operatingMargins'),
            'Return on Assets': info.get('returnOnAssets'),
            'Return on Equity': info.get('returnOnEquity'),
            'Revenue Growth': info.get('revenueGrowth'),
            'Earnings Growth': info.get('earningsGrowth')
        }
        return fundamentals

    def save_data_to_csv(self, df, filename_prefix):
        directory = f"./original data/{filename_prefix}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        index = 0
        filename = f"{directory}/{filename_prefix}_history.csv"
        while os.path.exists(filename):
            index += 1
            filename = f"{directory}/{filename_prefix}_history{index:02}.csv"
        
        df.to_csv(filename)
        print(f"文件已儲存為: {filename}")

    def add_technical_indicators(self, data, indicators=['MACD', 'RSI', 'MOM', 'STOCH']):
        for indicator in indicators:
            output = abstract.Function(indicator)(data)
            if isinstance(output, pd.DataFrame):
                data = data.join(output)
            else:
                data[indicator] = output
        return data


# Usage
# stock_data = StockData("TSLA")
# current_data = stock_data.fetch_current_data()
# historical_data = stock_data.fetch_historical_data("2020-01-01", "2024-05-01")
# historical_data = stock_data.add_technical_indicators(historical_data)
# stock_data.save_data_to_csv(historical_data, "TSLA")
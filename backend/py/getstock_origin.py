# pip install yfinance
# pip install TA-Lib

# py -3.10 -m pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl

import yfinance as yf
import pandas as pd
import os
from talib import abstract


def fetch_current_stock_data(stock_symbol):
    # 創建股票對象
    stock = yf.Ticker(stock_symbol)
    # 獲取股票的當前詳細資訊
    info = stock.info
    # 篩選出相關的即時股票資訊
    relevant_info = {
        'Current Price': info.get('regularMarketPrice'),  # 當前價格
        'Previous Close': info.get('regularMarketPreviousClose'),  # 昨日收盤價
        'Market Cap': info.get('marketCap'),  # 市值
        'Volume': info.get('volume'),  # 成交量
        '52 Week High': info.get('fiftyTwoWeekHigh'),  # 52週最高價
        '52 Week Low': info.get('fiftyTwoWeekLow')  # 52週最低價
    }
    return relevant_info

# 獲取股票的歷史資料
def fetch_historical_stock_data(stock_symbol, start_date, end_date):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(start=start_date, end=end_date)
    # 使用小寫列名以符合 TA-Lib 的要求
    hist.columns = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
    return hist


def fetch_stock_fundamentals(stock_symbol):
    # 創建股票對象
    stock = yf.Ticker(stock_symbol)
    # 獲取股票的當前資料
    info = stock.info
    # 從獲取的資料中篩選出相關的基本面資料
    fundamentals = {
        'Market Cap': info.get('marketCap'),  # 市值
        'Enterprise Value': info.get('enterpriseValue'),  # 企業價值
        'Trailing P/E': info.get('trailingPE'),  # 追蹤市盈率
        'Forward P/E': info.get('forwardPE'),  # 預期市盈率
        'PEG Ratio': info.get('pegRatio'),  # PEG比率
        'Price/Sales': info.get('priceToSalesTrailing12Months'),  # 價格/銷售比
        'Price/Book': info.get('priceToBook'),  # 價格/帳面價值比
        'Enterprise to Revenue': info.get('enterpriseToRevenue'),  # 企業價值/營收
        'Enterprise to EBITDA': info.get('enterpriseToEbitda'),  # 企業價值/息稅折舊攤銷前利潤
        'Dividend Yield': info.get('dividendYield'),  # 股息率
        'Profit Margins': info.get('profitMargins'),  # 利潤率
        'Operating Margins': info.get('operatingMargins'),  # 營運利潤率
        'Return on Assets': info.get('returnOnAssets'),  # 資產回報率
        'Return on Equity': info.get('returnOnEquity'),  # 股本回報率
        'Revenue Growth': info.get('revenueGrowth'),  # 營收增長率
        'Earnings Growth': info.get('earningsGrowth')  # 盈利增長率
    }
    return fundamentals

def save_dataframe_to_csv(df, filename_prefix):
    # 定義資料保存的文件夾路徑
    directory = f"./original data/{filename_prefix}"
    # 檢查文件夾是否存在，不存在則創建
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    index = 0
    filename = f"{directory}/{filename_prefix}_history.csv"
    # 檢查文件是否已存在，如果存在，則增加一個索引號來創建新的文件名
    while os.path.exists(filename):
        index += 1
        filename = f"{directory}/{filename_prefix}_history{index:02}.csv"
    
    # 將 DataFrame 保存到 CSV 文件
    df.to_csv(filename)
    print(f"文件已儲存為: {filename}")


# 股票編號/名稱
stock_symbol = "TSLA"
current_data = fetch_current_stock_data(stock_symbol)
print("股票的即時資訊如下：")
print(current_data)
print("\n")

historical_data = fetch_historical_stock_data(stock_symbol, "2020-01-01", "2024-05-01")
print("股票的歷史資訊如下：")
print(historical_data)

fundamentals_data = fetch_stock_fundamentals(stock_symbol)
print("公司的基本面資訊如下：")
for key, value in fundamentals_data.items():
    print(f"{key}: {value}")

# 計算技術指標
ta_list = ['MACD', 'RSI', 'MOM', 'STOCH']
for indicator in ta_list:
    output = abstract.Function(indicator)(historical_data)
    if isinstance(output, pd.DataFrame):
        historical_data = historical_data.join(output)
    else:
        historical_data[indicator] = output

print(historical_data.tail())  # 顯示帶有技術指標的最後幾行數據

# 儲存歷史數據到CSV
save_dataframe_to_csv(historical_data, stock_symbol)


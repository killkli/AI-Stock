from flask import Flask, jsonify

from py.getstock import StockData   # 導入取得股價資訊的類別程式

app = Flask(__name__)

@app.route('/<stock_code>/getcurrent')
def get_stock_data(stock_code):
    stock_data = StockData(stock_code)  # 每次請求都實例化新的股票數據對象
    current_data = stock_data.fetch_current_data()  # 獲取當前股票資訊
    return jsonify({
        'stock_code': stock_code,
        'current_data': current_data
    })

if __name__ == '__main__':
    app.run(debug=True)

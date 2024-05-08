'''
時間序列機器學習(像是股票分析)：由於這種資料前後相鄰的資料相關性非常高，
為了不破壞這種連續性的特性，所以資料不是採取隨機抽取的作法，
而是將前面 75 天的資料作為訓練集，後面 25 天的樣本就是測試集。

基本上應用在股市這種時間序列資料的機器學習模型，大多全部都採用『前進式學習』
優點：比較貼近真實、減少過度擬合的問題發生
'''
# pip install graphviz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 決策樹套件
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
import graphviz 

# 計算混淆矩陣的套件
from sklearn.metrics import confusion_matrix


# 讀取資料集
filepath = "original data/2618.TW/2618.TW_history.csv" 
data = pd.read_csv(filepath)

# 創建預測目標：下一日的收盤價
data['next_close'] = data['close'].shift(-1)

# 檢查每個欄位缺失值數量
# print(data.isnull().sum())

# 去除缺失值
data = data.dropna()

# 再次檢查缺失值數量
# print(data.isnull().sum())

# 選擇特徵和目標變量
X = data[['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits','macd', 'macdsignal', 'macdhist', 'RSI', 'MOM', 'slowk', 'slowd']]
y = data['next_close']

# 創建和訓練決策樹模型
tree_model = DecisionTreeRegressor(max_depth = 7)
tree_model.fit(X, y)

# 獲得特徵重要性
feature_importance = tree_model.feature_importances_

# 將特徵和其對應的重要性列印出來
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': feature_importance}).sort_values(by='importance', ascending=False)
print(feature_importances)

# 繪製條形圖
plt.figure(figsize=(8, 5))  # 設定圖表大小
plt.barh(feature_importances['feature'], feature_importances['importance'], color='skyblue')  # 繪製水平條形圖
plt.xlabel('Importance')  # X軸標籤
plt.ylabel('Feature')  # Y軸標籤
plt.title('Feature Importances')  # 圖表標題
plt.gca().invert_yaxis()  # 將Y軸反向，讓重要性最高的特徵顯示在頂部
plt.show()  # 顯示圖表

# 訓練 LSTM 選擇的特徵
selected_features = ['close']

# 進行特徵選擇
X_selected = X[selected_features]

# 正規化套件
from sklearn.preprocessing import MinMaxScaler

# 初始化MinMaxScaler
scaler = MinMaxScaler()

# 擬合數據並轉換
X_scaled = scaler.fit_transform(X_selected)

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# 創建序列
X_seq, y_seq = create_sequences(X_scaled, y.values)

# 分割訓練集和驗證集(使用 80% 的資料作為訓練集，其中訓練集的 20% 為驗證集)
train_size = int(len(X_seq) * 0.8)
validation_size = int(train_size * 0.1)

X_train = X_seq[:train_size - validation_size]
X_val = X_seq[train_size - validation_size:train_size]
X_test = X_seq[train_size:]

y_train = y_seq[:train_size - validation_size]
y_val = y_seq[train_size - validation_size:train_size]
y_test = y_seq[train_size:]

# 輸出訓練集、驗證集和測試集的形狀
print("訓練集 X 的形狀:", X_train.shape)
print("驗證集 X 的形狀:", X_val.shape)
print("測試集 X 的形狀:", X_test.shape)
print("訓練集 y 的形狀:", y_train.shape)
print("驗證集 y 的形狀:", y_val.shape)
print("測試集 y 的形狀:", y_test.shape)


# 模型相關套件
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 初始化模型
model = Sequential()

# 添加LSTM層
model.add(LSTM(256, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(16, activation='linear'))
# 添加輸出層
model.add(Dense(1, activation='linear'))

#顯示模型摘要資訊
model.summary()  

# 編譯模型
model.compile(optimizer='adam', loss='mse')

# 訓練模型，並添加早期停止以防止過擬合
early_stopping = EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)

model_filepath="lstm.best.hdf5"
checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=2, mode='min',save_best_only=True)
call_backlist = [early_stopping ,checkpoint]

history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=8,
    verbose=2,
    validation_data = (X_val, y_val),
    callbacks= call_backlist
)

# 評估模型
loss = model.evaluate(X_test, y_test)
print("Test loss:", loss)

# 繪製訓練和驗證損失圖表
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 預測
predicted_stock_price = model.predict(X_test)

# 計算和打印評估指標
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 計算評估指標
rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
mae = mean_absolute_error(y_test, predicted_stock_price)
mse = mean_squared_error(y_test, predicted_stock_price)

# 輸出評估指標
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MSE: {mse}')

# 儲存模型檔
model.save('model.keras')

# 繪製評估指標圖表
metrics = ['RMSE', 'MAE', 'MSE']
values = [rmse, mae, mse]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'red'])
plt.title('Model Evaluation Metrics')
plt.ylabel('Value')
plt.show()



'''
時間序列機器學習(像是股票分析)：由於這種資料前後相鄰的資料相關性非常高，
為了不破壞這種連續性的特性，所以資料不是採取隨機抽取的作法，
而是將前面 75 天的資料作為訓練集，後面 25 天的樣本就是測試集。

基本上應用在股市這種時間序列資料的機器學習模型，大多全部都採用『前進式學習』
優點：比較貼近真實、減少過度擬合的問題發生
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.regularizers import l1_l2
from keras.losses import Huber


class MetricsHistory(Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def on_train_begin(self, logs=None):
        self.train_rmse = []
        self.train_mae = []
        self.train_mse = []
        self.train_mape = []
        self.val_rmse = []
        self.val_mae = []
        self.val_mse = []
        self.val_mape = []

    def on_epoch_end(self, epoch, logs=None):
        epsilon = 1e-8  # 添加一个小常数以避免除以零
        train_pred = self.model.predict(self.X_train)
        val_pred = self.model.predict(self.X_val)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
        train_mae = mean_absolute_error(self.y_train, train_pred)
        val_mae = mean_absolute_error(self.y_val, val_pred)
        train_mse = mean_squared_error(self.y_train, train_pred)
        val_mse = mean_squared_error(self.y_val, val_pred)
        train_mape = np.mean(np.abs((self.y_train - train_pred.flatten()) / (self.y_train + epsilon))) * 100
        val_mape = np.mean(np.abs((self.y_val - val_pred.flatten()) / (self.y_val + epsilon))) * 100

        self.train_rmse.append(train_rmse)
        self.val_rmse.append(val_rmse)
        self.train_mae.append(train_mae)
        self.val_mae.append(val_mae)
        self.train_mse.append(train_mse)
        self.val_mse.append(val_mse)
        self.train_mape.append(train_mape)
        self.val_mape.append(val_mape)

# 讀取數據集
data_path = './original data/TSLA/TSLA_history.csv'
stock_data = pd.read_csv(data_path)

# 移除含有缺失值的行
stock_data_cleaned = stock_data.dropna()

# 選擇特徵
features = ['open', 'high', 'low', 'close', 'volume','macdhist','RSI','MOM','slowk','slowd']
X = stock_data_cleaned[features]

# 初始化MinMaxScaler並擬合數據
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 創建時間序列數據集
def create_dataset(data, time_steps=30):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, :-1])  # 所有特徵除了 'close' 都作為輸入
        y.append(data[i, -1])  # 'close' 價格作為目標
    return np.array(X), np.array(y)

# 使用30天的數據作為回測時間
time_steps = 30
X_series, y_series = create_dataset(X_scaled, time_steps)

# 分割數據集
train_val_size = int(len(X_series) * 0.90)
val_size = int(train_val_size * 0.15)

X_train_val, X_test = X_series[:train_val_size], X_series[train_val_size:]
y_train_val, y_test = y_series[:train_val_size], y_series[train_val_size:]

X_train, X_val = X_train_val[:-val_size], X_train_val[-val_size:]
y_train, y_val = y_train_val[:-val_size], y_train_val[-val_size:]

# 建立LSTM模型
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(32, activation='relu'),
    # 添加輸出層
    Dense(1, activation='linear')  # 輸出層使用linear，這個激活函數適合迴歸問題
])

#顯示模型摘要資訊
model.summary()  

# 編譯模型
model.compile(optimizer='adam', loss=Huber())

# 設定早停機制
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# 在模型訓練時傳入回調
metrics_history = MetricsHistory(X_train, y_train, X_val, y_val)

history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping, metrics_history])

model.save("py/model.keras")

# 繪製訓練和驗證的損失曲線
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Train and Validation history loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 繪製RMSE, MAE, MSE, MAPE
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(metrics_history.train_rmse, label='Train RMSE')
plt.plot(metrics_history.val_rmse, label='Validation RMSE', linestyle='--')
plt.title('Train and Validation RMSE')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(metrics_history.train_mae, label='Train MAE')
plt.plot(metrics_history.val_mae, label='Validation MAE', linestyle='--')
plt.title('Train and Validation MAE')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(metrics_history.train_mse, label='Train MSE')
plt.plot(metrics_history.val_mse, label='Validation MSE', linestyle='--')
plt.title('Train and Validation MSE')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(metrics_history.train_mape, label='Train MAPE')
plt.plot(metrics_history.val_mape, label='Validation MAPE', linestyle='--')
plt.title('Train and Validation MAPE')
plt.legend()

plt.tight_layout()
plt.show()

# 預測
y_pred_test = model.predict(X_test).flatten()

# 計算性能指標
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test MAPE: {mape:.4f}%")
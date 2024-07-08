import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Import data
file_path = './data/SMCI.csv'
data = pd.read_csv(file_path)

# Delete "$"s and convert data into float numbers
data['Close'] = data['Close/Last'].str.replace('$', '').astype(float)

# Preparing Data
data['Date'] = pd.to_datetime(data['Date'])

# Preprocessing steps
scaler = MinMaxScaler(feature_range=(0, 1))

time_steps = 90


# Define model functions
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=25, batch_size=32)


def make_prediction(model, x_valid, scaler):
    predictions = model.predict(x_valid)
    predictions = scaler.inverse_transform(predictions)
    return predictions


# Rolling window cross-validation
train_years = 3  # 训练数据年限
valid_years = 1  # 验证数据年限
train_days = train_years * 252  # 假设每年252个交易日
valid_days = valid_years * 252

start_date = data['Date'].min()  # 数据的起始日期
valid_rmse_scores = []
last_valid_preds = None

while start_date + pd.DateOffset(years=train_years + valid_years) <= data['Date'].max():
    end_train_date = start_date + pd.DateOffset(years=train_years)
    end_valid_date = end_train_date + pd.DateOffset(years=valid_years)

    train_set = data[(data['Date'] >= start_date) & (data['Date'] < end_train_date)].copy()
    valid_set = data[(data['Date'] >= end_train_date) & (data['Date'] < end_valid_date)].copy()

    if len(train_set) < train_days or len(valid_set) < valid_days:
        break

    train_data = scaler.fit_transform(train_set['Close'].values.reshape(-1, 1))
    valid_data = scaler.transform(valid_set['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for i in range(time_steps, len(train_data)):
        x_train.append(train_data[i - time_steps:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_valid, y_valid = [], []
    for i in range(time_steps, len(valid_data)):
        x_valid.append(valid_data[i - time_steps:i, 0])
        y_valid.append(valid_data[i, 0])

    if len(x_valid) == 0:
        continue

    x_valid, y_valid = np.array(x_valid), np.array(y_valid)
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

    # 构建并训练模型
    model = build_model()
    train_model(model, x_train, y_train)

    # 进行预测
    valid_preds = make_prediction(model, x_valid, scaler)
    if end_valid_date == data['Date'].max():  # 保存最后一个分割的预测结果用于可视化
        last_valid_preds = valid_preds
        last_valid_dates = valid_set['Date'][time_steps:].values

    # 计算当前分割的RMSE并保存
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
    valid_rmse_scores.append(valid_rmse)
    print(f"Train from {start_date.date()} to {end_train_date.date()}, "
          f"validate from {end_train_date.date()} to {end_valid_date.date()}, RMSE: {valid_rmse}")

    start_date = start_date + pd.DateOffset(years=valid_years)

# 打印所有分割的平均RMSE
average_rmse = np.mean(valid_rmse_scores)
print(f"Average RMSE across all splits: {average_rmse}")

# Visualization for the last split
plt.figure(figsize=(14, 7))
plt.plot(train_set['Date'], train_set['Close'], label='Training Data')
plt.plot(valid_set['Date'][time_steps:], valid_set['Close'].values[time_steps:], label='Validation Data')
if last_valid_preds is not None:
    plt.plot(last_valid_dates, last_valid_preds, label='Last Split Validation Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction with LSTM (Last Split)')
plt.legend()
plt.show()

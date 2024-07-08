# 0. Import all libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout# 1. Import data
file_path = './data/test_20210324.csv'
data = pd.read_csv(file_path)# Data Cleaning

# Remove dollar signs and convert to float for specified columns
columns_to_convert = ['Close/Last', 'Open', 'High', 'Low']
for column in columns_to_convert:
    data[column] = data[column].str.replace('$', '').str.replace(',', '').astype(float)# Preparing Data
data['Date'] = pd.to_datetime(data['Date'])
train_set = data[(data['Date'] >= '2021-03-24') & (data['Date'] < '2024-07-03')].copy()



# 2.1 Time horizon set to be all data for training, no validation
# 2.2 time steps for LSTM
time_steps = 90     # tested [30, 60, 90, 120], 90 is the most efficient one# 3. Data pre-processing
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalize the 'Close/Last' column for both training and validation sets
train_data = scaler.fit_transform(train_set['Close/Last'].values.reshape(-1, 1))

# Initialize lists to store training and validation data
x_train, y_train = [], []
for i in range(time_steps, len(train_data)):
    x_train.append(train_data[i-time_steps:i, 0])
    y_train.append(train_data[i, 0])


# Convert lists to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))



# 4. Build Model - function
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# 5. Training - Function
def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=25, batch_size=32)


# 6. Make Prediction - Function
# def make_prediction(model, scaler):
#     predictions = model.predict(x_train)
#     predictions = scaler.inverse_transform(predictions)
#     return predictions
#

def make_future_prediction(model, last_data, scaler, num_days):
    predictions = []
    current_input = last_data

    for _ in range(num_days):
        current_input_reshaped = np.reshape(current_input, (1, current_input.shape[0], 1))
        prediction = model.predict(current_input_reshaped)
        prediction_rescaled = scaler.inverse_transform(prediction)
        predictions.append(prediction_rescaled[0][0])

        # Update the input data with the latest prediction
        current_input = np.append(current_input[1:], prediction_rescaled, axis=0)

    return predictions


# Execute step 4, 5, 6
model = build_model()
train_model(model, x_train, y_train)

# Predict future prices
num_days_to_predict = 5  # 252 trading days in a year
last_data = train_data[-time_steps:]
future_predictions = make_future_prediction(model, last_data, scaler, num_days_to_predict)

# Generate future dates
future_dates = [train_set['Date'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, num_days_to_predict + 1)]

# 7. Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(train_set['Date'], train_set['Close/Last'], label='Training Data')
plt.plot(future_dates, future_predictions, 'ro', label='Future Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction with LSTM')
plt.legend()
plt.show()

print(f"Predicted prices for the next {num_days_to_predict} trading days: {future_predictions}")
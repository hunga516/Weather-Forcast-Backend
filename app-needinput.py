from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Tải mô hình LSTM
lstm_model = tf.keras.models.load_model('path_to_lstm_model.h5')

# Hàm chuẩn hóa và tạo dữ liệu đầu vào cho LSTM
def prepare_lstm_input(data, time_step):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = np.array(data).reshape(-1, 1)
    X = scaler.fit_transform(X)
    
    X_list = []
    for i in range(len(X) - time_step + 1):
        X_list.append(X[i:i + time_step, 0])
    X = np.array(X_list)

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Invalid data shape after processing: {}".format(X.shape))

    return X.reshape(X.shape[0], X.shape[1], 1), scaler

# Hàm Linear Regression
def linear_regression_predict(data):
    X = np.arange(len(data)).reshape(-1, 1)  # Biến độc lập: thời gian
    y = np.array(data)  # Biến phụ thuộc: dữ liệu nhiệt độ
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    time_step = 24

    # Dự đoán bằng Linear Regression
    linear_predictions = linear_regression_predict(data[-time_step:])

    # Dự đoán bằng LSTM
    X_input, scaler = prepare_lstm_input(data[-time_step:], time_step)
    lstm_predictions = []
    for _ in range(56):  # Dự đoán 56 thời điểm
        y_pred = lstm_model.predict(X_input)
        lstm_predictions.append(y_pred[0, 0])
        X_input = np.append(X_input[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))

    # Kết hợp dự đoán từ cả hai mô hình
    combined_predictions = (linear_predictions + lstm_predictions.flatten()) / 2

    # Tạo mảng kết quả
    result = []
    start_date = pd.to_datetime('2015-12-31')
    for i in range(len(combined_predictions)):
        prediction_date = start_date + pd.Timedelta(hours=(i * 3))
        result.append({
            'time': prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': float(combined_predictions[i])
        })

    return jsonify({'predictions': result})

if __name__ == '__main__':
    app.run(debug=True)

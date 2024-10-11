from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Tải mô hình LSTM
model = tf.keras.models.load_model('/Users/lengocloc/Documents/cloud-cache/vi-education-backend-AI/model/temperature_model_lstm.h5')

# Hàm chuẩn hóa và tạo dữ liệu đầu vào cho LSTM
def prepare_input(data, time_step):
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Chuyển đổi dữ liệu sang mảng numpy và reshape thành mảng 2 chiều
    X = np.array(data).reshape(-1, 1)
    
    # Chuẩn hóa dữ liệu
    X = scaler.fit_transform(X)
    
    # Tạo các chuỗi thời gian cho LSTM
    X_list = []
    for i in range(len(X) - time_step + 1):
        X_list.append(X[i:i + time_step, 0])
    
    X = np.array(X_list)

    # Kiểm tra kích thước của X trước khi reshape
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Invalid data shape after processing: {}".format(X.shape))

    # Trả về dữ liệu đã chuẩn bị (thêm chiều thứ 3 để LSTM sử dụng)
    return X.reshape(X.shape[0], X.shape[1], 1), scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận dữ liệu từ client
    data = request.json['data']
    
    # Chọn số bước thời gian
    time_step = 24  # Sử dụng 24 giờ để dự đoán

    # Chuẩn bị dữ liệu đầu vào
    X_input, scaler = prepare_input(data[-time_step:], time_step)
    
    # Dự đoán cho 56 thời điểm (7 ngày * 8 thời điểm mỗi ngày)
    num_predictions = 56
    predictions = []
    for _ in range(num_predictions):
        y_pred = model.predict(X_input)
        predictions.append(y_pred[0, 0])  # Lưu giá trị dự đoán
        
        # Cập nhật dữ liệu đầu vào cho lần dự đoán tiếp theo
        X_input = np.append(X_input[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

    # Chuyển đổi lại về kích thước gốc
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Tạo mảng kết quả
    result = []
    start_date = pd.to_datetime('2015-12-31')  # Ngày bắt đầu
    for i in range(num_predictions):
        prediction_date = start_date + pd.Timedelta(hours=(i * 3))  # Tính toán thời gian dự đoán
        result.append({
            'time': prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': float(predictions[i][0]),  # Chuyển đổi thành float
            'date': prediction_date.strftime('%Y-%m-%d')
        })

    # Trả về kết quả dự đoán
    return jsonify({'predictions': result})






if __name__ == '__main__':
    app.run(debug=True)

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
    data = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X = []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:i + time_step, 0])
    
    X = np.array(X)
    return X.reshape(X.shape[0], X.shape[1], 1), scaler

@app.route('/')
def index():
    return render_template('/Users/lengocloc/Documents/cloud-cache/vi-education-backend-AI/view/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận dữ liệu từ client
    data = request.json['data']
    
    # Chọn số bước thời gian
    time_step = 24  # Sử dụng 24 giờ để dự đoán 7 ngày tiếp theo

    # Chuẩn bị dữ liệu đầu vào
    X_input, scaler = prepare_input(data[-time_step:], time_step)
    
    # Dự đoán
    y_pred = model.predict(X_input)
    
    # Chuyển đổi lại về kích thước gốc
    y_pred = scaler.inverse_transform(y_pred)
    
    # Trả về kết quả dự đoán
    return jsonify({'predictions': y_pred.flatten().tolist()})

if __name__ == '__main__':
    app.run(debug=True)

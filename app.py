from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)


# Tải mô hình LSTM bằng TensorFlow
lstm_model = load_model('/Users/lengocloc/Documents/cloud-cache/vi-education-backend-AI/model/temperature_model_lstm.h5')

# Tải mô hình Linear Regression bằng joblib (nếu là mô hình scikit-learn)
linear_model = joblib.load('/Users/lengocloc/Documents/cloud-cache/vi-education-backend-AI/model/linear_regression_model.pkl')

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

# Hàm dự đoán bằng Linear Regression
def linear_regression_predict(features):
    features = np.array(features).reshape(1, -1)  # Đảm bảo dữ liệu có dạng (1, 8)
    predictions = linear_model.predict(features)  # Dự đoán bằng Linear Regression
    return predictions.flatten()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   # Nhận dữ liệu từ yêu cầu
    data = request.json['data']['data']  # Truy cập vào cấu trúc chính xác
    lstm_data = data['lstm_data']  # Truy cập lstm_data
    linear_data = data['linear_data']  # Truy cập linear_data
    
    # Số bước thời gian cho LSTM
    time_step = 24

    # Chuẩn bị dữ liệu cho Linear Regression
    linear_predictions = linear_regression_predict(linear_data)

    # Chuẩn bị dữ liệu cho LSTM
    X_input, scaler = prepare_lstm_input(lstm_data, time_step)
    
    # Dự đoán bằng LSTM cho 56 thời điểm (7 ngày * 8 thời điểm mỗi ngày)
    num_predictions = 56
    lstm_predictions = []
    for _ in range(num_predictions):
        y_pred = lstm_model.predict(X_input)
        lstm_predictions.append(y_pred[0, 0])  # Lưu giá trị dự đoán của LSTM
        
        # Cập nhật dữ liệu đầu vào cho lần dự đoán tiếp theo
        X_input = np.append(X_input[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

    # Chuyển đổi giá trị dự đoán của LSTM về giá trị gốc
    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()

    # Kết hợp kết quả từ LSTM và Linear Regression
    # Ở đây ta chỉ kết hợp linear_predictions với một phần dữ liệu từ lstm_predictions
    combined_predictions = (linear_predictions + lstm_predictions[:len(linear_predictions)]) / 2

    # Tạo mảng kết quả trả về
    result = []
    start_date = pd.to_datetime('2024-01-01')  # Ngày bắt đầu (có thể thay đổi tùy theo dữ liệu của bạn)
    for i in range(len(combined_predictions)):
        prediction_date = start_date + pd.Timedelta(hours=(i * 3))  # Mỗi dự đoán cách nhau 3 giờ
        result.append({
            'time': prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': float(combined_predictions[i])  # Chuyển đổi thành kiểu float
        })

    # Trả về kết quả dự đoán
    return jsonify({'predictions': result})

if __name__ == '__main__':
    app.run(debug=True)

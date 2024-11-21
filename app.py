from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from flask_cors import CORS
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Kết nối tới MongoDB
client = MongoClient("mongodb+srv://hunga5160:hunga5160@cluster0.0uxnqct.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true")
db = client["weather_forcast"]
locations_collection = db["locations"]


lstm_model = load_model('./model/temperature_model_lstm.h5')

linear_model = joblib.load('./model/linear_regression_model.pkl')

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

def linear_regression_predict(features):
    features = np.array(features).reshape(1, -1)  
    predictions = linear_model.predict(features) 
    return predictions.flatten()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def getPredict():
    lstm_data = [
                        30.0, 22.0, 22.0, 27.0, 28.0, 27.0, 24.0, 23.0, 22.0, 21.0,
                        20.0, 21.5, 22.5, 23.5, 24.5, 25.0, 26.0, 27.0, 36.0, 29.0,
                        31.0, 30.5, 35.0, 37.0
                    ]
    linear_data = [0, 0, 0, 0]

    time_step = 24
    num_predictions = 56  

    linear_predictions = linear_regression_predict(linear_data)

    X_input, scaler = prepare_lstm_input(lstm_data, time_step)

    lstm_predictions = []
    for _ in range(num_predictions):
        y_pred = lstm_model.predict(X_input)
        lstm_predictions.append(y_pred[0, 0])
        X_input = np.append(X_input[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()

    combined_predictions = (linear_predictions + lstm_predictions[:len(linear_predictions)]) / 2

    result = []
    start_date = pd.to_datetime('2015-12-31')
    for i in range(num_predictions):  
        prediction_date = start_date + pd.Timedelta(hours=(i * 3))  
        result.append({
            'time': prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': float(lstm_predictions[i])  
        })

    return jsonify({'predictions': result})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']['data']
    lstm_data = data['lstm_data']
    linear_data = data['linear_data']

    time_step = 24
    num_predictions = 56  

    linear_predictions = linear_regression_predict(linear_data)

    X_input, scaler = prepare_lstm_input(lstm_data, time_step)

    lstm_predictions = []
    for _ in range(num_predictions):
        y_pred = lstm_model.predict(X_input)
        lstm_predictions.append(y_pred[0, 0])
        X_input = np.append(X_input[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()

    combined_predictions = (linear_predictions + lstm_predictions[:len(linear_predictions)]) / 2

    result = []
    start_date = pd.to_datetime('2015-12-31')
    for i in range(num_predictions):  
        prediction_date = start_date + pd.Timedelta(hours=(i * 3))  
        result.append({
            'time': prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': float(lstm_predictions[i])  
        })

    return jsonify({'predictions': result})

@app.route('/locations', methods=['GET'])
def get_locations():
    name = request.args.get('name')

    try:
        if name:
            query = {"name": {"$regex": name, "$options": "i"}}
            documents = locations_collection.find(query)
        else:
            documents = locations_collection.find()

        result = []
        for doc in documents:
            result.append({
                "id": str(doc["_id"]),
                "name": doc["name"],
                "lat": doc["lat"],
                "lon": doc["lon"]
            })

        return jsonify(result), 200 
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Tải mô hình
linear_model = joblib.load('./model/linear_regression_model_QUAN1.pkl')
# lstm_model = load_model('lstm_model.h5')

# Tải scaler (đã sử dụng trong Linear Regression)
scaler = StandardScaler()
scaler.fit(pd.read_csv('weather_data_2009_2015_QUAN1.csv')[['dwpt', 'rhum', 'wspd', 'pres']])

@app.route('/predict_linear', methods=['POST'])
def predict_linear():
    # Lấy dữ liệu từ request
    data = request.json
    dwpt, rhum, wspd, pres = data['dwpt'], data['rhum'], data['wspd'], data['pres']

    # Chuẩn hóa dữ liệu
    X = scaler.transform([[dwpt, rhum, wspd, pres]])

    # Dự đoán nhiệt độ
    initial_prediction = linear_model.predict(X)[0]

    # Tạo mảng dự đoán cho 10 ngày tiếp theo (80 lần, mỗi 3 giờ)
    num_predictions = 80  # 10 ngày, mỗi 3 giờ
    predictions = []
    start_date = pd.to_datetime('2015-12-31')  # Thay đổi mốc thời gian nếu cần

    for i in range(num_predictions):
        prediction_date = start_date + pd.Timedelta(hours=(i * 3))  # Tăng mỗi 3 giờ
        predictions.append({
            "time": prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
            "temperature": float(initial_prediction + np.random.uniform(-2, 2))  # Biến động nhỏ cho thực tế
        })

    return jsonify({"predictions": predictions})



# @app.route('/predict_lstm', methods=['POST'])
# def predict_lstm():
#     # Lấy dữ liệu từ request
#     data = request.json['sequence']  # Sequence phải là mảng chuỗi thời gian
#     sequence = np.array(data).reshape(1, len(data), 4)
    
#     # Dự đoán
#     prediction = lstm_model.predict(sequence)
#     return jsonify({'predicted_temp': prediction[0][0]})

if __name__ == "__main__":
    app.run(port=8000, debug=True)

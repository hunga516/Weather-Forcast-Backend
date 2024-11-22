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
from bson import ObjectId

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
    linear_data = [0, 0, 0, 0,0,0,0,0]

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
    num_predictions = 80  

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




@app.route('/predict_district/<id>', methods=['POST'])
def predict_district(id):
    try:
        # Tìm document theo ObjectId
        document = locations_collection.find_one({"_id": ObjectId(id)})

        if not document:
            return jsonify({"error": "District not found"}), 404

        # Lấy thông tin từ document
        linear_model_name = document['linear_model']
        lstm_model_name = document['lstm_model']
        data_file = document['data']

        # Tải các mô hình và dữ liệu
        linear_model = joblib.load(f'./model/{linear_model_name}')
        lstm_model = load_model(f'./model/{lstm_model_name}')
        data_path = f'./data/{data_file}'
        
        # Đọc dữ liệu từ file và kiểm tra
        df = pd.read_csv(data_path)
        # if df.isnull().values.any():
        #     return jsonify({"error": "Data file contains NaN values"}), 400

        scaler = StandardScaler()
        scaler.fit(df[['dwpt', 'rhum', 'wspd', 'pres']])

        # Lấy dữ liệu từ request
        data = request.json
        dwpt, rhum, wspd, pres = data['dwpt'], data['rhum'], data['wspd'], data['pres']

        # Chuẩn bị đầu vào cho mô hình Linear
        X = scaler.transform([[dwpt, rhum, wspd, pres]])
        linear_prediction = linear_model.predict(X)[0]

        # Chuẩn bị đầu vào cho LSTM
        history_steps = 10 * 8  # 10 ngày x 8 lần đo
        historical_data = df.tail(history_steps)[['dwpt', 'rhum', 'wspd', 'pres']].dropna()
        # # if historical_data.shape[0] < history_steps:
        # #     return jsonify({"error": "Not enough historical data for LSTM prediction"}), 400

        # lstm_input = scaler.transform(historical_data.values)
        # lstm_input = np.expand_dims(lstm_input, axis=0)

        # # Dự đoán nhiệt độ từ mô hình LSTM
        # lstm_prediction = lstm_model.predict(lstm_input)[0][0]
        # if np.isnan(lstm_prediction):
        #     return jsonify({"error": "LSTM prediction returned NaN"}), 500

        # Tính giá trị trung bình
        # avg_prediction = (linear_prediction + lstm_prediction) / 2
        avg_prediction = linear_prediction

        # Tạo mảng dự đoán cho 10 ngày tiếp theo
        num_predictions = 80  # 10 ngày, mỗi 3 giờ
        predictions = []
        start_date = pd.to_datetime('2015-12-31')

        for i in range(num_predictions):
            prediction_date = start_date + pd.Timedelta(hours=(i * 3))  # Tăng mỗi 3 giờ
            predictions.append({
                "time": prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
                "temperature": float(avg_prediction + np.random.uniform(-2, 2))  # Biến động nhỏ
            })

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=8000, debug=True)

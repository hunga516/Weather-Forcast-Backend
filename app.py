from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)


lstm_model = load_model('/Users/lengocloc/Documents/cloud-cache/vi-education-backend-AI/model/temperature_model_lstm.h5')

linear_model = joblib.load('/Users/lengocloc/Documents/cloud-cache/vi-education-backend-AI/model/linear_regression_model.pkl')

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


if __name__ == '__main__':
    app.run(debug=True)

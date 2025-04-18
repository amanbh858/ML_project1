from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and preprocessing tools
model = joblib.load('XGBoost_model.pkl')  # Trained regression model
scaler = joblib.load('scaler.pkl')      # Fitted scaler for input features

# Feature order used during training (no label encoding needed now)
feature_names = [
    'Water Station P.17 [m³/s] ',
    'Water Station N.67 [m³/s] ',
    'Water Station C.13 [m³/s] ',
    'Rainfall TCP004 (mm)',
    'Rainfall CPY010 (mm)',
    'Rainfall 48415 (mm)',
    'Rainfall LBI001 (mm)',
    'Min_Temp',
    'Max_Temp',
    'relative humidity (%)'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form and convert to floats
        input_data = [float(request.form[name]) for name in feature_names]
        input_array = np.array(input_data).reshape(1, -1)

        # Scale features
        input_scaled = scaler.transform(input_array)

        # Make prediction (regression)
        prediction = model.predict(input_scaled)[0]

        # Interpret prediction as a flood category
        if prediction < 2500:
            label = "Normal"
        elif 2500 <= prediction <= 3500:
            label = "Flood Risk"
        else:
            label = "Flood"

        return render_template('result.html', prediction_text=f'Predicted Water Flow at Water Station C.29A: {prediction:.2f} m³/s ({label})')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import json
import numpy as np
import pmdarima as pm
from prophet import Prophet
import pandas as pd

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        values_str = data.get('values')
        future_periods = data.get('future_periods')
        model_type = data.get('model_type', 'AUTOARIMA').upper()
        indicator = data.get('indicator', 'Unknown')

        if not values_str or future_periods is None:
            return jsonify({'error': 'Missing values or future_periods'}), 400
        if not isinstance(future_periods, int) or future_periods <= 0:
            return jsonify({'error': 'future_periods must be a positive integer'}), 400

        try:
            past_values = json.loads(values_str)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON format for values'}), 400

        if not past_values or not all(isinstance(x, (int, float)) for x in past_values):
            return jsonify({'error': 'Values must be a non-empty list of numbers'}), 400
        if len(past_values) < 12:
            return jsonify({'error': 'At least 12 historical values required'}), 400

        past_values = np.array(past_values, dtype=float)

        forecasted_values = []
        if model_type == 'AUTOARIMA':
            model = pm.auto_arima(
                past_values,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                max_p=3, max_q=3, max_d=2,
                trace=False
            )
            forecasted_values = model.predict(n_periods=future_periods).tolist()

        elif model_type == 'SARIMA':
            return jsonify({'error': 'SARIMA not implemented yet'}), 501

        elif model_type == 'PROPHET':
            dates = pd.date_range(start='2024-01-01', periods=len(past_values), freq='ME')
            df = pd.DataFrame({'ds': dates, 'y': past_values})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(df)
            future = model.make_future_dataframe(periods=future_periods, freq='ME')
            forecast = model.predict(future)
            forecasted_values = forecast['yhat'][-future_periods:].tolist()

        else:
            return jsonify({'error': f'Unsupported model_type: {model_type}'}), 400

        return jsonify({'forecast': forecasted_values})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
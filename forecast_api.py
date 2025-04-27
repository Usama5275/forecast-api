from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def forecast():
    # Get JSON data from APEX
    data = request.get_json()

    # Process the data (simple mock example for now)
    past_values = data.get('values')
    future_periods = data.get('future_periods')

    # Let's pretend we ran a forecasting model (for simplicity, we return mock data)
    forecasted_values = [x + 10 for x in past_values]  # Just a simple transformation as a placeholder

    return jsonify({'forecast': forecasted_values})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


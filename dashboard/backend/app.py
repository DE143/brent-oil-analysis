from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Load precomputed data
data = pd.read_csv('data/processed/cleaned_prices.csv')
events = pd.read_csv('data/processed/events_dataset.csv')
change_points = pd.read_csv('results/change_point_impacts.csv')
correlations = pd.read_csv('results/event_correlations.csv')

@app.route('/api/prices', methods=['GET'])
def get_prices():
    """Get price data with optional date range"""
    start_date = request.args.get('start', '1987-05-20')
    end_date = request.args.get('end', '2022-09-30')
    
    filtered = data[
        (data['Date'] >= start_date) & 
        (data['Date'] <= end_date)
    ]
    
    return jsonify({
        'dates': filtered['Date'].tolist(),
        'prices': filtered['Price'].tolist(),
        'returns': filtered['Log_Return'].tolist()
    })

@app.route('/api/events', methods=['GET'])
def get_events():
    """Get events within date range"""
    start_date = request.args.get('start', '1987-05-20')
    end_date = request.args.get('end', '2022-09-30')
    
    filtered = events[
        (events['Date'] >= start_date) & 
        (events['Date'] <= end_date)
    ]
    
    return jsonify(filtered.to_dict('records'))

@app.route('/api/change-points', methods=['GET'])
def get_change_points():
    """Get detected change points"""
    return jsonify(change_points.to_dict('records'))

@app.route('/api/correlations', methods=['GET'])
def get_correlations():
    """Get event-price correlations"""
    event_filter = request.args.get('event')
    if event_filter:
        filtered = correlations[correlations['event_name'] == event_filter]
    else:
        filtered = correlations
    
    return jsonify(filtered.to_dict('records'))

@app.route('/api/volatility', methods=['GET'])
def get_volatility():
    """Calculate volatility metrics"""
    window = int(request.args.get('window', 30))
    
    data['Volatility'] = data['Log_Return'].rolling(window=window).std() * np.sqrt(252)
    
    return jsonify({
        'dates': data['Date'].tolist(),
        'volatility': data['Volatility'].tolist()
    })

@app.route('/api/summary-stats', methods=['GET'])
def get_summary_stats():
    """Get summary statistics"""
    return jsonify({
        'total_days': len(data),
        'average_price': float(data['Price'].mean()),
        'max_price': float(data['Price'].max()),
        'min_price': float(data['Price'].min()),
        'total_events': len(events),
        'detected_change_points': len(change_points),
        'average_volatility': float(data['Log_Return'].std() * np.sqrt(252))
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
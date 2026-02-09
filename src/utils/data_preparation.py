"""
Data preparation utilities for Brent oil price analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def prepare_event_dataset(output_path='data/processed/events_dataset.csv'):
    """
    Prepare comprehensive event dataset
    """
    events_data = [
        # Major conflicts
        {'Date': '1990-08-02', 'Event Name': 'Gulf War', 'Category': 'Conflict', 
         'Description': 'Iraq invades Kuwait', 'Region': 'Middle East', 'Expected Impact': 'Severe negative'},
        
        {'Date': '2003-03-20', 'Event Name': 'Iraq War', 'Category': 'Conflict',
         'Description': 'US-led invasion of Iraq', 'Region': 'Middle East', 'Expected Impact': 'Positive'},
        
        {'Date': '2011-02-15', 'Event Name': 'Arab Spring', 'Category': 'Conflict',
         'Description': 'Political instability across Middle East', 'Region': 'Middle East', 'Expected Impact': 'Positive'},
        
        {'Date': '2014-06-10', 'Event Name': 'ISIS Expansion', 'Category': 'Conflict',
         'Description': 'ISIS captures Iraqi oil fields', 'Region': 'Middle East', 'Expected Impact': 'Positive'},
        
        {'Date': '2022-02-24', 'Event Name': 'Russia-Ukraine War', 'Category': 'Conflict',
         'Description': 'Russia invades Ukraine, sanctions on Russian oil', 'Region': 'Europe', 'Expected Impact': 'Severe positive'},
        
        # OPEC decisions
        {'Date': '2008-10-24', 'Event Name': 'OPEC Production Cut', 'Category': 'Policy',
         'Description': 'OPEC cuts production by 1.5M bpd', 'Organization': 'OPEC', 'Expected Impact': 'Positive'},
        
        {'Date': '2014-11-27', 'Event Name': 'OPEC Maintains Production', 'Category': 'Policy',
         'Description': 'OPEC decides not to cut production despite price drop', 'Organization': 'OPEC', 'Expected Impact': 'Negative'},
        
        {'Date': '2016-11-30', 'Event Name': 'OPEC+ Production Cut', 'Category': 'Policy',
         'Description': 'OPEC and allies agree to cut production by 1.2M bpd', 'Organization': 'OPEC+', 'Expected Impact': 'Positive'},
        
        {'Date': '2020-03-06', 'Event Name': 'OPEC+ Price War', 'Category': 'Policy',
         'Description': 'Russia-Saudi Arabia price war begins', 'Organization': 'OPEC+', 'Expected Impact': 'Severe negative'},
        
        {'Date': '2020-04-12', 'Event Name': 'OPEC+ Historic Cut', 'Category': 'Policy',
         'Description': 'OPEC+ agrees to cut 9.7M bpd', 'Organization': 'OPEC+', 'Expected Impact': 'Positive'},
        
        # Economic crises
        {'Date': '2008-09-15', 'Event Name': 'Lehman Brothers Collapse', 'Category': 'Economic',
         'Description': 'Global financial crisis begins', 'Region': 'Global', 'Expected Impact': 'Severe negative'},
        
        {'Date': '2011-08-05', 'Event Name': 'US Credit Downgrade', 'Category': 'Economic',
         'Description': 'S&P downgrades US credit rating', 'Region': 'USA', 'Expected Impact': 'Negative'},
        
        {'Date': '2015-08-11', 'Event Name': 'Chinese Stock Market Crash', 'Category': 'Economic',
         'Description': 'Chinese market turmoil affects global demand', 'Region': 'China', 'Expected Impact': 'Negative'},
        
        {'Date': '2020-03-11', 'Event Name': 'COVID-19 Pandemic Declaration', 'Category': 'Pandemic',
         'Description': 'WHO declares COVID-19 pandemic, demand collapses', 'Region': 'Global', 'Expected Impact': 'Extreme negative'},
        
        # Sanctions
        {'Date': '2012-07-01', 'Event Name': 'EU Iran Oil Embargo', 'Category': 'Sanctions',
         'Description': 'EU imposes oil embargo on Iran', 'Region': 'Middle East', 'Expected Impact': 'Positive'},
        
        {'Date': '2015-07-14', 'Event Name': 'Iran Nuclear Deal', 'Category': 'Sanctions',
         'Description': 'Sanctions lifted on Iran, oil returns to market', 'Region': 'Middle East', 'Expected Impact': 'Negative'},
        
        {'Date': '2018-05-08', 'Event Name': 'US Iran Sanctions', 'Category': 'Sanctions',
         'Description': 'US reinstates sanctions on Iran', 'Region': 'Middle East', 'Expected Impact': 'Positive'},
        
        {'Date': '2022-03-08', 'Event Name': 'US Ban on Russian Oil', 'Category': 'Sanctions',
         'Description': 'US bans Russian oil imports', 'Region': 'Global', 'Expected Impact': 'Positive'},
        
        # Natural disasters
        {'Date': '2005-08-29', 'Event Name': 'Hurricane Katrina', 'Category': 'Natural Disaster',
         'Description': 'Disrupts US Gulf Coast oil production', 'Region': 'USA', 'Expected Impact': 'Positive'},
        
        {'Date': '2017-09-20', 'Event Name': 'Hurricane Harvey', 'Category': 'Natural Disaster',
         'Description': 'Texas refinery shutdowns', 'Region': 'USA', 'Expected Impact': 'Positive'},
        
        # Other significant events
        {'Date': '2019-09-14', 'Event Name': 'Saudi Oil Attacks', 'Category': 'Conflict',
         'Description': 'Drone attacks on Saudi Aramco facilities', 'Region': 'Middle East', 'Expected Impact': 'Positive'},
        
        {'Date': '2021-01-20', 'Event Name': 'Biden Inauguration', 'Category': 'Policy',
         'Description': 'US energy policy shift', 'Region': 'USA', 'Expected Impact': 'Mixed'},
        
        {'Date': '2021-11-23', 'Event Name': 'US Strategic Petroleum Release', 'Category': 'Policy',
         'Description': 'US coordinates release of 50M barrels from SPR', 'Region': 'USA', 'Expected Impact': 'Negative'},
    ]
    
    events_df = pd.DataFrame(events_data)
    events_df['Date'] = pd.to_datetime(events_df['Date'])
    events_df = events_df.sort_values('Date').reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    events_df.to_csv(output_path, index=False)
    print(f"Event dataset saved to {output_path}")
    print(f"Total events: {len(events_df)}")
    
    return events_df

def create_macroeconomic_dataset(output_path='data/external/macroeconomic_indicators.csv'):
    """
    Create synthetic macroeconomic dataset for analysis
    """
    # Generate date range
    dates = pd.date_range(start='1987-05-20', end='2022-09-30', freq='M')
    
    # Create synthetic data
    np.random.seed(42)
    n = len(dates)
    
    macro_data = pd.DataFrame({
        'Date': dates,
        'GDP_Growth': np.random.normal(2.5, 1.5, n),  # Annual GDP growth %
        'Inflation': np.random.normal(2.0, 1.0, n),   # Annual inflation %
        'Interest_Rate': np.random.normal(3.0, 2.0, n),  # Central bank interest rate %
        'USD_Index': np.random.normal(90, 10, n),     # US Dollar Index
        'VIX': np.random.normal(20, 5, n),           # Volatility index
        'Industrial_Production': np.random.normal(100, 5, n),  # Industrial production index
        'Unemployment': np.random.normal(6.0, 1.5, n)  # Unemployment rate %
    })
    
    # Add trends and seasonality
    t = np.arange(n)
    macro_data['GDP_Growth'] += 0.01 * np.sin(2 * np.pi * t / 12)  # Annual seasonality
    macro_data['Inflation'] += 0.005 * t  # Slight upward trend
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    macro_data.to_csv(output_path, index=False)
    print(f"Macroeconomic dataset saved to {output_path}")
    
    return macro_data

def prepare_training_data(price_data, events_data, lookback_days=30, forecast_days=7):
    """
    Prepare data for machine learning models
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data
    events_data : pd.DataFrame
        Events data
    lookback_days : int
        Number of days to look back for features
    forecast_days : int
        Number of days to forecast ahead
        
    Returns:
    --------
    tuple
        (X, y, feature_names)
    """
    # Create lag features
    feature_data = price_data.copy()
    
    # Price-based features
    for lag in range(1, lookback_days + 1):
        feature_data[f'Price_Lag_{lag}'] = feature_data['Price'].shift(lag)
        feature_data[f'Return_Lag_{lag}'] = feature_data['Return'].shift(lag)
    
    # Rolling statistics
    feature_data['Price_MA_7'] = feature_data['Price'].rolling(window=7).mean()
    feature_data['Price_MA_30'] = feature_data['Price'].rolling(window=30).mean()
    feature_data['Return_Std_30'] = feature_data['Return'].rolling(window=30).std()
    
    # Event features
    feature_data['Event_Count_30D'] = 0
    feature_data['Major_Event_Count_30D'] = 0
    
    for idx, row in feature_data.iterrows():
        date = row['Date']
        
        # Count events in past 30 days
        past_events = events_data[
            (events_data['Date'] <= date) & 
            (events_data['Date'] > date - pd.Timedelta(days=30))
        ]
        
        feature_data.loc[idx, 'Event_Count_30D'] = len(past_events)
        
        # Count major events (conflicts, major policy changes)
        major_events = past_events[
            past_events['Category'].isin(['Conflict', 'Policy', 'Economic'])
        ]
        feature_data.loc[idx, 'Major_Event_Count_30D'] = len(major_events)
    
    # Target variable: price change in next N days
    feature_data['Target_Price'] = feature_data['Price'].shift(-forecast_days)
    feature_data['Target_Return'] = (feature_data['Target_Price'] - feature_data['Price']) / feature_data['Price']
    
    # Drop NaN values
    feature_data = feature_data.dropna()
    
    # Separate features and target
    exclude_cols = ['Date', 'Target_Price', 'Target_Return', 'Log_Price', 'Log_Return']
    feature_cols = [col for col in feature_data.columns if col not in exclude_cols]
    
    X = feature_data[feature_cols].values
    y = feature_data['Target_Return'].values
    
    print(f"Prepared dataset with {len(X)} samples")
    print(f"Number of features: {len(feature_cols)}")
    
    return X, y, feature_cols
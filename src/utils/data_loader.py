"""
Data loading utilities for Brent oil price analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_brent_prices(filepath):
    """
    Load and clean Brent oil price data
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with Date, Price, and derived columns
    """
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Handle missing values
        if df['Price'].isnull().sum() > 0:
            print(f"Found {df['Price'].isnull().sum()} missing values. Interpolating...")
            df['Price'] = df['Price'].interpolate(method='linear')
            
            # Forward fill any remaining NaNs at beginning
            df['Price'] = df['Price'].ffill()
            df['Price'] = df['Price'].bfill()
        
        # Calculate derived metrics
        df['Log_Price'] = np.log(df['Price'])
        df['Return'] = df['Price'].pct_change()
        df['Log_Return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
        
        # Calculate rolling metrics
        df['Rolling_Mean_30'] = df['Price'].rolling(window=30).mean()
        df['Rolling_Std_30'] = df['Price'].rolling(window=30).std()
        df['Volatility_30'] = df['Log_Return'].rolling(window=30).std() * np.sqrt(252)
        
        print(f"Loaded data from {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"Total observations: {len(df)}")
        print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def load_events_data(filepath):
    """
    Load and clean events data
    
    Parameters:
    -----------
    filepath : str
        Path to events CSV file
        
    Returns:
    --------
    pd.DataFrame
        Cleaned events DataFrame
    """
    try:
        events_df = pd.read_csv(filepath)
        events_df['Date'] = pd.to_datetime(events_df['Date'])
        events_df = events_df.sort_values('Date').reset_index(drop=True)
        
        # Add impact score based on category
        impact_scores = {
            'Conflict': 3,
            'Policy': 2,
            'Economic': 2,
            'Sanctions': 2,
            'Political': 1,
            'Natural Disaster': 2,
            'Pandemic': 3
        }
        
        events_df['Impact_Score'] = events_df['Category'].map(
            lambda x: impact_scores.get(x, 1)
        )
        
        print(f"Loaded {len(events_df)} events")
        print(f"Events by category:\n{events_df['Category'].value_counts()}")
        
        return events_df
        
    except Exception as e:
        print(f"Error loading events data: {e}")
        raise

def create_train_test_split(data, test_size=0.2, date_split=None):
    """
    Create train-test split for time series data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    test_size : float
        Proportion of data for testing
    date_split : str
        Specific date to split on (format: 'YYYY-MM-DD')
        
    Returns:
    --------
    tuple
        (train_data, test_data)
    """
    if date_split:
        split_date = pd.to_datetime(date_split)
        train_data = data[data['Date'] < split_date]
        test_data = data[data['Date'] >= split_date]
    else:
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
    
    print(f"Training data: {len(train_data)} observations ({train_data['Date'].min().date()} to {train_data['Date'].max().date()})")
    print(f"Test data: {len(test_data)} observations ({test_data['Date'].min().date()} to {test_data['Date'].max().date()})")
    
    return train_data, test_data

def filter_data_by_date(data, start_date=None, end_date=None):
    """
    Filter data by date range
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    start_date : str
        Start date (format: 'YYYY-MM-DD')
    end_date : str
        End date (format: 'YYYY-MM-DD')
        
    Returns:
    --------
    pd.DataFrame
        Filtered data
    """
    filtered = data.copy()
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        filtered = filtered[filtered['Date'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        filtered = filtered[filtered['Date'] <= end_date]
    
    print(f"Filtered to {len(filtered)} observations")
    return filtered

def get_price_statistics(data):
    """
    Calculate comprehensive price statistics
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data
        
    Returns:
    --------
    dict
        Dictionary of statistics
    """
    stats = {
        'mean_price': float(data['Price'].mean()),
        'median_price': float(data['Price'].median()),
        'std_price': float(data['Price'].std()),
        'min_price': float(data['Price'].min()),
        'max_price': float(data['Price'].max()),
        'total_returns': float(data['Return'].sum()),
        'avg_daily_return': float(data['Return'].mean()),
        'volatility': float(data['Log_Return'].std() * np.sqrt(252)),
        'skewness': float(data['Return'].skew()),
        'kurtosis': float(data['Return'].kurtosis()),
        'sharpe_ratio': float(data['Return'].mean() / data['Return'].std() * np.sqrt(252) if data['Return'].std() > 0 else 0)
    }
    
    return stats
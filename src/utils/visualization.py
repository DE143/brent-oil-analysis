"""
Visualization utilities for Brent oil price analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_price_timeline(data, events_df=None, figsize=(15, 8), title="Brent Oil Price Timeline"):
    """
    Plot price timeline with optional event markers
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data
    events_df : pd.DataFrame, optional
        Events data
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price
    ax1.plot(data['Date'], data['Price'], color='steelblue', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Price (USD/barrel)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add event markers if provided
    if events_df is not None:
        for _, event in events_df.iterrows():
            event_date = pd.to_datetime(event['Date'])
            # Find closest price
            closest_idx = (data['Date'] - event_date).abs().argmin()
            event_price = data.iloc[closest_idx]['Price']
            
            # Plot marker
            ax1.scatter(event_date, event_price, color='red', s=50, 
                       alpha=0.7, zorder=5)
            
            # Add annotation for major events
            if event['Impact_Score'] >= 2:
                ax1.annotate(event['Event Name'], 
                           (event_date, event_price),
                           xytext=(10, 10), 
                           textcoords='offset points',
                           fontsize=8,
                           arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    # Plot returns
    ax2.plot(data['Date'], data['Return'] * 100, color='coral', linewidth=0.8, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Daily Return (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    return fig

def plot_distributions(data, figsize=(15, 10)):
    """
    Plot distribution analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Price distribution
    axes[0, 0].hist(data['Price'].dropna(), bins=50, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Price Distribution', fontsize=12)
    axes[0, 0].set_xlabel('Price (USD)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Log returns distribution
    axes[0, 1].hist(data['Log_Return'].dropna(), bins=100, color='coral', alpha=0.7)
    axes[0, 1].set_title('Log Returns Distribution', fontsize=12)
    axes[0, 1].set_xlabel('Log Return')
    axes[0, 1].set_ylabel('Frequency')
    
    # QQ plot of returns
    from scipy import stats
    stats.probplot(data['Log_Return'].dropna(), dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot of Log Returns', fontsize=12)
    
    # Rolling volatility
    axes[1, 0].plot(data['Date'], data['Volatility_30'] * 100, color='purple', linewidth=1)
    axes[1, 0].set_title('30-Day Rolling Volatility', fontsize=12)
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Volatility (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Autocorrelation of returns
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(data['Return'].dropna(), ax=axes[1, 1])
    axes[1, 1].set_title('Autocorrelation of Returns', fontsize=12)
    
    # Rolling statistics
    axes[1, 2].plot(data['Date'], data['Rolling_Mean_30'], label='30-Day MA', color='green', alpha=0.7)
    axes[1, 2].plot(data['Date'], data['Price'], label='Price', color='steelblue', alpha=0.3, linewidth=0.5)
    axes[1, 2].fill_between(data['Date'], 
                           data['Rolling_Mean_30'] - data['Rolling_Std_30'],
                           data['Rolling_Mean_30'] + data['Rolling_Std_30'],
                           alpha=0.2, color='green')
    axes[1, 2].set_title('Price with Moving Average ± Std Dev', fontsize=12)
    axes[1, 2].set_xlabel('Date')
    axes[1, 2].set_ylabel('Price (USD)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_regime_analysis(data, change_points, figsize=(15, 8)):
    """
    Plot regime analysis with change points
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data
    change_points : list
        List of change point dates
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price with regimes
    ax1.plot(data['Date'], data['Price'], color='steelblue', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Price (USD/barrel)', fontsize=12)
    ax1.set_title('Price Regimes with Change Points', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Highlight regimes
    dates = data['Date'].values
    for i, cp_date in enumerate(change_points):
        if i == 0:
            start_idx = 0
        else:
            start_idx = np.where(dates == change_points[i-1])[0][0]
        
        end_idx = np.where(dates == cp_date)[0][0]
        
        ax1.axvspan(dates[start_idx], dates[end_idx], 
                   alpha=0.2, color=f'C{i}', 
                   label=f'Regime {i+1}' if i == 0 else "")
    
    # Plot volatility
    ax2.plot(data['Date'], data['Volatility_30'] * 100, color='coral', linewidth=1)
    ax2.set_ylabel('Volatility (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add change point lines
    for cp_date in change_points:
        for ax in [ax1, ax2]:
            ax.axvline(x=cp_date, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    return fig

def plot_event_impact(data, event_date, window_days=60, figsize=(12, 8)):
    """
    Plot price impact around a specific event
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data
    event_date : str or datetime
        Event date
    window_days : int
        Days before and after event
    figsize : tuple
        Figure size
    """
    event_date = pd.to_datetime(event_date)
    
    # Filter data around event
    start_date = event_date - pd.Timedelta(days=window_days)
    end_date = event_date + pd.Timedelta(days=window_days)
    
    event_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()
    
    # Calculate percentage change from event date
    event_price = data[data['Date'] == event_date]['Price'].values
    if len(event_price) > 0:
        event_price = event_price[0]
        event_data['Pct_Change'] = (event_data['Price'] - event_price) / event_price * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot price
    ax1.plot(event_data['Date'], event_data['Price'], color='steelblue', linewidth=2)
    ax1.axvline(x=event_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Event Date')
    ax1.set_ylabel('Price (USD/barrel)', fontsize=12)
    ax1.set_title(f'Price Impact Analysis', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot percentage change
    if 'Pct_Change' in event_data.columns:
        ax2.bar(event_data['Date'], event_data['Pct_Change'], 
               color=np.where(event_data['Pct_Change'] >= 0, 'green', 'red'),
               alpha=0.6)
        ax2.axvline(x=event_date, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax2.set_ylabel('Price Change (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    return fig
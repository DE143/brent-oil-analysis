"""
Advanced models for time series analysis
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class VectorAutoregressionModel:
    """
    Vector Autoregression model for multivariate time series
    """
    
    def __init__(self, max_lags: int = 10):
        self.max_lags = max_lags
        self.model = None
        self.results = None
        self.selected_lags = None
        
    def select_lags(self, data: pd.DataFrame, ic: str = 'aic'):
        """
        Select optimal lag length using information criteria
        
        Parameters:
        -----------
        data : pd.DataFrame
            Multivariate time series data
        ic : str
            Information criterion: 'aic', 'bic', 'hqic', 'fpe'
        """
        # Fit VAR with different lag lengths
        best_ic = np.inf
        best_lags = 1
        
        for lags in range(1, self.max_lags + 1):
            try:
                model = VAR(data)
                results = model.fit(lags)
                
                # Get information criterion
                if ic == 'aic':
                    current_ic = results.aic
                elif ic == 'bic':
                    current_ic = results.bic
                elif ic == 'hqic':
                    current_ic = results.hqic
                elif ic == 'fpe':
                    current_ic = results.fpe
                else:
                    raise ValueError(f"Unknown information criterion: {ic}")
                
                if current_ic < best_ic:
                    best_ic = current_ic
                    best_lags = lags
                    
            except Exception as e:
                continue
        
        self.selected_lags = best_lags
        print(f"Selected {best_lags} lags based on {ic.upper()}")
        
        return best_lags
    
    def fit(self, data: pd.DataFrame, lags: int = None):
        """
        Fit VAR model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Multivariate time series data
        lags : int, optional
            Number of lags (if None, auto-select)
        """
        if lags is None:
            lags = self.select_lags(data)
        
        self.model = VAR(data)
        self.results = self.model.fit(lags=lags)
        
        print("VAR Model Summary:")
        print(self.results.summary())
        
        return self.results
    
    def forecast(self, steps: int = 10):
        """
        Generate forecasts
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        pd.DataFrame
            Forecasted values
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.results.forecast(self.results.y, steps=steps)
        forecast_df = pd.DataFrame(forecast, columns=self.results.names)
        
        return forecast_df
    
    def impulse_response(self, periods: int = 20):
        """
        Compute impulse response functions
        
        Parameters:
        -----------
        periods : int
            Number of periods for IRF
            
        Returns:
        --------
        dict
            Impulse response functions
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        irf = self.results.irf(periods=periods)
        
        return {
            'irfs': irf.irfs,
            'orth_irfs': irf.orth_irfs,
            'cum_effects': irf.cum_effects,
            'orth_cum_effects': irf.orth_cum_effects
        }

class MarkovSwitchingModel:
    """
    Markov Switching model for regime detection
    """
    
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.results = None
        
    def fit(self, data: pd.Series, model_type: str = 'mean'):
        """
        Fit Markov Switching model
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
        model_type : str
            Type of model: 'mean', 'mean_var', 'ar'
        """
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        
        if model_type == 'mean':
            # Switching mean
            self.model = MarkovRegression(data, k_regimes=self.n_regimes, trend='c')
        elif model_type == 'mean_var':
            # Switching mean and variance
            self.model = MarkovRegression(data, k_regimes=self.n_regimes, trend='c', 
                                         switching_variance=True)
        elif model_type == 'ar':
            # AR model with switching parameters
            self.model = MarkovRegression(data, k_regimes=self.n_regimes, trend='c',
                                         order=1, switching_variance=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.results = self.model.fit()
        
        print("Markov Switching Model Summary:")
        print(self.results.summary())
        
        return self.results
    
    def get_regime_probabilities(self):
        """
        Get smoothed regime probabilities
        
        Returns:
        --------
        pd.DataFrame
            Regime probabilities
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        probs = self.results.smoothed_marginal_probabilities
        return probs
    
    def get_regime_parameters(self):
        """
        Get estimated regime parameters
        
        Returns:
        --------
        dict
            Regime parameters
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        params = self.results.params
        return params

class GaussianMixtureRegimeDetection:
    """
    Gaussian Mixture Model for regime detection
    """
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.gmm = None
        self.scaler = StandardScaler()
        
    def extract_features(self, data: pd.Series, window: int = 20):
        """
        Extract features for regime detection
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
        window : int
            Rolling window size
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix
        """
        features = pd.DataFrame()
        
        # Basic statistics
        features['returns'] = data.pct_change()
        features['log_returns'] = np.log(data) - np.log(data.shift(1))
        
        # Rolling statistics
        features['rolling_mean'] = data.rolling(window=window).mean()
        features['rolling_std'] = data.rolling(window=window).std()
        features['rolling_skew'] = features['returns'].rolling(window=window).skew()
        features['rolling_kurt'] = features['returns'].rolling(window=window).kurt()
        
        # Volatility measures
        features['volatility'] = features['log_returns'].rolling(window=window).std()
        features['range'] = (data.rolling(window=window).max() - 
                            data.rolling(window=window).min()) / data
        
        # Drop NaN
        features = features.dropna()
        
        return features
    
    def fit(self, features: pd.DataFrame):
        """
        Fit Gaussian Mixture Model
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit GMM
        self.gmm = GaussianMixture(n_components=self.n_regimes, 
                                  covariance_type='full',
                                  random_state=42)
        self.gmm.fit(scaled_features)
        
        # Predict regimes
        regimes = self.gmm.predict(scaled_features)
        features['regime'] = regimes
        
        # Calculate regime statistics
        self.regime_stats = {}
        for regime in range(self.n_regimes):
            regime_data = features[features['regime'] == regime]
            self.regime_stats[regime] = {
                'count': len(regime_data),
                'mean_return': regime_data['returns'].mean(),
                'volatility': regime_data['volatility'].mean(),
                'probability': len(regime_data) / len(features)
            }
        
        return features
    
    def predict_regime(self, features: pd.DataFrame):
        """
        Predict regime for new data
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            Predicted regimes
        """
        if self.gmm is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scaled_features = self.scaler.transform(features)
        return self.gmm.predict(scaled_features)
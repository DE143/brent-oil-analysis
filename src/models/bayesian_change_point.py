"""
Bayesian change point models for Brent oil price analysis
"""

import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class BayesianChangePointModel:
    """
    Bayesian change point detection model for time series
    """
    
    def __init__(self, n_changepoints: int = 3, model_type: str = 'mean_var'):
        """
        Initialize the model
        
        Parameters:
        -----------
        n_changepoints : int
            Number of change points to detect
        model_type : str
            Type of model: 'mean', 'mean_var', or 'regime'
        """
        self.n_changepoints = n_changepoints
        self.model_type = model_type
        self.model = None
        self.trace = None
        self.summary = None
        
    def build_mean_change_model(self, data: np.ndarray):
        """
        Build model with changing mean only
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        """
        n_obs = len(data)
        
        with pm.Model() as model:
            # Priors for change points
            tau = pm.DiscreteUniform('tau', 
                                     lower=0, 
                                     upper=n_obs-1, 
                                     shape=self.n_changepoints)
            
            # Sort change points
            tau_sorted = pm.Deterministic('tau_sorted', pm.math.sort(tau))
            
            # Segment means
            segment_means = []
            for i in range(self.n_changepoints + 1):
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = tau_sorted[i-1]
                
                if i == self.n_changepoints:
                    end_idx = n_obs
                else:
                    end_idx = tau_sorted[i]
                
                # Segment mean prior
                segment_data = data[int(start_idx):int(end_idx)]
                if len(segment_data) > 0:
                    mu_mean = np.mean(segment_data)
                    mu_sigma = np.std(segment_data) * 2
                else:
                    mu_mean = np.mean(data)
                    mu_sigma = np.std(data) * 2
                
                mu = pm.Normal(f'mu_{i}', mu=mu_mean, sigma=mu_sigma)
                segment_means.append(mu)
            
            # Shared variance
            sigma = pm.HalfNormal('sigma', sigma=np.std(data))
            
            # Likelihood
            idx = np.arange(n_obs)
            mu_combined = segment_means[0]
            
            for i in range(1, self.n_changepoints + 1):
                mask = idx >= tau_sorted[i-1]
                mu_combined = pm.math.switch(mask, segment_means[i], mu_combined)
            
            likelihood = pm.Normal('likelihood', 
                                   mu=mu_combined, 
                                   sigma=sigma, 
                                   observed=data)
        
        self.model = model
        return model
    
    def build_mean_var_change_model(self, data: np.ndarray):
        """
        Build model with changing mean and variance
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        """
        n_obs = len(data)
        
        with pm.Model() as model:
            # Priors for change points
            tau = pm.DiscreteUniform('tau', 
                                     lower=0, 
                                     upper=n_obs-1, 
                                     shape=self.n_changepoints)
            
            # Sort change points
            tau_sorted = pm.Deterministic('tau_sorted', pm.math.sort(tau))
            
            # Segment parameters
            segment_means = []
            segment_sigmas = []
            
            for i in range(self.n_changepoints + 1):
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = tau_sorted[i-1]
                
                if i == self.n_changepoints:
                    end_idx = n_obs
                else:
                    end_idx = tau_sorted[i]
                
                # Segment statistics
                segment_data = data[int(start_idx):int(end_idx)]
                if len(segment_data) > 0:
                    mu_mean = np.mean(segment_data)
                    mu_sigma = np.std(segment_data) * 2
                    sigma_sigma = np.std(segment_data)
                else:
                    mu_mean = np.mean(data)
                    mu_sigma = np.std(data) * 2
                    sigma_sigma = np.std(data)
                
                # Priors
                mu = pm.Normal(f'mu_{i}', mu=mu_mean, sigma=mu_sigma)
                sigma = pm.HalfNormal(f'sigma_{i}', sigma=sigma_sigma)
                
                segment_means.append(mu)
                segment_sigmas.append(sigma)
            
            # Likelihood
            idx = np.arange(n_obs)
            mu_combined = segment_means[0]
            sigma_combined = segment_sigmas[0]
            
            for i in range(1, self.n_changepoints + 1):
                mask = idx >= tau_sorted[i-1]
                mu_combined = pm.math.switch(mask, segment_means[i], mu_combined)
                sigma_combined = pm.math.switch(mask, segment_sigmas[i], sigma_combined)
            
            likelihood = pm.Normal('likelihood', 
                                   mu=mu_combined, 
                                   sigma=sigma_combined, 
                                   observed=data)
        
        self.model = model
        return model
    
    def build_regime_switching_model(self, data: np.ndarray, n_regimes: int = 2):
        """
        Build regime switching model
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        n_regimes : int
            Number of regimes
        """
        n_obs = len(data)
        
        with pm.Model() as model:
            # Regime probabilities
            regime_probs = pm.Dirichlet('regime_probs', a=np.ones(n_regimes))
            
            # Regime parameters
            regime_means = pm.Normal('regime_means', 
                                     mu=np.mean(data), 
                                     sigma=np.std(data)*2, 
                                     shape=n_regimes)
            regime_sigmas = pm.HalfNormal('regime_sigmas', 
                                          sigma=np.std(data), 
                                          shape=n_regimes)
            
            # Hidden Markov states
            states = pm.Categorical('states', 
                                    p=regime_probs, 
                                    shape=n_obs)
            
            # Likelihood
            likelihood = pm.Normal('likelihood', 
                                   mu=regime_means[states], 
                                   sigma=regime_sigmas[states], 
                                   observed=data)
        
        self.model = model
        return model
    
    def fit(self, data: np.ndarray, draws: int = 3000, tune: int = 1000, 
            chains: int = 4, target_accept: float = 0.8):
        """
        Fit the model using MCMC
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        draws : int
            Number of posterior samples
        tune : int
            Number of tuning samples
        chains : int
            Number of MCMC chains
        target_accept : float
            Target acceptance rate
        """
        # Build model based on type
        if self.model_type == 'mean':
            self.build_mean_change_model(data)
        elif self.model_type == 'mean_var':
            self.build_mean_var_change_model(data)
        elif self.model_type == 'regime':
            self.build_regime_switching_model(data)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Run MCMC
        with self.model:
            self.trace = pm.sample(draws=draws,
                                  tune=tune,
                                  chains=chains,
                                  target_accept=target_accept,
                                  return_inferencedata=True,
                                  progressbar=True)
        
        # Compute summary statistics
        self.summary = az.summary(self.trace)
        
        return self.trace
    
    def get_change_points(self, data_dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """
        Extract change points from trace
        
        Parameters:
        -----------
        data_dates : pd.DatetimeIndex
            Dates corresponding to data indices
            
        Returns:
        --------
        List[pd.Timestamp]
            Detected change point dates
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model_type == 'regime':
            # For regime switching, find points where state changes
            states = self.trace.posterior['states'].mean(dim=['chain', 'draw']).values
            change_points = []
            
            for i in range(1, len(states)):
                if states[i] != states[i-1]:
                    change_points.append(data_dates[i])
            
            return change_points
        
        else:
            # For change point models
            tau_samples = self.trace.posterior['tau_sorted'].values
            
            # Get median change points
            median_tau = np.median(tau_samples, axis=(0, 1)).astype(int)
            
            # Convert to dates
            change_point_dates = [data_dates[idx] for idx in median_tau]
            
            return change_point_dates
    
    def get_regime_parameters(self) -> Dict:
        """
        Get estimated regime parameters
        
        Returns:
        --------
        Dict
            Dictionary with regime parameters
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model_type == 'regime':
            params = {
                'regime_means': self.trace.posterior['regime_means'].mean(dim=['chain', 'draw']).values,
                'regime_sigmas': self.trace.posterior['regime_sigmas'].mean(dim=['chain', 'draw']).values,
                'regime_probs': self.trace.posterior['regime_probs'].mean(dim=['chain', 'draw']).values,
            }
            return params
        
        else:
            params = {}
            for i in range(self.n_changepoints + 1):
                mu_key = f'mu_{i}'
                sigma_key = f'sigma_{i}' if self.model_type == 'mean_var' else 'sigma'
                
                if mu_key in self.trace.posterior:
                    params[f'mean_{i}'] = self.trace.posterior[mu_key].mean(dim=['chain', 'draw']).values
                
                if sigma_key in self.trace.posterior:
                    params[f'sigma_{i}'] = self.trace.posterior[sigma_key].mean(dim=['chain', 'draw']).values
            
            return params
    
    def plot_trace(self):
        """Plot MCMC trace"""
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        az.plot_trace(self.trace)
        plt.tight_layout()
        plt.show()
    
    def plot_posterior(self, var_name: str = 'tau_sorted'):
        """Plot posterior distribution of specified variable"""
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        az.plot_posterior(self.trace, var_names=[var_name])
        plt.tight_layout()
        plt.show()
"""
Benchmark Portfolio Implementation
"""

import pandas as pd
import numpy as np
from typing import Dict

class BenchmarkPortfolio:
    """Simple benchmark portfolio (60% SPY / 40% BND)."""
    
    def __init__(self, spy_weight: float = 0.6, bnd_weight: float = 0.4):
        self.weights = {
            'SPY': spy_weight,
            'BND': bnd_weight,
            'TSLA': 0.0  # No TSLA in benchmark
        }
    
    def get_weights(self) -> Dict[str, float]:
        """Return benchmark weights."""
        return self.weights
    
    def simulate_performance(self, returns_data: pd.DataFrame) -> pd.Series:
        """Simulate benchmark portfolio performance."""
        # Convert weights to array matching returns_data columns
        weight_array = np.array([self.weights.get(col, 0) for col in returns_data.columns])
        
        # Calculate portfolio returns
        portfolio_returns = (returns_data * weight_array).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        return cumulative_returns
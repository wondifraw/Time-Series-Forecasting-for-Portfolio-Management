"""
Base Model Interface for Time Series Forecasting
Provides common interface and functionality for all forecasting models
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pickle
import os

class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, symbols: list = None):
        """Initialize base forecaster."""
        self.symbols = symbols or ['TSLA']
        self.is_fitted = False
        self.model = None
        
    @abstractmethod
    def fit(self, train_data: pd.Series, **kwargs) -> bool:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        pass
    
    def save_model(self, filepath: str) -> bool:
        """Save the fitted model."""
        if not self.is_fitted:
            print("❌ No fitted model to save")
            return False
            
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✅ Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a saved model."""
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.is_fitted = True
            print(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
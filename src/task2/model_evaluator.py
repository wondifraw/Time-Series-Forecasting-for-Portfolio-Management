"""
Model Evaluation Utilities
Provides standardized evaluation metrics and visualization for forecasting models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Optional

class ModelEvaluator:
    """Standardized model evaluation utilities."""
    
    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate standard forecasting metrics."""
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual_vals = np.array(actual[:min_len]).flatten()
        pred_vals = np.array(predicted[:min_len]).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(actual_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
        mape = np.mean(np.abs((actual_vals - pred_vals) / actual_vals)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    @staticmethod
    def calculate_residuals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Calculate residuals between actual and predicted values."""
        min_len = min(len(actual), len(predicted))
        actual_vals = np.array(actual[:min_len]).flatten()
        pred_vals = np.array(predicted[:min_len]).flatten()
        
        return actual_vals - pred_vals
    
    @staticmethod
    def plot_predictions(actual: np.ndarray, predicted: np.ndarray, 
                        title: str = "Model Predictions", 
                        data_index: Optional[pd.DatetimeIndex] = None):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(12, 6))
        
        # Create index if not provided
        if data_index is None:
            data_index = range(len(actual))
        else:
            data_index = data_index[-len(actual):]
        
        plt.plot(data_index, actual, label='Actual', color='green', linewidth=2)
        plt.plot(data_index, predicted, label='Predicted', color='red', linewidth=2)
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_residuals(residuals: np.ndarray, title: str = "Residuals Analysis"):
        """Plot residuals analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals over time
        axes[0].plot(residuals, alpha=0.7)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_title(f'{title} - Over Time')
        axes[0].set_ylabel('Residuals')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_title(f'{title} - Distribution')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compare_metrics(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare metrics across multiple models."""
        return pd.DataFrame(metrics_dict).T
    
    @staticmethod
    def rank_models(metrics_df: pd.DataFrame) -> pd.Series:
        """Rank models by average performance across metrics."""
        ranks = metrics_df.rank()
        return ranks.mean(axis=1).sort_values()
    
    @staticmethod
    def generate_evaluation_report(actual: np.ndarray, predicted: np.ndarray, 
                                 model_name: str = "Model") -> Dict:
        """Generate comprehensive evaluation report."""
        metrics = ModelEvaluator.calculate_metrics(actual, predicted)
        residuals = ModelEvaluator.calculate_residuals(actual, predicted)
        
        report = {
            'model_name': model_name,
            'metrics': metrics,
            'residual_stats': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals)
            },
            'sample_size': len(actual)
        }
        
        return report
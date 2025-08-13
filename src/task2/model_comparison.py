"""
Model Comparison Module for ARIMA vs LSTM
Refactored with modular design principles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

from .model_evaluator import ModelEvaluator

class ModelComparison:
    """Compare ARIMA and LSTM model performance."""
    
    def __init__(self):
        """Initialize model comparison."""
        self.results = {}
        
    def add_model_results(self, model_name: str, predictions: np.ndarray, 
                         actual: np.ndarray, metrics: Dict[str, float] = None):
        """Add model results for comparison."""
        self.results[model_name] = {
            'predictions': predictions,
            'actual': actual,
            'metrics': metrics or ModelEvaluator.calculate_metrics(actual, predictions)
        }
        

    
    def compare_metrics(self):
        """Compare metrics across all models."""
        if not self.results:
            print("❌ No model results to compare")
            return None
            
        comparison_df = pd.DataFrame({
            model: result['metrics'] 
            for model, result in self.results.items()
        }).T
        
        return comparison_df
    
    def plot_predictions_comparison(self, data_index: Optional[pd.DatetimeIndex] = None):
        """Plot predictions from all models for comparison."""
        if not self.results:
            print("❌ No model results to plot")
            return
            
        plt.figure(figsize=(15, 8))
        
        # Get actual values (assuming they're the same across models)
        first_model = list(self.results.keys())[0]
        actual = self.results[first_model]['actual']
        
        # Create index if not provided
        if data_index is None:
            data_index = range(len(actual))
        else:
            # Adjust index length to match actual data
            data_index = data_index[-len(actual):]
        
        # Plot actual values
        plt.plot(data_index, actual, label='Actual', 
                color='black', linewidth=2, alpha=0.8)
        
        # Plot predictions from each model
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, result) in enumerate(self.results.items()):
            predictions = result['predictions']
            color = colors[i % len(colors)]
            
            # Ensure predictions match actual length
            pred_len = min(len(predictions), len(actual))
            pred_index = data_index[-pred_len:]
            
            plt.plot(pred_index, predictions[:pred_len], 
                    label=f'{model_name} Predictions', 
                    color=color, linewidth=2, alpha=0.7)
        
        plt.title('Model Predictions Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self):
        """Plot metrics comparison as bar chart."""
        if not self.results:
            print("❌ No model results to plot")
            return
            
        comparison_df = self.compare_metrics()
        if comparison_df is None:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['MAE', 'RMSE', 'MAPE']
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                comparison_df[metric].plot(kind='bar', ax=axes[i], color=colors[i])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals_analysis(self):
        """Plot residuals analysis for all models using ModelEvaluator."""
        if not self.results:
            print("❌ No model results to analyze")
            return
            
        n_models = len(self.results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Residuals Analysis', fontsize=16, fontweight='bold')
        
        for i, (model_name, result) in enumerate(self.results.items()):
            actual = np.array(result['actual']).flatten()
            predictions = np.array(result['predictions']).flatten()
            
            # Calculate residuals using ModelEvaluator
            residuals = ModelEvaluator.calculate_residuals(actual, predictions)
            
            # Residuals over time
            axes[0, i].plot(residuals, alpha=0.7)
            axes[0, i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, i].set_title(f'{model_name} - Residuals Over Time')
            axes[0, i].set_ylabel('Residuals')
            axes[0, i].grid(True, alpha=0.3)
            
            # Residuals distribution
            axes[1, i].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, i].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[1, i].set_title(f'{model_name} - Residuals Distribution')
            axes[1, i].set_xlabel('Residuals')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_comparison_report(self) -> Optional[Dict[str, Any]]:
        """Generate comprehensive comparison report using ModelEvaluator."""
        if not self.results:
            print("❌ No model results to report")
            return None
            
        print("📊 MODEL COMPARISON REPORT")
        print("=" * 50)
        
        # Metrics comparison
        comparison_df = self.compare_metrics()
        print("\n📈 Performance Metrics:")
        print(comparison_df.round(4))
        
        # Best model for each metric
        print("\n🏆 Best Model by Metric:")
        for metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmin()
            best_value = comparison_df.loc[best_model, metric]
            print(f"{metric}: {best_model} ({best_value:.4f})")
        
        # Overall ranking using ModelEvaluator
        print("\n🥇 Overall Ranking (by average rank):")
        avg_ranks = ModelEvaluator.rank_models(comparison_df)
        
        for i, (model, avg_rank) in enumerate(avg_ranks.items(), 1):
            print(f"{i}. {model} (Average Rank: {avg_rank:.2f})")
        
        # Statistical summary using ModelEvaluator
        print("\n📊 Statistical Summary:")
        for model_name, result in self.results.items():
            actual = np.array(result['actual']).flatten()
            predictions = np.array(result['predictions']).flatten()
            
            residuals = ModelEvaluator.calculate_residuals(actual, predictions)
            
            print(f"\n{model_name}:")
            print(f"  Mean Residual: {np.mean(residuals):.4f}")
            print(f"  Std Residual: {np.std(residuals):.4f}")
            print(f"  Min Residual: {np.min(residuals):.4f}")
            print(f"  Max Residual: {np.max(residuals):.4f}")
        
        return {
            'metrics_comparison': comparison_df,
            'best_models': {metric: comparison_df[metric].idxmin() 
                          for metric in comparison_df.columns},
            'overall_ranking': avg_ranks
        }
    
    def export_results(self, filename: str = 'model_comparison_results.csv') -> bool:
        """Export comparison results to CSV."""
        if not self.results:
            print("❌ No results to export")
            return False
            
        try:
            comparison_df = self.compare_metrics()
            comparison_df.to_csv(filename)
            print(f"✅ Results exported to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error exporting results: {str(e)}")
            return False
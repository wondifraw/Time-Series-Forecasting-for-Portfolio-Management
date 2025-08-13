"""
Forecasting Pipeline - Orchestrates ARIMA and LSTM models
Refactored with modular design principles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional

from .arima_model import ARIMAForecaster
from .lstm_model import LSTMForecaster
from .model_comparison import ModelComparison
from .data_handler import TimeSeriesDataHandler
from .config import ModelConfig, ConfigManager
from .model_factory import ModelFactory

class ForecastingPipeline:
    """Streamlined forecasting pipeline with modular design."""
    
    def __init__(self, config: ModelConfig = None, symbols: list = None):
        """Initialize pipeline with lazy model creation."""
        self.config = config or ConfigManager.get_default_config()
        self.symbols = symbols or self.config.data.symbols
        self.data_handler = TimeSeriesDataHandler(self.symbols)
        self.comparison = ModelComparison()
        
        # Lazy initialization
        self._models = {}
        self.data = None
        self.results = {}
        
    def _get_model(self, model_type: str):
        """Get or create model instance."""
        if model_type not in self._models:
            self._models[model_type] = ModelFactory.create_model(model_type, self.config, symbols=self.symbols)
        return self._models[model_type]
    
    def load_data(self, symbol: str = 'TSLA') -> bool:
        """Load and validate data."""
        self.data = self.data_handler.load_processed_data(symbol)
        return self.data is not None
    
    def _run_model_analysis(self, model_type: str) -> bool:
        """Generic model analysis runner."""
        model = self._get_model(model_type)
        
        if model_type == 'ARIMA':
            train_data, test_data = model.prepare_data(self.data, self.config.data.test_size)
            model.check_stationarity(train_data)
            success = model.fit(train_data)
            if success:
                predictions = model.predict(len(test_data))
                actual_data = test_data
                model.save_model()  # Auto-save to models folder
        else:  # LSTM
            X_train, X_test, y_train, y_test = model.prepare_data(self.data, self.config.data.test_size)
            success = model.fit(X_train, y_train)
            if success:
                predictions = model.predict(X_test)
                actual_data = y_test
                model.save_model()  # Auto-save to models folder
        
        if not success or predictions is None:
            return False
        
        metrics = model.evaluate(actual_data, predictions)
        self.results[model_type] = {
            'model': model.model if hasattr(model, 'model') else model.fitted_model,
            'predictions': predictions,
            'metrics': metrics
        }
        
        self.comparison.add_model_results(model_type, predictions, actual_data, metrics)
        return True
    
    def run_models(self, model_types: list = None) -> Dict[str, bool]:
        """Run analysis for specified models."""
        if self.data is None:
            print("❌ Load data first")
            return {}
        
        model_types = model_types or ['ARIMA', 'LSTM']
        results = {}
        
        for model_type in model_types:
            print(f"\n🔧 {model_type} Analysis")
            try:
                success = self._run_model_analysis(model_type)
                results[model_type] = success
                if success:
                    metrics = self.results[model_type]['metrics']
                    print(f"📊 {model_type} - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            except Exception as e:
                print(f"❌ {model_type} failed: {str(e)}")
                results[model_type] = False
        
        return results
    
    def compare_models(self, plot: bool = True) -> Optional[Dict[str, Any]]:
        """Compare model results."""
        if len(self.results) < 2:
            return None
        
        report = self.comparison.generate_comparison_report()
        
        if plot:
            self.comparison.plot_metrics_comparison()
            self.comparison.plot_predictions_comparison(self.data.index)
        
        return report
    
    def plot_results(self):
        """Plot analysis results."""
        if not self.results:
            return
        
        try:
            n_models = len(self.results)
            fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 6))
            
            # Ensure axes is always a list
            if n_models == 0:
                axes = [axes]
            elif not hasattr(axes, '__len__'):
                axes = [axes]
            
            # Original data - handle both Series and array data
            data_values = self.data.values if hasattr(self.data, 'values') else self.data
            data_index = self.data.index if hasattr(self.data, 'index') else range(len(data_values))
            
            axes[0].plot(data_index, data_values, 'b-', alpha=0.7)
            axes[0].set_title('Original Data')
            axes[0].grid(True, alpha=0.3)
            
            # Model results
            for i, (model_name, result) in enumerate(self.results.items(), 1):
                predictions = result['predictions']
                
                # Plot data and predictions
                axes[i].plot(data_index, data_values, 'b-', alpha=0.5, label='Actual')
                
                # Adjust prediction index based on model type
                if model_name == 'ARIMA':
                    split_idx = int(len(self.data) * (1 - self.config.data.test_size))
                    pred_index = data_index[split_idx:split_idx + len(predictions)]
                else:  # LSTM
                    split_idx = int(len(self.data) * (1 - self.config.data.test_size))
                    try:
                        seq_len = self._get_model('LSTM').sequence_length
                        start_idx = split_idx + seq_len
                        pred_index = data_index[start_idx:start_idx + len(predictions)]
                    except:
                        pred_index = data_index[split_idx:split_idx + len(predictions)]
                
                axes[i].plot(pred_index, predictions, 'r-', linewidth=2, label='Predicted')
                axes[i].set_title(f'{model_name} Results')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Plotting failed: {str(e)}")
            print("📊 Results available but plotting skipped")
    
    def run(self, symbol: str = 'TSLA', models: list = None, plot: bool = True) -> Optional[Dict[str, Any]]:
        """Run complete pipeline."""
        print("🚀 FORECASTING PIPELINE")
        
        if not self.load_data(symbol):
            return None
        
        model_results = self.run_models(models)
        successful_models = [k for k, v in model_results.items() if v]
        
        comparison = None
        if len(successful_models) >= 2:
            comparison = self.compare_models(plot)
        
        if plot:
            self.plot_results()
        
        self.export_results()
        print("\n✅ Pipeline completed")
        
        return {
            'data': self.data,
            'results': self.results,
            'comparison': comparison
        }
    
    def export_results(self, output_dir: str = 'results/task2'):
        """Export all results to files."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Export comparison metrics
            if len(self.results) >= 2:
                self.comparison.export_results(f'{output_dir}/model_comparison.csv')
            
            # Export individual model results
            for model_name, result in self.results.items():
                if 'predictions' in result:
                    pred_df = pd.DataFrame({
                        'predictions': result['predictions']
                    })
                    pred_df.to_csv(f'{output_dir}/{model_name.lower()}_predictions.csv')
            
            print(f"✅ Results exported to {output_dir}/")
            
        except Exception as e:
            print(f"❌ Error exporting results: {str(e)}")
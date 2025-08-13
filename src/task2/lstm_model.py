"""
LSTM Model Implementation for Time Series Forecasting
Refactored with modular design principles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseForecaster
from .data_handler import TimeSeriesDataHandler
from .model_evaluator import ModelEvaluator
from .config import ModelConfig, LSTMConfig

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")

class LSTMForecaster(BaseForecaster):
    """LSTM model for time series forecasting with modular design."""
    
    def __init__(self, config: ModelConfig = None, symbols: list = None):
        """Initialize LSTM forecaster."""
        super().__init__(symbols)
        self.config = config.lstm if config else LSTMConfig()
        self.data_handler = TimeSeriesDataHandler(self.symbols)
        self.sequence_length = self.config.sequence_length
        
    def load_data(self, symbol: str = 'TSLA'):
        """Load data using data handler."""
        return self.data_handler.load_processed_data(symbol)
    
    def prepare_data(self, data, test_size: float = None):
        """Prepare data for LSTM training using data handler."""
        if test_size is None:
            test_size = 0.2
        return self.data_handler.prepare_lstm_data(data, self.sequence_length, test_size)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture using config."""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available. Cannot build LSTM model.")
            return None
        
        model = Sequential()
        
        # Add LSTM layers based on config
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(self.config.dropout_rate))
        
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=self.config.learning_rate), loss='mse')
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Fit LSTM model to training data."""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available. Cannot train LSTM model.")
            return False
        
        if X_train is None or y_train is None:
            print("❌ No training data provided")
            return False
        
        try:
            # Reshape input for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            # Build model
            input_shape = (X_train.shape[1], 1)
            self.model = self.build_model(input_shape)
            
            if self.model is None:
                return False
            
            # Prepare validation data if provided
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                validation_data = (X_val, y_val)
            
            # Train model
            print(f"Training LSTM model for {self.config.epochs} epochs...")
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=validation_data,
                verbose=0
            )
            
            self.is_fitted = True
            print("✅ LSTM model trained successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error training LSTM model: {str(e)}")
            return False
    
    def predict(self, X_test, **kwargs):
        """Generate predictions using trained LSTM model."""
        if not self.is_fitted or self.model is None:
            print("❌ Model not fitted. Call fit() first.")
            return None
        
        try:
            # Reshape input for prediction
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Make predictions
            predictions_scaled = self.model.predict(X_test, verbose=0)
            
            # Inverse transform to original scale
            predictions = self.data_handler.inverse_transform(predictions_scaled.flatten())
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error making predictions: {str(e)}")
            return None
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance using standardized metrics."""
        if y_true is None or y_pred is None:
            return None
        
        # Inverse transform true values
        y_true_original = self.data_handler.inverse_transform(y_true.flatten())
        
        return ModelEvaluator.calculate_metrics(y_true_original, y_pred)
    
    def plot_results(self, data, train_size, y_test, predictions):
        """Plot training data, test data, and predictions."""
        if any(x is None for x in [data, y_test, predictions]):
            print("❌ Missing data for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create index for plotting
        data_index = data.index
        
        # Plot full data
        plt.plot(data_index, data.values, label='Full Data', alpha=0.5, color='gray')
        
        # Calculate test data start index
        test_start_idx = train_size + self.sequence_length
        test_index = data_index[test_start_idx:test_start_idx + len(predictions)]
        
        # Plot actual test values
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        actual_test_index = data_index[test_start_idx:test_start_idx + len(y_test_original)]
        plt.plot(actual_test_index, y_test_original, 
                label='Actual', color='green', linewidth=2)
        
        # Plot predictions
        plt.plot(test_index, predictions, 
                label='LSTM Predictions', color='red', linewidth=2)
        
        plt.title('LSTM Model Prediction Results')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run complete LSTM analysis pipeline."""
        print("🧠 LSTM FORECASTING ANALYSIS")
        print("=" * 40)
        
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available. Please install: pip install tensorflow")
            return None
        
        # Load data
        data = self.load_data()
        if data is None:
            return None
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        if any(x is None for x in [X_train, X_test, y_train, y_test]):
            return None
        
        print(f"\n📊 Data Preparation:")
        print(f"Training sequences: {len(X_train)}")
        print(f"Testing sequences: {len(X_test)}")
        print(f"Sequence length: {self.sequence_length}")
        
        # Fit model
        print(f"\n⚙️ Model Training:")
        success = self.fit(X_train, y_train, epochs=50)
        if not success:
            return None
        
        # Generate predictions
        print(f"\n📈 Generating Predictions:")
        predictions = self.predict(X_test)
        if predictions is None:
            return None
        
        # Evaluate performance
        print(f"\n📊 Model Evaluation:")
        metrics = self.evaluate(y_test, predictions)
        if metrics:
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        # Plot results
        train_size = len(X_train)
        self.plot_results(data, train_size, y_test, predictions)
        
        return {
            'model': self.model,
            'predictions': predictions,
            'metrics': metrics,
            'y_test': y_test,
            'scaler': self.scaler
        }
    
    def save_model(self, model_path: str = None):
        """Save the fitted LSTM model."""
        if not self.is_fitted or self.model is None:
            print("❌ No fitted model to save")
            return False
        
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available. Cannot save LSTM model.")
            return False
            
        try:
            if model_path is None:
                os.makedirs('models', exist_ok=True)
                model_path = 'models/lstm_model.h5'
            else:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            self.model.save(model_path)
            print(f"✅ LSTM model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving LSTM model: {str(e)}")
            return False
    
    def load_model(self, model_path: str = None):
        """Load a saved LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available. Cannot load LSTM model.")
            return False
        
        if model_path is None:
            model_path = 'models/lstm_model.h5'
        
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
            self.is_fitted = True
            print(f"✅ LSTM model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading LSTM model: {str(e)}")
            return False
"""
Unit Tests for Task 2: Time Series Forecasting Models
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from task2.arima_model import ARIMAForecaster
from task2.lstm_model import LSTMForecaster
from task2.model_comparison import ModelComparison
import warnings
warnings.filterwarnings('ignore')


class TestARIMAModel(unittest.TestCase):
    """Test cases for ARIMA model implementation."""
    
    def setUp(self):
        """Set up test data."""
        self.arima_model = ARIMAForecaster()
        
        # Create synthetic time series data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        trend = np.linspace(100, 200, 1000)
        noise = np.random.normal(0, 5, 1000)
        self.test_data = pd.Series(trend + noise, index=dates)
    
    def test_data_preparation(self):
        """Test data preparation and splitting."""
        train_data, test_data = self.arima_model.prepare_data(self.test_data, test_size=0.2)
        
        self.assertEqual(len(train_data), 800)
        self.assertEqual(len(test_data), 200)
        self.assertTrue(train_data.index[-1] < test_data.index[0])  # Chronological split
    
    def test_stationarity_check(self):
        """Test stationarity checking functionality."""
        # Test with non-stationary data (trend)
        is_stationary = self.arima_model.check_stationarity(self.test_data)
        self.assertIsInstance(is_stationary, bool)
        
        # Test with stationary data (differences)
        stationary_data = self.test_data.diff().dropna()
        is_stationary_diff = self.arima_model.check_stationarity(stationary_data)
        self.assertIsInstance(is_stationary_diff, bool)
    
    def test_model_fitting(self):
        """Test ARIMA model fitting."""
        train_data, _ = self.arima_model.prepare_data(self.test_data, test_size=0.2)
        
        # Test with manual parameters
        self.arima_model.fit_model(train_data, order=(1, 1, 1))
        self.assertIsNotNone(self.arima_model.fitted_model)
    
    def test_forecasting(self):
        """Test forecasting functionality."""
        train_data, test_data = self.arima_model.prepare_data(self.test_data, test_size=0.2)
        self.arima_model.fit_model(train_data, order=(1, 1, 1))
        
        forecast, conf_int = self.arima_model.forecast(len(test_data))
        
        self.assertEqual(len(forecast), len(test_data))
        self.assertIsNotNone(conf_int)
    
    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        train_data, test_data = self.arima_model.prepare_data(self.test_data, test_size=0.2)
        self.arima_model.fit_model(train_data, order=(1, 1, 1))
        
        forecast, _ = self.arima_model.forecast(len(test_data))
        metrics = self.arima_model.evaluate_model(test_data, forecast)
        
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAPE', metrics)
        self.assertTrue(all(v >= 0 for v in metrics.values()))


class TestLSTMModel(unittest.TestCase):
    """Test cases for LSTM model implementation."""
    
    def setUp(self):
        """Set up test data."""
        self.lstm_model = LSTMForecaster(sequence_length=30)
        
        # Create synthetic time series data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        trend = np.linspace(100, 200, 500)
        noise = np.random.normal(0, 5, 500)
        self.test_data = pd.Series(trend + noise, index=dates)
    
    def test_data_preparation(self):
        """Test LSTM data preparation."""
        X_train, y_train, X_test, y_test, train_data, test_data = self.lstm_model.prepare_data(
            self.test_data, test_size=0.2
        )
        
        # Check shapes
        self.assertEqual(X_train.shape[1], self.lstm_model.sequence_length)
        self.assertEqual(len(y_train), X_train.shape[0])
        self.assertEqual(X_test.shape[1], self.lstm_model.sequence_length)
        self.assertEqual(len(y_test), X_test.shape[0])
        
        # Check chronological split
        self.assertTrue(train_data.index[-1] < test_data.index[0])
    
    def test_sequence_creation(self):
        """Test sequence creation for LSTM."""
        # Create simple test data
        data = np.array([[i] for i in range(100)])
        X, y = self.lstm_model._create_sequences(data)
        
        expected_samples = len(data) - self.lstm_model.sequence_length
        self.assertEqual(len(X), expected_samples)
        self.assertEqual(len(y), expected_samples)
        self.assertEqual(X.shape[1], self.lstm_model.sequence_length)
    
    def test_model_building(self):
        """Test LSTM model architecture building."""
        self.lstm_model.build_model(lstm_units=32, dropout_rate=0.2)
        
        self.assertIsNotNone(self.lstm_model.model)
        self.assertEqual(len(self.lstm_model.model.layers), 7)  # 3 LSTM + 3 Dropout + 2 Dense
    
    def test_model_evaluation(self):
        """Test LSTM model evaluation."""
        # Create dummy predictions and actual values
        actual = np.array([100, 101, 102, 103, 104])
        predicted = np.array([99, 102, 101, 104, 103])
        
        metrics = self.lstm_model.evaluate_model(actual, predicted)
        
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAPE', metrics)
        self.assertTrue(all(v >= 0 for v in metrics.values()))


class TestModelComparison(unittest.TestCase):
    """Test cases for model comparison functionality."""
    
    def setUp(self):
        """Set up test data and comparison object."""
        self.comparison = ModelComparison()
        
        # Create synthetic Tesla-like data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        # Simulate stock price with trend and volatility
        returns = np.random.normal(0.001, 0.02, 1000)  # Daily returns
        prices = [100]  # Starting price
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        self.test_data = pd.Series(prices[1:], index=dates)
    
    def test_data_loading_simulation(self):
        """Test data loading simulation (without actual file)."""
        # This would test the data loading logic
        self.assertIsNotNone(self.test_data)
        self.assertEqual(len(self.test_data), 1000)
        self.assertTrue(all(self.test_data > 0))  # Stock prices should be positive
    
    def test_comparison_structure(self):
        """Test comparison object structure."""
        self.assertIsInstance(self.comparison.arima_model, ARIMAForecaster)
        self.assertIsInstance(self.comparison.lstm_model, LSTMForecaster)
        self.assertIsInstance(self.comparison.results, dict)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_insufficient_data_arima(self):
        """Test ARIMA with insufficient data."""
        arima_model = ARIMAForecaster()
        
        # Very small dataset
        small_data = pd.Series([1, 2, 3, 4, 5], 
                              index=pd.date_range('2020-01-01', periods=5))
        
        train_data, test_data = arima_model.prepare_data(small_data, test_size=0.2)
        
        # Should handle small datasets gracefully
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(test_data), 0)
    
    def test_insufficient_data_lstm(self):
        """Test LSTM with insufficient data."""
        lstm_model = LSTMForecaster(sequence_length=10)
        
        # Dataset smaller than sequence length
        small_data = pd.Series([1, 2, 3, 4, 5], 
                              index=pd.date_range('2020-01-01', periods=5))
        
        X_train, y_train, X_test, y_test, _, _ = lstm_model.prepare_data(small_data, test_size=0.2)
        
        # Should handle gracefully (may result in empty sequences)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
    
    def test_missing_values(self):
        """Test handling of missing values."""
        # Create data with missing values
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100)
        values[10:15] = np.nan  # Insert missing values
        
        data_with_nan = pd.Series(values, index=dates)
        
        # Test that models can handle NaN values
        arima_model = ARIMAForecaster()
        train_data, test_data = arima_model.prepare_data(data_with_nan, test_size=0.2)
        
        # Should not contain NaN after preparation
        self.assertFalse(train_data.isnull().any())
        self.assertFalse(test_data.isnull().any())


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metric calculations."""
    
    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        from sklearn.metrics import mean_absolute_error
        expected_mae = mean_absolute_error(actual, predicted)
        
        # Test through ARIMA model
        arima_model = ARIMAForecaster()
        metrics = arima_model.evaluate_model(
            pd.Series(actual), predicted
        )
        
        self.assertAlmostEqual(metrics['MAE'], expected_mae, places=6)
    
    def test_rmse_calculation(self):
        """Test Root Mean Square Error calculation."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        from sklearn.metrics import mean_squared_error
        expected_rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # Test through LSTM model
        lstm_model = LSTMForecaster()
        metrics = lstm_model.evaluate_model(actual, predicted)
        
        self.assertAlmostEqual(metrics['RMSE'], expected_rmse, places=6)
    
    def test_mape_calculation(self):
        """Test Mean Absolute Percentage Error calculation."""
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([110, 190, 310, 390, 510])
        
        expected_mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Test through ARIMA model
        arima_model = ARIMAForecaster()
        metrics = arima_model.evaluate_model(
            pd.Series(actual), predicted
        )
        
        self.assertAlmostEqual(metrics['MAPE'], expected_mape, places=6)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestARIMAModel))
    test_suite.addTest(unittest.makeSuite(TestLSTMModel))
    test_suite.addTest(unittest.makeSuite(TestModelComparison))
    test_suite.addTest(unittest.makeSuite(TestEdgeCases))
    test_suite.addTest(unittest.makeSuite(TestPerformanceMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
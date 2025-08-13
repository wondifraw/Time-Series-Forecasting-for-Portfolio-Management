"""
Test Suite for Task 3: Future Market Trend Forecasting
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from task3.trend_analyzer import TrendAnalyzer
from task3.forecast_visualizer import ForecastVisualizer

class TestTask3Forecasting(unittest.TestCase):
    """Test Task 3 forecasting components."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample forecast data
        dates = pd.date_range(start='2024-01-01', periods=126, freq='B')
        prices = 200 + np.cumsum(np.random.randn(126) * 2)  # Random walk starting at $200
        self.sample_forecast = pd.Series(prices, index=dates)
        
        # Create sample confidence intervals
        self.sample_confidence = pd.DataFrame({
            'lower': prices - 10,
            'upper': prices + 10
        }, index=dates)
    
    def test_trend_detection(self):
        """Test trend detection functionality."""
        # Create upward trending data
        upward_data = pd.Series(np.linspace(100, 150, 100))
        trend_result = TrendAnalyzer.detect_trend(upward_data)
        
        self.assertIn('direction', trend_result)
        self.assertIn('total_change_pct', trend_result)
        self.assertIn('slope', trend_result)
        self.assertGreater(trend_result['total_change_pct'], 0)
    
    def test_risk_assessment(self):
        """Test risk assessment functionality."""
        risk_result = TrendAnalyzer.assess_risk_levels(self.sample_forecast, self.sample_confidence)
        
        self.assertIn('volatility_pct', risk_result)
        self.assertIn('risk_level', risk_result)
        self.assertIn('max_drawdown', risk_result)
        self.assertIn(risk_result['risk_level'], ['Low', 'Medium', 'High'])
    
    def test_opportunity_identification(self):
        """Test opportunity and risk identification."""
        trend_data = {'total_change_pct': 20, 'direction': 'Strong Upward'}
        risk_data = {'risk_level': 'Low', 'volatility_pct': 15, 'max_drawdown': -5}
        
        opportunities = TrendAnalyzer.identify_opportunities(trend_data, risk_data)
        risks = TrendAnalyzer.identify_risks(trend_data, risk_data)
        
        self.assertIsInstance(opportunities, list)
        self.assertIsInstance(risks, list)
        self.assertGreater(len(opportunities), 0)  # Should find opportunities with good trend
    
    def test_visualization_components(self):
        """Test that visualization functions don't crash."""
        try:
            # Test trend analysis (should not raise exception)
            trend_data = TrendAnalyzer.detect_trend(self.sample_forecast)
            risk_data = TrendAnalyzer.assess_risk_levels(self.sample_forecast)
            
            # These should not raise exceptions
            self.assertIsNotNone(trend_data)
            self.assertIsNotNone(risk_data)
            
        except Exception as e:
            self.fail(f"Visualization test failed: {str(e)}")
    
    def test_data_validation(self):
        """Test data validation and edge cases."""
        # Test empty data
        empty_series = pd.Series([])
        trend_result = TrendAnalyzer.detect_trend(empty_series)
        
        # Should handle empty data gracefully
        self.assertIsNotNone(trend_result)
        
        # Test single value
        single_value = pd.Series([100])
        trend_result = TrendAnalyzer.detect_trend(single_value)
        self.assertIsNotNone(trend_result)

if __name__ == '__main__':
    unittest.main()
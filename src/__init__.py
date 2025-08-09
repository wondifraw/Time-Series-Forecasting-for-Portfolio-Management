"""
Financial Time Series Analysis Package

A modular financial analysis system for preprocessing and exploring financial time series data.
"""

from .data_loader import FinancialDataLoader
from .data_preprocessor import FinancialDataPreprocessor
from .eda_analyzer import EDAAnalyzer
from .risk_calculator import RiskCalculator
from .report_generator import ReportGenerator


__version__ = "1.0.0"
__author__ = "Financial Analysis Team"

__all__ = [
    'FinancialDataLoader',
    'FinancialDataPreprocessor', 
    'EDAAnalyzer',
    'RiskCalculator',
    'ReportGenerator',

]
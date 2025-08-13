"""
Performance Analysis and Visualization
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from eda_analyzer import EDAAnalyzer
from report_generator import ReportGenerator

class PerformanceAnalyzer:
    """Analyze and visualize portfolio performance using Task 1 components."""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.eda_analyzer = EDAAnalyzer()
        self.report_generator = ReportGenerator()
    
    def plot_cumulative_returns(self, strategy_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> None:
        """Plot cumulative returns comparison."""
        plt.figure(figsize=(12, 6))
        
        plt.plot(strategy_returns.index, strategy_returns, 
                label='Strategy Portfolio', linewidth=2, color='blue')
        plt.plot(benchmark_returns.index, benchmark_returns, 
                label='Benchmark (60/40)', linewidth=2, color='red', linestyle='--')
        
        plt.title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_performance_summary(self, strategy_metrics: Dict, 
                                  benchmark_metrics: Dict) -> pd.DataFrame:
        """Create performance comparison table."""
        comparison = pd.DataFrame({
            'Strategy': strategy_metrics,
            'Benchmark': benchmark_metrics
        })
        
        # Format percentages
        for metric in ['total_return', 'annual_return', 'annual_volatility', 'max_drawdown']:
            comparison.loc[metric] = comparison.loc[metric].apply(lambda x: f"{x:.2%}")
        
        # Format Sharpe ratio
        comparison.loc['sharpe_ratio'] = comparison.loc['sharpe_ratio'].apply(lambda x: f"{x:.3f}")
        
        return comparison
    
    def print_backtest_summary(self, strategy_metrics: Dict, benchmark_metrics: Dict,
                              strategy_weights: Dict) -> None:
        """Print comprehensive backtest summary."""
        print("=" * 60)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        
        print("\nStrategy Portfolio Weights:")
        for asset, weight in strategy_weights.items():
            print(f"  {asset}: {weight:.1%}")
        
        print(f"\nBenchmark: 60% SPY / 40% BND")
        
        print("\nPerformance Comparison:")
        comparison = self.create_performance_summary(strategy_metrics, benchmark_metrics)
        print(comparison)
        
        # Determine outperformance
        strategy_return = strategy_metrics['total_return']
        benchmark_return = benchmark_metrics['total_return']
        outperformance = strategy_return - benchmark_return
        
        print(f"\nOutperformance: {outperformance:.2%}")
        
        if outperformance > 0:
            print("✓ Strategy OUTPERFORMED benchmark")
        else:
            print("✗ Strategy UNDERPERFORMED benchmark")
        
        # Risk-adjusted performance
        strategy_sharpe = strategy_metrics['sharpe_ratio']
        benchmark_sharpe = benchmark_metrics['sharpe_ratio']
        
        if strategy_sharpe > benchmark_sharpe:
            print("✓ Strategy has BETTER risk-adjusted returns")
        else:
            print("✗ Strategy has WORSE risk-adjusted returns")
        
        print("\n" + "=" * 60)
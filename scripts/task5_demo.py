"""
Task 5: Strategy Backtesting Demonstration
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from task5.backtest_engine import BacktestEngine
from task5.benchmark_portfolio import BenchmarkPortfolio
from task5.performance_analyzer import PerformanceAnalyzer

def main():
    """Run complete backtesting analysis."""
    print("Task 5: Strategy Backtesting")
    print("=" * 40)
    
    # Initialize components
    symbols = ['TSLA', 'BND', 'SPY']
    backtest_engine = BacktestEngine(symbols, start_date='2024-01-01', end_date='2024-12-31')
    benchmark = BenchmarkPortfolio()
    analyzer = PerformanceAnalyzer()
    
    print("Loading backtest data...")
    price_data = backtest_engine.load_backtest_data()
    print(f"Backtest period: {price_data.index.min().date()} to {price_data.index.max().date()}")
    print(f"Trading days: {len(price_data)}")
    
    # Get optimal strategy weights
    print("\nGetting optimal portfolio weights...")
    strategy_weights = backtest_engine.get_optimal_weights()
    
    # Simulate strategy performance
    print("Simulating strategy performance...")
    strategy_returns = backtest_engine.simulate_strategy(strategy_weights)
    strategy_metrics = backtest_engine.calculate_performance_metrics(strategy_returns)
    
    # Simulate benchmark performance
    print("Simulating benchmark performance...")
    benchmark_weights = benchmark.get_weights()
    benchmark_returns = benchmark.simulate_performance(backtest_engine.returns_data)
    benchmark_metrics = backtest_engine.calculate_performance_metrics(benchmark_returns)
    
    # Analyze and visualize results
    print("\nGenerating performance analysis...")
    analyzer.plot_cumulative_returns(strategy_returns, benchmark_returns)
    analyzer.print_backtest_summary(strategy_metrics, benchmark_metrics, strategy_weights)
    
    # Conclusion
    print("\nCONCLUSION:")
    outperformance = strategy_metrics['total_return'] - benchmark_metrics['total_return']
    
    if outperformance > 0:
        print(f"The model-driven strategy outperformed the benchmark by {outperformance:.2%}.")
        print("This suggests the forecasting approach adds value to portfolio construction.")
    else:
        print(f"The model-driven strategy underperformed the benchmark by {abs(outperformance):.2%}.")
        print("This suggests the simple benchmark may be more robust in this period.")
    
    print("\nKey insights:")
    print("- Backtesting validates strategy performance on historical data")
    print("- Risk-adjusted returns (Sharpe ratio) provide better comparison metric")
    print("- Model-driven approaches require ongoing validation and refinement")

if __name__ == "__main__":
    main()
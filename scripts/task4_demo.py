"""
Task 4 Demo: Portfolio Optimization Based on Forecasts
Complete implementation of Modern Portfolio Theory with forecast integration
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from task4.portfolio_optimizer import PortfolioOptimizer
from task4.efficient_frontier import EfficientFrontier
from task4.forecast_integrator import ForecastIntegrator

def main():
    """Run complete Task 4 portfolio optimization demo."""
    
    print("TASK 4: PORTFOLIO OPTIMIZATION BASED ON FORECASTS")
    print("=" * 60)
    
    # Step 1: Initialize components
    print("\nStep 1: Initializing Portfolio Optimization")
    print("-" * 40)
    
    # Initialize forecast integrator
    forecast_integrator = ForecastIntegrator()
    
    # Get market context
    forecast_integrator.print_market_context()
    
    # Get TSLA forecast
    print("\nGetting TSLA Forecast...")
    tsla_forecast = forecast_integrator.get_best_model_forecast()
    
    if not forecast_integrator.validate_forecast(tsla_forecast):
        print("Using conservative fallback forecast")
        tsla_forecast = 0.15  # 15% annual return
    
    # Step 2: Initialize portfolio optimizer
    print(f"\nStep 2: Setting up Portfolio Optimizer")
    print("-" * 40)
    
    optimizer = PortfolioOptimizer(symbols=['TSLA', 'BND', 'SPY'], risk_free_rate=0.02)
    
    # Load data and calculate inputs
    print("Loading returns data...")
    optimizer.load_returns_data()
    
    print("Calculating expected returns (with TSLA forecast)...")
    optimizer.calculate_expected_returns(forecast_return=tsla_forecast)
    
    print("Calculating covariance matrix...")
    optimizer.calculate_covariance_matrix()
    
    # Print expected returns
    print("\nExpected Annual Returns:")
    for i, symbol in enumerate(optimizer.symbols):
        print(f"  {symbol}: {optimizer.expected_returns[i]:.2%}")
    
    # Step 3: Generate Efficient Frontier
    print(f"\nStep 3: Generating Efficient Frontier")
    print("-" * 40)
    
    frontier = EfficientFrontier(optimizer)
    
    print("Generating efficient frontier portfolios...")
    frontier_data = frontier.generate_frontier(n_portfolios=50)
    
    print(f"Generated {len(frontier_data)} efficient portfolios")
    
    # Step 4: Find Key Portfolios
    print(f"\nStep 4: Identifying Key Portfolios")
    print("-" * 40)
    
    # Maximum Sharpe Ratio Portfolio
    max_sharpe = optimizer.optimize_max_sharpe()
    frontier.print_portfolio_summary(max_sharpe, "Maximum Sharpe Ratio")
    
    # Minimum Volatility Portfolio
    min_vol = optimizer.optimize_min_volatility()
    frontier.print_portfolio_summary(min_vol, "Minimum Volatility")
    
    # Step 5: Portfolio Recommendation
    print(f"\nStep 5: Portfolio Recommendation")
    print("-" * 40)
    
    # Analyze risk-return characteristics
    max_sharpe_return = max_sharpe['return']
    max_sharpe_vol = max_sharpe['volatility']
    max_sharpe_sharpe = max_sharpe['sharpe_ratio']
    
    min_vol_return = min_vol['return']
    min_vol_vol = min_vol['volatility']
    min_vol_sharpe = min_vol['sharpe_ratio']
    
    print("PORTFOLIO ANALYSIS:")
    print(f"Max Sharpe: {max_sharpe_return:.2%} return, {max_sharpe_vol:.2%} risk, {max_sharpe_sharpe:.3f} Sharpe")
    print(f"Min Vol:    {min_vol_return:.2%} return, {min_vol_vol:.2%} risk, {min_vol_sharpe:.3f} Sharpe")
    
    # Make recommendation based on analysis
    if max_sharpe_sharpe > 1.0 and max_sharpe_vol < 0.25:
        recommended = max_sharpe
        recommendation_reason = "Maximum risk-adjusted return with acceptable volatility"
        rec_name = "Maximum Sharpe Ratio"
    elif min_vol_vol < 0.15 and min_vol_sharpe > 0.5:
        recommended = min_vol
        recommendation_reason = "Low risk with reasonable returns"
        rec_name = "Minimum Volatility"
    else:
        # Moderate blend
        recommended = frontier.recommend_portfolio('moderate')
        recommendation_reason = "Balanced risk-return profile"
        rec_name = "Moderate Blend"
    
    print(f"\nRECOMMENDED PORTFOLIO: {rec_name}")
    print("=" * 50)
    print(f"Rationale: {recommendation_reason}")
    frontier.print_portfolio_summary(recommended, "RECOMMENDED")
    
    # Step 6: Sensitivity Analysis
    print(f"\nStep 6: Forecast Sensitivity Analysis")
    print("-" * 40)
    
    scenarios = forecast_integrator.create_forecast_scenarios(tsla_forecast)
    
    print("Testing different TSLA forecast scenarios:")
    sensitivity_results = {}
    
    for scenario_name, scenario_forecast in scenarios.items():
        print(f"\n{scenario_name.upper()} Scenario ({scenario_forecast:.2%}):")
        
        # Recalculate with new forecast
        temp_optimizer = PortfolioOptimizer(symbols=['TSLA', 'BND', 'SPY'], risk_free_rate=0.02)
        temp_optimizer.load_returns_data()
        temp_optimizer.calculate_expected_returns(forecast_return=scenario_forecast)
        temp_optimizer.calculate_covariance_matrix()
        
        scenario_max_sharpe = temp_optimizer.optimize_max_sharpe()
        
        print(f"  TSLA Weight: {scenario_max_sharpe['weights']['TSLA']:.1%}")
        print(f"  Portfolio Return: {scenario_max_sharpe['return']:.2%}")
        print(f"  Portfolio Risk: {scenario_max_sharpe['volatility']:.2%}")
        print(f"  Sharpe Ratio: {scenario_max_sharpe['sharpe_ratio']:.3f}")
        
        sensitivity_results[scenario_name] = scenario_max_sharpe
    
    # Step 7: Final Summary
    print(f"\nFINAL PORTFOLIO SUMMARY")
    print("=" * 50)
    
    print(f"Based on TSLA forecast of {tsla_forecast:.2%} annual return:")
    print(f"\nOPTIMAL PORTFOLIO WEIGHTS:")
    total_weight = 0
    for symbol, weight in recommended['weights'].items():
        print(f"  {symbol}: {weight:.1%}")
        total_weight += weight
    
    print(f"\nPORTFOLIO METRICS:")
    print(f"  Expected Annual Return: {recommended['return']:.2%}")
    print(f"  Annual Volatility: {recommended['volatility']:.2%}")
    print(f"  Sharpe Ratio: {recommended['sharpe_ratio']:.3f}")
    
    # Risk assessment
    if recommended['volatility'] < 0.15:
        risk_level = "Conservative"
    elif recommended['volatility'] < 0.25:
        risk_level = "Moderate"
    else:
        risk_level = "Aggressive"
    
    print(f"  Risk Level: {risk_level}")
    
    # Investment insights
    print(f"\nINVESTMENT INSIGHTS:")
    tsla_weight = recommended['weights']['TSLA']
    bnd_weight = recommended['weights']['BND']
    spy_weight = recommended['weights']['SPY']
    
    if tsla_weight > 0.4:
        print("  - High TSLA allocation suggests strong confidence in forecast")
    elif tsla_weight < 0.1:
        print("  - Low TSLA allocation suggests conservative approach")
    else:
        print("  - Moderate TSLA allocation balances growth and risk")
    
    if bnd_weight > 0.3:
        print("  - Significant bond allocation provides portfolio stability")
    
    if spy_weight > 0.4:
        print("  - High SPY allocation provides broad market exposure")
    
    # Plot efficient frontier
    print(f"\nGenerating Efficient Frontier Plot...")
    try:
        frontier.plot_frontier(show_key_portfolios=True)
        print("Efficient frontier plot displayed")
    except Exception as e:
        print(f"Could not display plot: {e}")
    
    print(f"\nTask 4 Portfolio Optimization Complete!")
    print("=" * 50)
    
    return {
        'recommended_portfolio': recommended,
        'max_sharpe_portfolio': max_sharpe,
        'min_volatility_portfolio': min_vol,
        'sensitivity_analysis': sensitivity_results,
        'tsla_forecast': tsla_forecast
    }

if __name__ == "__main__":
    results = main()
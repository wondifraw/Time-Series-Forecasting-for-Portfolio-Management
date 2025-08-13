"""
Efficient Frontier Generator and Visualizer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from .portfolio_optimizer import PortfolioOptimizer

class EfficientFrontier:
    """Generate and visualize the efficient frontier."""
    
    def __init__(self, optimizer: PortfolioOptimizer):
        self.optimizer = optimizer
        self.frontier_data = None
        
    def generate_frontier(self, n_portfolios: int = 100) -> pd.DataFrame:
        """Generate efficient frontier portfolios."""
        if self.optimizer.expected_returns is None or self.optimizer.cov_matrix is None:
            raise ValueError("Expected returns and covariance matrix must be calculated first")
        
        # Get min and max returns
        min_vol_portfolio = self.optimizer.optimize_min_volatility()
        max_sharpe_portfolio = self.optimizer.optimize_max_sharpe()
        
        min_return = min_vol_portfolio['return']
        max_return = max(self.optimizer.expected_returns) * 0.95  # Slightly below max individual return
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        frontier_portfolios = []
        
        for target_return in target_returns:
            portfolio = self.optimizer.optimize_for_return(target_return)
            if portfolio is not None:
                frontier_portfolios.append({
                    'return': portfolio['return'],
                    'volatility': portfolio['volatility'],
                    'sharpe_ratio': portfolio['sharpe_ratio'],
                    'weights': portfolio['weights']
                })
        
        self.frontier_data = pd.DataFrame(frontier_portfolios)
        return self.frontier_data
    
    def plot_frontier(self, show_key_portfolios: bool = True) -> None:
        """Plot the efficient frontier."""
        if self.frontier_data is None:
            self.generate_frontier()
        
        plt.figure(figsize=(12, 8))
        
        # Plot efficient frontier
        plt.plot(self.frontier_data['volatility'], self.frontier_data['return'], 
                'b-', linewidth=2, label='Efficient Frontier')
        
        if show_key_portfolios:
            # Plot key portfolios
            max_sharpe = self.optimizer.optimize_max_sharpe()
            min_vol = self.optimizer.optimize_min_volatility()
            
            plt.scatter(max_sharpe['volatility'], max_sharpe['return'], 
                       marker='*', s=300, c='red', label='Max Sharpe Ratio')
            plt.scatter(min_vol['volatility'], min_vol['return'], 
                       marker='o', s=200, c='green', label='Min Volatility')
            
            # Plot individual assets
            for i, symbol in enumerate(self.optimizer.symbols):
                asset_return = self.optimizer.expected_returns[i]
                asset_vol = np.sqrt(self.optimizer.cov_matrix[i, i])
                plt.scatter(asset_vol, asset_return, marker='s', s=100, 
                           label=f'{symbol}', alpha=0.7)
        
        plt.xlabel('Volatility (Risk)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.title('Efficient Frontier', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_key_portfolios(self) -> Dict:
        """Get key portfolio recommendations."""
        max_sharpe = self.optimizer.optimize_max_sharpe()
        min_vol = self.optimizer.optimize_min_volatility()
        
        return {
            'max_sharpe': max_sharpe,
            'min_volatility': min_vol
        }
    
    def print_portfolio_summary(self, portfolio: Dict, name: str) -> None:
        """Print portfolio summary."""
        print(f"\n{name.upper()} PORTFOLIO:")
        print("=" * 40)
        print(f"Expected Annual Return: {portfolio['return']:.2%}")
        print(f"Annual Volatility: {portfolio['volatility']:.2%}")
        print(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        print("\nAsset Allocation:")
        for symbol, weight in portfolio['weights'].items():
            print(f"  {symbol}: {weight:.1%}")
    
    def recommend_portfolio(self, risk_tolerance: str = 'moderate') -> Dict:
        """Recommend portfolio based on risk tolerance."""
        key_portfolios = self.get_key_portfolios()
        
        if risk_tolerance.lower() == 'conservative':
            return key_portfolios['min_volatility']
        elif risk_tolerance.lower() == 'aggressive':
            return key_portfolios['max_sharpe']
        else:  # moderate
            # Blend of max sharpe and min volatility
            max_sharpe = key_portfolios['max_sharpe']
            min_vol = key_portfolios['min_volatility']
            
            # 60% max sharpe, 40% min vol
            blended_weights = {}
            for symbol in self.optimizer.symbols:
                blended_weights[symbol] = (0.6 * max_sharpe['weights'][symbol] + 
                                         0.4 * min_vol['weights'][symbol])
            
            # Calculate performance
            weights_array = np.array([blended_weights[symbol] for symbol in self.optimizer.symbols])
            ret, vol = self.optimizer.portfolio_performance(weights_array)
            sharpe = self.optimizer.sharpe_ratio(weights_array)
            
            return {
                'weights': blended_weights,
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            }
    
    def analyze_forecast_impact(self, base_forecast: float, forecast_scenarios: List[float]) -> Dict:
        """Analyze impact of different forecast scenarios."""
        results = {}
        
        for scenario_return in forecast_scenarios:
            # Recalculate expected returns with new forecast
            self.optimizer.calculate_expected_returns(forecast_return=scenario_return)
            self.optimizer.calculate_covariance_matrix()
            
            # Get optimal portfolios
            max_sharpe = self.optimizer.optimize_max_sharpe()
            
            results[f"forecast_{scenario_return:.1%}"] = {
                'max_sharpe': max_sharpe,
                'tsla_weight': max_sharpe['weights']['TSLA']
            }
        
        return results
"""
Forecast Visualization Module
Minimal implementation for plotting forecast results
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional

class ForecastVisualizer:
    """Create visualizations for forecast analysis."""
    
    @staticmethod
    def plot_forecast_with_confidence(historical: pd.Series, forecast: pd.Series, 
                                    confidence_intervals: Optional[pd.DataFrame] = None,
                                    title: str = "Market Forecast"):
        """Plot forecast with historical data and confidence intervals."""
        plt.figure(figsize=(14, 8))
        
        # Plot recent historical data (last 6 months)
        recent_hist = historical.iloc[-126:] if len(historical) > 126 else historical
        plt.plot(recent_hist.index, recent_hist.values, 
                label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        plt.plot(forecast.index, forecast.values, 
                label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals
        if confidence_intervals is not None:
            plt.fill_between(confidence_intervals.index, 
                           confidence_intervals['lower'], 
                           confidence_intervals['upper'],
                           alpha=0.3, color='red', label='95% Confidence Interval')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_trend_analysis(forecast: pd.Series, trend_data: Dict):
        """Plot trend analysis with key metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Forecast with trend line
        x = np.arange(len(forecast))
        trend_line = np.polyval([trend_data['slope'], forecast.iloc[0]], x)
        
        ax1.plot(forecast.index, forecast.values, label='Forecast', color='blue')
        ax1.plot(forecast.index, trend_line, label='Trend Line', color='red', linestyle='--')
        ax1.set_title(f"Trend: {trend_data['direction']}")
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Monthly returns
        monthly_returns = forecast.resample('M').last().pct_change().dropna() * 100
        colors = ['green' if x > 0 else 'red' for x in monthly_returns]
        ax2.bar(range(len(monthly_returns)), monthly_returns, color=colors)
        ax2.set_title('Monthly Returns (%)')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_risk_analysis(forecast: pd.Series, risk_data: Dict):
        """Plot risk analysis visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Price distribution
        ax1.hist(forecast.values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(forecast.mean(), color='red', linestyle='--', label=f'Mean: ${forecast.mean():.2f}')
        ax1.set_title('Forecast Price Distribution')
        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volatility over time
        rolling_vol = forecast.pct_change().rolling(21).std() * np.sqrt(252) * 100
        ax2.plot(forecast.index[21:], rolling_vol.dropna(), color='orange')
        ax2.set_title(f'Rolling Volatility - Risk Level: {risk_data["risk_level"]}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_summary_dashboard(historical: pd.Series, forecast: pd.Series, 
                               trend_data: Dict, risk_data: Dict, 
                               opportunities: list, risks: list):
        """Create comprehensive summary dashboard."""
        fig = plt.figure(figsize=(16, 10))
        
        # Main forecast plot
        ax1 = plt.subplot(2, 3, (1, 2))
        recent_hist = historical.iloc[-126:]
        ax1.plot(recent_hist.index, recent_hist.values, 'b-', label='Historical', linewidth=2)
        ax1.plot(forecast.index, forecast.values, 'r--', label='Forecast', linewidth=2)
        ax1.set_title('Price Forecast')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Trend metrics
        ax2 = plt.subplot(2, 3, 3)
        metrics = [
            f"Direction: {trend_data['direction']}",
            f"Total Change: {trend_data['total_change_pct']:.1f}%",
            f"Volatility: {trend_data['volatility']:.1f}%",
            f"Risk Level: {risk_data['risk_level']}"
        ]
        ax2.text(0.1, 0.9, '\n'.join(metrics), transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax2.set_title('Key Metrics')
        ax2.axis('off')
        
        # Opportunities
        ax3 = plt.subplot(2, 3, 4)
        opp_text = '\n'.join([f"• {opp}" for opp in opportunities[:4]])
        ax3.text(0.05, 0.95, opp_text, transform=ax3.transAxes, 
                fontsize=9, verticalalignment='top', color='green')
        ax3.set_title('Opportunities')
        ax3.axis('off')
        
        # Risks
        ax4 = plt.subplot(2, 3, 5)
        risk_text = '\n'.join([f"• {risk}" for risk in risks[:4]])
        ax4.text(0.05, 0.95, risk_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top', color='red')
        ax4.set_title('Risks')
        ax4.axis('off')
        
        # Price range
        ax5 = plt.subplot(2, 3, 6)
        current_price = historical.iloc[-1]
        forecast_end = forecast.iloc[-1]
        price_range = [current_price, forecast.min(), forecast.max(), forecast_end]
        labels = ['Current', 'Min', 'Max', 'End']
        colors = ['blue', 'red', 'green', 'orange']
        ax5.bar(labels, price_range, color=colors, alpha=0.7)
        ax5.set_title('Price Levels')
        ax5.set_ylabel('Price ($)')
        
        plt.tight_layout()
        plt.show()
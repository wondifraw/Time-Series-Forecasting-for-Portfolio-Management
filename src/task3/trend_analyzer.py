"""
Trend Analysis Module for Future Market Forecasts
Minimal implementation for trend detection and risk assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List

class TrendAnalyzer:
    """Analyze trends and patterns in forecast data."""
    
    @staticmethod
    def detect_trend(forecast: pd.Series) -> Dict:
        """Detect overall trend direction and strength."""
        if len(forecast) == 0:
            return {
                'direction': 'No Data',
                'total_change_pct': 0.0,
                'slope': 0.0,
                'volatility': 0.0
            }
        
        if len(forecast) == 1:
            return {
                'direction': 'Insufficient Data',
                'total_change_pct': 0.0,
                'slope': 0.0,
                'volatility': 0.0
            }
        
        start_price = forecast.iloc[0]
        end_price = forecast.iloc[-1]
        total_change = (end_price - start_price) / start_price * 100
        
        # Calculate trend strength using linear regression slope
        x = np.arange(len(forecast))
        slope = np.polyfit(x, forecast.values, 1)[0]
        
        if total_change > 10:
            direction = "Strong Upward"
        elif total_change > 2:
            direction = "Moderate Upward"
        elif total_change < -10:
            direction = "Strong Downward"
        elif total_change < -2:
            direction = "Moderate Downward"
        else:
            direction = "Sideways"
        
        volatility = forecast.std() / forecast.mean() * 100 if forecast.mean() != 0 else 0.0
        
        return {
            'direction': direction,
            'total_change_pct': total_change,
            'slope': slope,
            'volatility': volatility
        }
    
    @staticmethod
    def assess_risk_levels(forecast: pd.Series, confidence_intervals: pd.DataFrame = None) -> Dict:
        """Assess risk levels from forecast data."""
        volatility = forecast.pct_change().std() * np.sqrt(252) * 100
        
        risk_level = "Low" if volatility < 20 else "Medium" if volatility < 40 else "High"
        
        risks = {
            'volatility_pct': volatility,
            'risk_level': risk_level,
            'max_drawdown': TrendAnalyzer._calculate_max_drawdown(forecast)
        }
        
        if confidence_intervals is not None:
            width = (confidence_intervals['upper'] - confidence_intervals['lower']) / forecast * 100
            risks['uncertainty_pct'] = width.mean()
            risks['uncertainty_trend'] = "Increasing" if width.iloc[-1] > width.iloc[0] else "Stable"
        
        return risks
    
    @staticmethod
    def _calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak * 100
        return drawdown.min()
    
    @staticmethod
    def identify_opportunities(trend_data: Dict, risk_data: Dict) -> List[str]:
        """Identify investment opportunities."""
        opportunities = []
        
        if trend_data['total_change_pct'] > 15:
            opportunities.append(f"Strong growth potential: {trend_data['total_change_pct']:.1f}% expected return")
        
        if risk_data['risk_level'] == "Low" and trend_data['total_change_pct'] > 5:
            opportunities.append("Low-risk growth opportunity")
        
        if trend_data['direction'] in ["Strong Upward", "Moderate Upward"]:
            opportunities.append("Positive trend momentum")
        
        return opportunities
    
    @staticmethod
    def identify_risks(trend_data: Dict, risk_data: Dict) -> List[str]:
        """Identify investment risks."""
        risks = []
        
        if trend_data['total_change_pct'] < -10:
            risks.append(f"Significant decline risk: {trend_data['total_change_pct']:.1f}%")
        
        if risk_data['volatility_pct'] > 50:
            risks.append(f"High volatility: {risk_data['volatility_pct']:.1f}%")
        
        if risk_data['max_drawdown'] < -20:
            risks.append(f"Large potential drawdown: {risk_data['max_drawdown']:.1f}%")
        
        if 'uncertainty_trend' in risk_data and risk_data['uncertainty_trend'] == "Increasing":
            risks.append("Forecast uncertainty increases over time")
        
        return risks
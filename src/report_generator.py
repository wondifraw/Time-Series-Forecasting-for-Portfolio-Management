"""
Report Generation Module for Financial Analysis

This module handles generating comprehensive analysis reports and summaries.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime


class ReportGenerator:
    """
    Generates comprehensive reports and summaries for financial analysis.
    """
    
    def __init__(self):
        """Initialize the report generator."""
        self.report_data = {}
    
    def generate_summary_report(self, symbols: list, statistics: Dict[str, Dict], 
                              var_results: Dict[str, Dict], sharpe_results: Dict[str, Dict],
                              adf_results: Dict[str, Dict] = None) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            symbols (list): List of analyzed symbols
            statistics (Dict[str, Dict]): Statistical measures
            var_results (Dict[str, Dict]): VaR calculation results
            sharpe_results (Dict[str, Dict]): Sharpe ratio results
            adf_results (Dict[str, Dict]): ADF test results (optional)
            
        Returns:
            str: Formatted summary report
        """
        print("Generating comprehensive summary report...")
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("FINANCIAL DATA ANALYSIS SUMMARY REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Symbols Analyzed: {', '.join(symbols)}")
        report_lines.append("")
        
        # Performance Summary
        report_lines.append("PERFORMANCE SUMMARY")
        report_lines.append("-" * 30)
        
        performance_data = []
        for symbol in symbols:
            if symbol in sharpe_results:
                perf_data = {
                    'Symbol': symbol,
                    'Annual Return': f"{sharpe_results[symbol]['annual_return']*100:.2f}%",
                    'Volatility': f"{sharpe_results[symbol]['annual_volatility']*100:.2f}%",
                    'Sharpe Ratio': f"{sharpe_results[symbol]['sharpe_ratio']:.3f}",
                    'Max Drawdown': f"{sharpe_results[symbol]['max_drawdown']*100:.2f}%"
                }
                performance_data.append(perf_data)
        
        if performance_data:
            df_performance = pd.DataFrame(performance_data)
            report_lines.append(df_performance.to_string(index=False))
        
        report_lines.append("")
        
        # Risk Summary
        report_lines.append("RISK SUMMARY (95% VaR)")
        report_lines.append("-" * 25)
        
        risk_data = []
        for symbol in symbols:
            if symbol in var_results:
                risk_data.append({
                    'Symbol': symbol,
                    'Historical VaR': f"{var_results[symbol]['historical_var']*100:.2f}%",
                    'Parametric VaR': f"{var_results[symbol]['parametric_var']*100:.2f}%"
                })
        
        if risk_data:
            df_risk = pd.DataFrame(risk_data)
            report_lines.append(df_risk.to_string(index=False))
        
        report_lines.append("")
        
        # Key Insights
        report_lines.append("KEY INSIGHTS")
        report_lines.append("-" * 15)
        
        insights = self._generate_insights(symbols, statistics, var_results, sharpe_results)
        for insight in insights:
            report_lines.append(f"• {insight}")
        
        report_lines.append("")
        
        # Investment Recommendations
        report_lines.append("INVESTMENT RECOMMENDATIONS")
        report_lines.append("-" * 30)
        
        recommendations = self._generate_recommendations(symbols, sharpe_results, var_results)
        for rec in recommendations:
            report_lines.append(f"• {rec}")
        
        # Stationarity Results (if available)
        if adf_results:
            report_lines.append("")
            report_lines.append("STATIONARITY TEST RESULTS")
            report_lines.append("-" * 30)
            
            for symbol, results in adf_results.items():
                price_status = "Stationary" if results['prices']['is_stationary'] else "Non-stationary"
                return_status = "Stationary" if results['returns']['is_stationary'] else "Non-stationary"
                
                report_lines.append(f"{symbol}:")
                report_lines.append(f"  Prices: {price_status} (p-value: {results['prices']['p_value']:.4f})")
                report_lines.append(f"  Returns: {return_status} (p-value: {results['returns']['p_value']:.4f})")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        full_report = "\n".join(report_lines)
        print("✓ Summary report generated")
        
        return full_report
    
    def _generate_insights(self, symbols: list, statistics: Dict[str, Dict], 
                          var_results: Dict[str, Dict], sharpe_results: Dict[str, Dict]) -> list:
        """
        Generate key insights from the analysis.
        
        Args:
            symbols (list): Analyzed symbols
            statistics (Dict[str, Dict]): Statistical measures
            var_results (Dict[str, Dict]): VaR results
            sharpe_results (Dict[str, Dict]): Sharpe ratio results
            
        Returns:
            list: List of insight strings
        """
        insights = []
        
        # Find highest and lowest volatility
        if sharpe_results:
            volatilities = {symbol: data['annual_volatility'] for symbol, data in sharpe_results.items()}
            highest_vol = max(volatilities, key=volatilities.get)
            lowest_vol = min(volatilities, key=volatilities.get)
            
            insights.append(f"Highest Volatility: {highest_vol} ({volatilities[highest_vol]*100:.1f}%)")
            insights.append(f"Lowest Volatility: {lowest_vol} ({volatilities[lowest_vol]*100:.1f}%)")
        
        # Find best and worst performers
        if sharpe_results:
            returns = {symbol: data['annual_return'] for symbol, data in sharpe_results.items()}
            best_performer = max(returns, key=returns.get)
            worst_performer = min(returns, key=returns.get)
            
            insights.append(f"Best Performer: {best_performer} ({returns[best_performer]*100:.1f}% annual return)")
            insights.append(f"Worst Performer: {worst_performer} ({returns[worst_performer]*100:.1f}% annual return)")
        
        # Sharpe ratio insights
        if sharpe_results:
            sharpe_ratios = {symbol: data['sharpe_ratio'] for symbol, data in sharpe_results.items()}
            best_sharpe = max(sharpe_ratios, key=sharpe_ratios.get)
            
            insights.append(f"Best Risk-Adjusted Return: {best_sharpe} (Sharpe: {sharpe_ratios[best_sharpe]:.3f})")
        
        return insights
    
    def _generate_recommendations(self, symbols: list, sharpe_results: Dict[str, Dict], 
                                var_results: Dict[str, Dict]) -> list:
        """
        Generate investment recommendations based on analysis.
        
        Args:
            symbols (list): Analyzed symbols
            sharpe_results (Dict[str, Dict]): Sharpe ratio results
            var_results (Dict[str, Dict]): VaR results
            
        Returns:
            list: List of recommendation strings
        """
        recommendations = []
        
        # Symbol-specific recommendations
        symbol_profiles = {
            'TSLA': 'High-risk, high-reward growth investment with significant volatility',
            'BND': 'Low-risk, stable income investment suitable for portfolio diversification',
            'SPY': 'Moderate-risk market exposure providing diversified equity returns'
        }
        
        for symbol in symbols:
            if symbol in symbol_profiles:
                recommendations.append(f"{symbol}: {symbol_profiles[symbol]}")
        
        # Risk-based recommendations
        if sharpe_results and var_results:
            # Find the symbol with best risk-adjusted returns
            sharpe_ratios = {symbol: data['sharpe_ratio'] for symbol, data in sharpe_results.items()}
            best_sharpe_symbol = max(sharpe_ratios, key=sharpe_ratios.get)
            
            recommendations.append(f"For risk-adjusted returns, consider higher allocation to {best_sharpe_symbol}")
            
            # Portfolio diversification recommendation
            if len(symbols) > 1:
                recommendations.append("Consider portfolio diversification across all assets to reduce overall risk")
        
        return recommendations
    
    def export_detailed_data(self, filename: str, **data_dict) -> None:
        """
        Export detailed analysis data to CSV files.
        
        Args:
            filename (str): Base filename for exports
            **data_dict: Dictionary of data to export
        """
        print(f"Exporting detailed data to {filename}_*.csv files...")
        
        for data_name, data in data_dict.items():
            if isinstance(data, dict):
                # Convert dict to DataFrame if possible
                try:
                    if all(isinstance(v, (dict, pd.Series)) for v in data.values()):
                        df = pd.DataFrame(data).T
                        df.to_csv(f"{filename}_{data_name}.csv")
                        print(f"✓ Exported {data_name} data")
                except Exception as e:
                    print(f"Warning: Could not export {data_name}: {str(e)}")
    
    def create_executive_summary(self, symbols: list, sharpe_results: Dict[str, Dict]) -> str:
        """
        Create a brief executive summary for stakeholders.
        
        Args:
            symbols (list): Analyzed symbols
            sharpe_results (Dict[str, Dict]): Sharpe ratio results
            
        Returns:
            str: Executive summary
        """
        summary_lines = []
        summary_lines.append("EXECUTIVE SUMMARY")
        summary_lines.append("=" * 20)
        summary_lines.append("")
        
        if sharpe_results:
            # Portfolio overview
            avg_return = np.mean([data['annual_return'] for data in sharpe_results.values()])
            avg_volatility = np.mean([data['annual_volatility'] for data in sharpe_results.values()])
            
            summary_lines.append(f"Portfolio Analysis of {len(symbols)} assets:")
            summary_lines.append(f"• Average Annual Return: {avg_return*100:.1f}%")
            summary_lines.append(f"• Average Volatility: {avg_volatility*100:.1f}%")
            summary_lines.append("")
            
            # Top performer
            returns = {symbol: data['annual_return'] for symbol, data in sharpe_results.items()}
            top_performer = max(returns, key=returns.get)
            summary_lines.append(f"Top Performer: {top_performer} ({returns[top_performer]*100:.1f}% annual return)")
            
            # Risk assessment
            volatilities = {symbol: data['annual_volatility'] for symbol, data in sharpe_results.items()}
            riskiest = max(volatilities, key=volatilities.get)
            summary_lines.append(f"Highest Risk: {riskiest} ({volatilities[riskiest]*100:.1f}% volatility)")
        
        return "\n".join(summary_lines)
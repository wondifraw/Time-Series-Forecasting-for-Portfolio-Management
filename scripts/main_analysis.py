"""
Main Analysis Pipeline for Financial Time Series Forecasting

This module orchestrates the complete financial data analysis pipeline using
modular components for data loading, preprocessing, EDA, risk calculation, and reporting.
"""

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import FinancialDataLoader
from data_preprocessor import FinancialDataPreprocessor
from eda_analyzer import EDAAnalyzer
from risk_calculator import RiskCalculator
from report_generator import ReportGenerator
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class FinancialAnalysisPipeline:
    """
    Main pipeline class that orchestrates the complete financial analysis workflow.
    """
    
    def __init__(self, symbols: List[str] = ['TSLA', 'BND', 'SPY'], 
                 period: str = '5y', risk_free_rate: float = 0.02):
        """
        Initialize the analysis pipeline.
        
        Args:
            symbols (List[str]): List of stock symbols to analyze
            period (str): Time period for data collection
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
        """
        self.symbols = symbols
        self.period = period
        self.risk_free_rate = risk_free_rate
        
        # Initialize modular components
        self.data_loader = FinancialDataLoader(symbols, period)
        self.preprocessor = FinancialDataPreprocessor()
        self.eda_analyzer = EDAAnalyzer()
        self.risk_calculator = RiskCalculator(risk_free_rate)
        self.report_generator = ReportGenerator()
        
        # Storage for analysis results
        self.results = {}
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Execute the complete financial analysis pipeline.
        
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        print("Starting Complete Financial Analysis Pipeline")
        print("=" * 50)
        
        try:
            # Step 1: Load Data
            print("\n[STEP 1] Loading Financial Data")
            raw_data = self.data_loader.load_data()
            data_info = self.data_loader.get_data_info()
            self.results['data_info'] = data_info
            
            # Step 2: Preprocess Data
            print("\n[STEP 2] Preprocessing Data")
            cleaned_data = self.preprocessor.clean_data(raw_data)
            returns_data = self.preprocessor.calculate_returns(cleaned_data)
            statistics = self.preprocessor.calculate_statistics()
            outliers = self.preprocessor.detect_outliers()
            
            self.results.update({
                'cleaned_data': cleaned_data,
                'returns_data': returns_data,
                'statistics': statistics,
                'outliers': outliers
            })
            
            # Step 3: Exploratory Data Analysis
            print("\n[STEP 3] Performing Exploratory Data Analysis")
            self.eda_analyzer.create_comprehensive_plots(cleaned_data, returns_data)
            adf_results = self.eda_analyzer.perform_stationarity_tests(cleaned_data, returns_data)
            volatility_analysis = self.eda_analyzer.analyze_volatility_patterns(returns_data)
            
            self.results.update({
                'adf_results': adf_results,
                'volatility_analysis': volatility_analysis
            })
            
            # Step 4: Risk Calculations
            print("\n[STEP 4] Calculating Risk Metrics")
            var_results = self.risk_calculator.calculate_var(returns_data)
            sharpe_results = self.risk_calculator.calculate_sharpe_ratio(returns_data)
            portfolio_metrics = self.risk_calculator.calculate_portfolio_metrics(returns_data)
            beta_results = self.risk_calculator.calculate_beta(returns_data)
            
            self.results.update({
                'var_results': var_results,
                'sharpe_results': sharpe_results,
                'portfolio_metrics': portfolio_metrics,
                'beta_results': beta_results
            })
            
            # Step 5: Generate Reports
            print("\n[STEP 5] Generating Analysis Reports")
            summary_report = self.report_generator.generate_summary_report(
                self.symbols, statistics, var_results, sharpe_results, adf_results
            )
            executive_summary = self.report_generator.create_executive_summary(
                self.symbols, sharpe_results
            )
            
            self.results.update({
                'summary_report': summary_report,
                'executive_summary': executive_summary
            })
            
            # Display final report
            print("\n" + summary_report)
            
            print("\n✅ Complete Analysis Pipeline Finished Successfully!")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Analysis Pipeline Failed: {str(e)}")
            raise
    
    def run_quick_analysis(self) -> Dict[str, Any]:
        """
        Execute a quick analysis with essential metrics only.
        
        Returns:
            Dict[str, Any]: Essential analysis results
        """
        print("Starting Quick Financial Analysis")
        print("=" * 35)
        
        # Load and preprocess data
        raw_data = self.data_loader.load_data()
        cleaned_data = self.preprocessor.clean_data(raw_data)
        returns_data = self.preprocessor.calculate_returns(cleaned_data)
        statistics = self.preprocessor.calculate_statistics()
        
        # Calculate essential risk metrics
        var_results = self.risk_calculator.calculate_var(returns_data)
        sharpe_results = self.risk_calculator.calculate_sharpe_ratio(returns_data)
        
        # Generate quick summary
        executive_summary = self.report_generator.create_executive_summary(
            self.symbols, sharpe_results
        )
        
        print("\n" + executive_summary)
        
        return {
            'returns_data': returns_data,
            'statistics': statistics,
            'var_results': var_results,
            'sharpe_results': sharpe_results,
            'executive_summary': executive_summary
        }
    
    def export_results(self, filename_base: str = 'financial_analysis') -> None:
        """
        Export analysis results to files.
        
        Args:
            filename_base (str): Base filename for exports
        """
        if not self.results:
            print("No results to export. Run analysis first.")
            return
        
        # Export detailed data
        exportable_data = {
            'statistics': self.results.get('statistics'),
            'var_results': self.results.get('var_results'),
            'sharpe_results': self.results.get('sharpe_results'),
            'adf_results': self.results.get('adf_results')
        }
        
        self.report_generator.export_detailed_data(filename_base, **exportable_data)
        
        # Export text reports
        if 'summary_report' in self.results:
            with open(f"{filename_base}_summary.txt", 'w') as f:
                f.write(self.results['summary_report'])
            print(f"✓ Summary report exported to {filename_base}_summary.txt")


def main():
    """
    Main execution function demonstrating the analysis pipeline.
    """
    # Initialize pipeline with default parameters
    pipeline = FinancialAnalysisPipeline(
        symbols=['TSLA', 'BND', 'SPY'],
        period='5y',
        risk_free_rate=0.02
    )
    
    # Run complete analysis
    results = pipeline.run_complete_analysis()
    
    # Optional: Export results
    # pipeline.export_results('financial_analysis_results')
    
    return pipeline, results


def quick_demo():
    """
    Quick demonstration of essential functionality.
    """
    pipeline = FinancialAnalysisPipeline(['TSLA', 'SPY'], period='2y')
    results = pipeline.run_quick_analysis()
    return pipeline, results


if __name__ == "__main__":
    # Run the complete analysis
    pipeline, results = main()
    
    print("\nAnalysis completed! Use 'pipeline' and 'results' objects for further exploration.")
    print("For a quick demo with fewer symbols, run: quick_demo()")
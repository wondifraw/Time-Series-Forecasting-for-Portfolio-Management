#!/usr/bin/env python3
"""
Task 2 Demonstration Script
Showcases ARIMA and LSTM forecasting models with Task 1 integration
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from task2.forecasting_pipeline import ForecastingPipeline
from task2.arima_model import ARIMAForecaster
from task2.lstm_model import LSTMForecaster

def run_individual_models():
    """Run individual model demonstrations."""
    print("🔧 INDIVIDUAL MODEL DEMONSTRATIONS")
    print("=" * 45)
    
    # ARIMA Model Demo
    print("\n1. ARIMA Model Demonstration:")
    arima_model = ARIMAForecaster(['TSLA'])
    arima_results = arima_model.run_complete_analysis()
    
    if arima_results:
        print("✅ ARIMA analysis completed successfully")
    else:
        print("❌ ARIMA analysis failed")
    
    # LSTM Model Demo
    print("\n2. LSTM Model Demonstration:")
    lstm_model = LSTMForecaster(['TSLA'])
    lstm_results = lstm_model.run_complete_analysis()
    
    if lstm_results:
        print("✅ LSTM analysis completed successfully")
    else:
        print("❌ LSTM analysis failed")
    
    return arima_results, lstm_results

def run_complete_pipeline():
    """Run complete forecasting pipeline."""
    print("\n🚀 COMPLETE FORECASTING PIPELINE")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = ForecastingPipeline(['TSLA'], test_size=0.2)
    
    # Run complete analysis
    results = pipeline.run_complete_pipeline()
    
    return results

def main():
    """Main demonstration function."""
    print("📊 TASK 2: TIME SERIES FORECASTING DEMONSTRATION")
    print("=" * 55)
    print("This demo showcases ARIMA and LSTM models for Tesla stock forecasting")
    print("Models inherit data loading capabilities from Task 1")
    print()
    
    try:
        # Option 1: Run individual models
        print("Choose demonstration mode:")
        print("1. Individual Models (ARIMA then LSTM)")
        print("2. Complete Pipeline (Both models + comparison)")
        print("3. Both")
        
        choice = input("\nEnter choice (1/2/3) or press Enter for complete pipeline: ").strip()
        
        if choice == '1':
            run_individual_models()
        elif choice == '3':
            run_individual_models()
            run_complete_pipeline()
        else:
            # Default: Complete pipeline
            run_complete_pipeline()
        
        print("\n🎉 Task 2 demonstration completed!")
        print("\nKey Features Demonstrated:")
        print("✅ Modular architecture inheriting from Task 1")
        print("✅ ARIMA model with auto-parameter optimization")
        print("✅ LSTM model with proper time series validation")
        print("✅ Comprehensive model comparison")
        print("✅ Professional visualizations")
        print("✅ Performance metrics (MAE, RMSE, MAPE)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Please ensure:")
        print("1. All required packages are installed")
        print("2. Task 1 data is available (run main_analysis.py first)")
        print("3. Python environment is properly configured")

if __name__ == "__main__":
    main()
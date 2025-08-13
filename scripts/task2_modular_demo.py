"""
Task 2 Modular Demo Script
Demonstrates the enhanced modular forecasting system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from task2 import (
    ForecastingPipeline, 
    ModelFactory, 
    ConfigManager,
    ModelConfig
)

def main():
    """Run modular forecasting demo."""
    print("🚀 MODULAR TIME SERIES FORECASTING DEMO")
    print("=" * 50)
    
    # 1. Show available models
    print("\n📋 Available Models:")
    for model_type in ModelFactory.list_available_models():
        info = ModelFactory.get_model_info(model_type)
        print(f"  - {model_type}: {info['class']}")
    
    # 2. Create custom configuration
    config = ConfigManager.get_default_config()
    config.data.test_size = 0.25
    config.arima.max_p = 3
    config.lstm.epochs = 30
    
    print(f"\n⚙️ Configuration:")
    print(f"  Test size: {config.data.test_size}")
    print(f"  ARIMA max_p: {config.arima.max_p}")
    print(f"  LSTM epochs: {config.lstm.epochs}")
    
    # 3. Run pipeline with custom config
    pipeline = ForecastingPipeline(config=config)
    results = pipeline.run()
    
    if results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Models analyzed: {list(results['results'].keys())}")
        
        if results['comparison']:
            ranking = results['comparison']['overall_ranking']
            print(f"🏆 Best model: {ranking.index[0]} (rank: {ranking.iloc[0]:.2f})")
    
    # 4. Run specific models only
    print(f"\n🎯 Running ARIMA only:")
    arima_results = pipeline.run(models=['ARIMA'], plot=False)
    if arima_results:
        arima_metrics = arima_results['results']['ARIMA']['metrics']
        print(f"   MAE: {arima_metrics['MAE']:.4f}")
    
    # 5. Demonstrate individual model creation
    print(f"\n🔧 Individual Model Creation:")
    arima_model = ModelFactory.create_model('ARIMA', config)
    lstm_model = ModelFactory.create_model('LSTM', config)
    
    print(f"  ARIMA model: {type(arima_model).__name__}")
    print(f"  LSTM model: {type(lstm_model).__name__}")
    
    # 6. Quick pipeline usage
    print(f"\n⚡ Quick Usage:")
    quick_pipeline = ForecastingPipeline()
    quick_results = quick_pipeline.run(plot=False)
    if quick_results:
        print(f"   Quick analysis completed with {len(quick_results['results'])} models")

if __name__ == "__main__":
    main()
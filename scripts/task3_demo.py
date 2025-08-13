"""
Task 3 Demo: Future Market Trend Forecasting
Demonstrates 6-12 month forecasting using trained models from Task 2
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from task3.future_forecaster import FutureMarketForecaster
from task3.trend_analyzer import TrendAnalyzer
from task3.forecast_visualizer import ForecastVisualizer

def run_task3_demo():
    """Run complete Task 3 demonstration."""
    print("🔮 TASK 3: FUTURE MARKET TREND FORECASTING")
    print("=" * 60)
    
    # Initialize forecaster
    forecaster = FutureMarketForecaster('TSLA')
    
    # Load trained models
    print("\n📂 Loading trained models from Task 2...")
    if not forecaster.load_trained_models():
        print("❌ Please run Task 2 first to train models")
        return
    
    # Generate forecasts for both models
    models_to_test = ['arima', 'lstm']
    results = {}
    
    for model_type in models_to_test:
        print(f"\n🔍 Testing {model_type.upper()} model...")
        
        # Generate 6-month forecast
        forecast_result = forecaster.generate_future_forecast(months=6, model_type=model_type)
        
        if forecast_result:
            print(f"✅ {model_type.upper()} forecast generated successfully")
            
            # Analyze trends
            trend_data = TrendAnalyzer.detect_trend(forecast_result['forecast'])
            risk_data = TrendAnalyzer.assess_risk_levels(
                forecast_result['forecast'], 
                forecast_result.get('confidence_intervals')
            )
            
            # Identify opportunities and risks
            opportunities = TrendAnalyzer.identify_opportunities(trend_data, risk_data)
            risks = TrendAnalyzer.identify_risks(trend_data, risk_data)
            
            results[model_type] = {
                'forecast_result': forecast_result,
                'trend_data': trend_data,
                'risk_data': risk_data,
                'opportunities': opportunities,
                'risks': risks
            }
            
            # Print summary
            print(f"\n📊 {model_type.upper()} FORECAST SUMMARY:")
            print(f"Trend: {trend_data['direction']}")
            print(f"Expected Return: {trend_data['total_change_pct']:.1f}%")
            print(f"Risk Level: {risk_data['risk_level']}")
            print(f"Volatility: {risk_data['volatility_pct']:.1f}%")
        else:
            print(f"❌ {model_type.upper()} model not available")
    
    # Create visualizations for available models
    print("\n📈 Creating forecast visualizations...")
    
    for model_type, result in results.items():
        print(f"\nVisualizing {model_type.upper()} forecast...")
        
        forecast_data = result['forecast_result']
        
        # Main forecast plot
        ForecastVisualizer.plot_forecast_with_confidence(
            forecast_data['historical_data'],
            forecast_data['forecast'],
            forecast_data.get('confidence_intervals'),
            f"TSLA - {model_type.upper()} 6-Month Forecast"
        )
        
        # Trend analysis plot
        ForecastVisualizer.plot_trend_analysis(
            forecast_data['forecast'],
            result['trend_data']
        )
        
        # Risk analysis plot
        ForecastVisualizer.plot_risk_analysis(
            forecast_data['forecast'],
            result['risk_data']
        )
        
        # Summary dashboard
        ForecastVisualizer.create_summary_dashboard(
            forecast_data['historical_data'],
            forecast_data['forecast'],
            result['trend_data'],
            result['risk_data'],
            result['opportunities'],
            result['risks']
        )
    
    # Compare models if both available
    if len(results) > 1:
        print("\n🔄 MODEL COMPARISON:")
        print("-" * 40)
        
        for model_type, result in results.items():
            trend = result['trend_data']
            risk = result['risk_data']
            print(f"\n{model_type.upper()}:")
            print(f"  Expected Return: {trend['total_change_pct']:.1f}%")
            print(f"  Risk Level: {risk['risk_level']}")
            print(f"  Volatility: {risk['volatility_pct']:.1f}%")
            print(f"  Opportunities: {len(result['opportunities'])}")
            print(f"  Risks: {len(result['risks'])}")
    
    # Generate investment recommendation
    print("\n💡 INVESTMENT RECOMMENDATIONS:")
    print("-" * 40)
    
    for model_type, result in results.items():
        trend = result['trend_data']
        risk = result['risk_data']
        
        if trend['total_change_pct'] > 10 and risk['risk_level'] != 'High':
            recommendation = "BUY - Positive outlook with manageable risk"
        elif trend['total_change_pct'] < -10:
            recommendation = "SELL - Negative outlook"
        elif risk['risk_level'] == 'High':
            recommendation = "HOLD - High uncertainty"
        else:
            recommendation = "HOLD - Neutral outlook"
        
        print(f"{model_type.upper()}: {recommendation}")
    
    print(f"\n✅ Task 3 demonstration completed!")
    print(f"📊 Generated forecasts and analysis for {len(results)} model(s)")
    
    return results

if __name__ == "__main__":
    run_task3_demo()
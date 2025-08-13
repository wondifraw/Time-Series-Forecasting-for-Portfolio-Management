"""
Future Market Trend Forecaster
Generates 6-12 month forecasts using trained models from Task 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Import Task 2 models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'task2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from task2.arima_model import ARIMAForecaster
    from task2.lstm_model import LSTMForecaster
    from task2.data_handler import TimeSeriesDataHandler
except ImportError:
    # Fallback for direct imports
    from arima_model import ARIMAForecaster
    from lstm_model import LSTMForecaster
    from data_handler import TimeSeriesDataHandler

class FutureMarketForecaster:
    """Generate and analyze future market trend forecasts."""
    
    def __init__(self, symbol: str = 'TSLA'):
        self.symbol = symbol
        self.data_handler = TimeSeriesDataHandler([symbol])
        self.arima_model = None
        self.lstm_model = None
        
    def load_trained_models(self) -> bool:
        """Load pre-trained models from Task 2."""
        try:
            # Load ARIMA model
            self.arima_model = ARIMAForecaster()
            arima_loaded = self.arima_model.load_model('../models/arima_model.pkl')
            
            # Load LSTM model
            self.lstm_model = LSTMForecaster()
            lstm_loaded = self.lstm_model.load_model('../models/lstm_model.h5')
            
            if arima_loaded or lstm_loaded:
                print(f"✅ Models loaded - ARIMA: {arima_loaded}, LSTM: {lstm_loaded}")
                return True
            else:
                print("❌ No trained models found. Please run Task 2 first.")
                return False
                
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False
    
    def generate_future_forecast(self, months: int = 6, model_type: str = 'arima') -> Dict:
        """Generate future forecasts for specified months."""
        if months < 6 or months > 12:
            print("⚠️ Forecast period should be 6-12 months")
            months = max(6, min(12, months))
        
        # Calculate forecast steps (trading days)
        forecast_steps = months * 21  # ~21 trading days per month
        
        # Load current data
        data = self.data_handler.load_processed_data(self.symbol)
        if data is None:
            return None
        
        # Generate forecast based on model type
        if model_type.lower() == 'arima' and self.arima_model and self.arima_model.is_fitted:
            return self._generate_arima_forecast(data, forecast_steps, months)
        elif model_type.lower() == 'lstm' and self.lstm_model and self.lstm_model.is_fitted:
            return self._generate_lstm_forecast(data, forecast_steps, months)
        else:
            print(f"❌ {model_type.upper()} model not available")
            return None
    
    def _generate_arima_forecast(self, data: pd.Series, steps: int, months: int) -> Dict:
        """Generate ARIMA-based future forecast."""
        try:
            # Fit model on full dataset
            if not self.arima_model.fit(data):
                return None
            
            # Generate forecast
            forecast_values = self.arima_model.predict(steps)
            if forecast_values is None:
                return None
            
            # Create future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=len(forecast_values),
                freq='B'  # Business days
            )
            
            forecast_series = pd.Series(forecast_values, index=future_dates)
            
            # Try to get confidence intervals
            conf_int_df = None
            try:
                forecast_result = self.arima_model.forecast_with_intervals(steps)
                if forecast_result and 'conf_int' in forecast_result:
                    conf_int_df = pd.DataFrame(
                        forecast_result['conf_int'],
                        index=future_dates,
                        columns=['lower', 'upper']
                    )
            except:
                pass
            
            result = {
                'model_type': 'ARIMA',
                'forecast': forecast_series,
                'forecast_months': months,
                'historical_data': data
            }
            
            if conf_int_df is not None:
                result['confidence_intervals'] = conf_int_df
            
            return result
            
        except Exception as e:
            print(f"❌ Error generating ARIMA forecast: {str(e)}")
            return None
    
    def _generate_lstm_forecast(self, data: pd.Series, steps: int, months: int) -> Dict:
        """Generate LSTM-based future forecast."""
        try:
            # Check if TensorFlow is available
            try:
                import tensorflow as tf
            except ImportError:
                print("❌ TensorFlow not available for LSTM forecasting")
                return None
            
            # Prepare data for LSTM
            sequence_length = 60
            if len(data) < sequence_length:
                print(f"❌ Insufficient data for LSTM (need {sequence_length}, have {len(data)})")
                return None
            
            data_values = data.values.reshape(-1, 1)
            scaled_data = self.lstm_model.data_handler.scaler.fit_transform(data_values)
            
            # Use last sequence as starting point
            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            
            # Generate iterative predictions
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(min(steps, 126)):  # Limit to 6 months max
                next_pred = self.lstm_model.model.predict(current_sequence, verbose=0)
                predictions.append(next_pred[0, 0])
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred[0, 0]
            
            # Inverse transform predictions
            predictions_array = np.array(predictions).reshape(-1, 1)
            forecast_values = self.lstm_model.data_handler.scaler.inverse_transform(predictions_array).flatten()
            
            # Create future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=len(predictions),
                freq='B'
            )
            
            forecast_series = pd.Series(forecast_values, index=future_dates)
            
            return {
                'model_type': 'LSTM',
                'forecast': forecast_series,
                'forecast_months': months,
                'historical_data': data
            }
            
        except Exception as e:
            print(f"❌ Error generating LSTM forecast: {str(e)}")
            return None
    
    def analyze_forecast_trends(self, forecast_result: Dict) -> Dict:
        """Analyze forecast for trends, volatility, and patterns."""
        if not forecast_result:
            return None
        
        forecast = forecast_result['forecast']
        historical = forecast_result['historical_data']
        
        # Calculate trend metrics
        current_price = historical.iloc[-1]
        forecast_end = forecast.iloc[-1]
        total_return = (forecast_end - current_price) / current_price * 100
        
        # Monthly returns
        monthly_points = len(forecast) // forecast_result['forecast_months']
        monthly_returns = []
        
        for i in range(forecast_result['forecast_months']):
            start_idx = i * monthly_points
            end_idx = min((i + 1) * monthly_points, len(forecast))
            if start_idx < len(forecast):
                start_price = forecast.iloc[start_idx] if i > 0 else current_price
                end_price = forecast.iloc[end_idx - 1]
                monthly_return = (end_price - start_price) / start_price * 100
                monthly_returns.append(monthly_return)
        
        # Volatility analysis
        forecast_returns = forecast.pct_change().dropna()
        forecast_volatility = forecast_returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Trend direction
        trend_direction = "Upward" if total_return > 5 else "Downward" if total_return < -5 else "Stable"
        
        return {
            'current_price': current_price,
            'forecast_end_price': forecast_end,
            'total_return_pct': total_return,
            'monthly_returns': monthly_returns,
            'forecast_volatility': forecast_volatility,
            'trend_direction': trend_direction,
            'max_price': forecast.max(),
            'min_price': forecast.min(),
            'price_range_pct': (forecast.max() - forecast.min()) / current_price * 100
        }
    
    def assess_confidence_intervals(self, forecast_result: Dict) -> Dict:
        """Analyze confidence interval behavior over forecast horizon."""
        if 'confidence_intervals' not in forecast_result:
            return {'analysis': 'No confidence intervals available for this model'}
        
        conf_int = forecast_result['confidence_intervals']
        forecast = forecast_result['forecast']
        
        # Calculate interval width over time
        interval_width = conf_int['upper'] - conf_int['lower']
        width_pct = interval_width / forecast * 100
        
        # Analyze width expansion
        initial_width = width_pct.iloc[:21].mean()  # First month
        final_width = width_pct.iloc[-21:].mean()   # Last month
        width_expansion = (final_width - initial_width) / initial_width * 100
        
        return {
            'initial_width_pct': initial_width,
            'final_width_pct': final_width,
            'width_expansion_pct': width_expansion,
            'avg_width_pct': width_pct.mean(),
            'reliability_assessment': self._assess_reliability(width_expansion)
        }
    
    def _assess_reliability(self, width_expansion: float) -> str:
        """Assess forecast reliability based on confidence interval expansion."""
        if width_expansion < 20:
            return "High reliability - confidence intervals remain stable"
        elif width_expansion < 50:
            return "Moderate reliability - some uncertainty increase over time"
        else:
            return "Low reliability - significant uncertainty in long-term forecasts"
    
    def identify_opportunities_risks(self, analysis: Dict, conf_analysis: Dict = None) -> Dict:
        """Identify market opportunities and risks from forecast analysis."""
        opportunities = []
        risks = []
        
        # Price-based opportunities and risks
        if analysis['total_return_pct'] > 10:
            opportunities.append(f"Strong upward trend: {analysis['total_return_pct']:.1f}% expected return")
        elif analysis['total_return_pct'] < -10:
            risks.append(f"Significant decline expected: {analysis['total_return_pct']:.1f}% potential loss")
        
        # Volatility assessment
        if analysis['forecast_volatility'] > 50:
            risks.append(f"High volatility: {analysis['forecast_volatility']:.1f}% annualized")
            opportunities.append("High volatility creates trading opportunities for active investors")
        elif analysis['forecast_volatility'] < 20:
            opportunities.append(f"Low volatility: {analysis['forecast_volatility']:.1f}% provides stability")
        
        # Monthly performance patterns
        positive_months = sum(1 for ret in analysis['monthly_returns'] if ret > 0)
        if positive_months >= len(analysis['monthly_returns']) * 0.7:
            opportunities.append(f"Consistent growth: {positive_months}/{len(analysis['monthly_returns'])} positive months")
        
        # Price range analysis
        if analysis['price_range_pct'] > 30:
            opportunities.append("Wide price range suggests potential for significant gains")
            risks.append("Wide price range indicates high uncertainty")
        
        # Confidence interval risks
        if conf_analysis and 'width_expansion_pct' in conf_analysis:
            if conf_analysis['width_expansion_pct'] > 50:
                risks.append("Forecast uncertainty increases significantly over time")
        
        return {
            'opportunities': opportunities,
            'risks': risks,
            'investment_recommendation': self._generate_recommendation(analysis, opportunities, risks)
        }
    
    def _generate_recommendation(self, analysis: Dict, opportunities: list, risks: list) -> str:
        """Generate investment recommendation based on analysis."""
        if analysis['total_return_pct'] > 15 and analysis['forecast_volatility'] < 40:
            return "BUY - Strong upward trend with manageable risk"
        elif analysis['total_return_pct'] < -15:
            return "SELL - Significant downside risk identified"
        elif analysis['forecast_volatility'] > 60:
            return "HOLD - High volatility suggests waiting for clearer signals"
        elif len(opportunities) > len(risks):
            return "BUY - Opportunities outweigh risks"
        elif len(risks) > len(opportunities):
            return "HOLD - Risks outweigh opportunities"
        else:
            return "HOLD - Balanced risk-reward profile"
    
    def visualize_forecast(self, forecast_result: Dict, analysis: Dict):
        """Create comprehensive forecast visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Main forecast plot
        historical = forecast_result['historical_data']
        forecast = forecast_result['forecast']
        
        # Plot historical data (last 6 months)
        recent_historical = historical.iloc[-126:]  # ~6 months
        axes[0, 0].plot(recent_historical.index, recent_historical.values, 
                       label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        axes[0, 0].plot(forecast.index, forecast.values, 
                       label=f'{forecast_result["model_type"]} Forecast', 
                       color='red', linewidth=2)
        
        # Plot confidence intervals if available
        if 'confidence_intervals' in forecast_result:
            conf_int = forecast_result['confidence_intervals']
            axes[0, 0].fill_between(conf_int.index, conf_int['lower'], conf_int['upper'],
                                   alpha=0.3, color='red', label='Confidence Interval')
        
        axes[0, 0].set_title(f'{self.symbol} - {forecast_result["forecast_months"]} Month Forecast')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Monthly returns
        axes[0, 1].bar(range(1, len(analysis['monthly_returns']) + 1), 
                      analysis['monthly_returns'], 
                      color=['green' if x > 0 else 'red' for x in analysis['monthly_returns']])
        axes[0, 1].set_title('Forecasted Monthly Returns')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Price distribution
        axes[1, 0].hist(forecast.values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(analysis['current_price'], color='blue', linestyle='--', 
                          label=f'Current: ${analysis["current_price"]:.2f}')
        axes[1, 0].axvline(analysis['forecast_end_price'], color='red', linestyle='--', 
                          label=f'Forecast End: ${analysis["forecast_end_price"]:.2f}')
        axes[1, 0].set_title('Forecast Price Distribution')
        axes[1, 0].set_xlabel('Price ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Key metrics summary
        axes[1, 1].axis('off')
        metrics_text = f"""
        Key Forecast Metrics:
        
        Current Price: ${analysis['current_price']:.2f}
        Forecast End: ${analysis['forecast_end_price']:.2f}
        Total Return: {analysis['total_return_pct']:.1f}%
        
        Trend: {analysis['trend_direction']}
        Volatility: {analysis['forecast_volatility']:.1f}%
        
        Price Range: ${analysis['min_price']:.2f} - ${analysis['max_price']:.2f}
        Range %: {analysis['price_range_pct']:.1f}%
        """
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def generate_forecast_report(self, months: int = 6, model_type: str = 'arima') -> Dict:
        """Generate comprehensive forecast report."""
        print(f"🔮 FUTURE MARKET TREND ANALYSIS - {self.symbol}")
        print("=" * 50)
        
        # Load models
        if not self.load_trained_models():
            return None
        
        # Generate forecast
        print(f"\n📈 Generating {months}-month forecast using {model_type.upper()} model...")
        forecast_result = self.generate_future_forecast(months, model_type)
        
        if not forecast_result:
            return None
        
        # Analyze trends
        print("\n📊 Analyzing forecast trends...")
        trend_analysis = self.analyze_forecast_trends(forecast_result)
        
        # Analyze confidence intervals
        conf_analysis = self.assess_confidence_intervals(forecast_result)
        
        # Identify opportunities and risks
        print("\n⚖️ Assessing opportunities and risks...")
        opp_risk_analysis = self.identify_opportunities_risks(trend_analysis, conf_analysis)
        
        # Print summary
        self._print_forecast_summary(trend_analysis, conf_analysis, opp_risk_analysis)
        
        # Visualize results
        print("\n📊 Generating forecast visualization...")
        self.visualize_forecast(forecast_result, trend_analysis)
        
        return {
            'forecast_result': forecast_result,
            'trend_analysis': trend_analysis,
            'confidence_analysis': conf_analysis,
            'opportunities_risks': opp_risk_analysis
        }
    
    def _print_forecast_summary(self, trend_analysis: Dict, conf_analysis: Dict, opp_risk: Dict):
        """Print formatted forecast summary."""
        print(f"\n📋 FORECAST SUMMARY")
        print("-" * 30)
        print(f"Current Price: ${trend_analysis['current_price']:.2f}")
        print(f"Forecast End Price: ${trend_analysis['forecast_end_price']:.2f}")
        print(f"Expected Return: {trend_analysis['total_return_pct']:.1f}%")
        print(f"Trend Direction: {trend_analysis['trend_direction']}")
        print(f"Forecast Volatility: {trend_analysis['forecast_volatility']:.1f}%")
        
        if 'reliability_assessment' in conf_analysis:
            print(f"\nConfidence Assessment: {conf_analysis['reliability_assessment']}")
        
        print(f"\n🎯 OPPORTUNITIES:")
        for opp in opp_risk['opportunities']:
            print(f"  • {opp}")
        
        print(f"\n⚠️ RISKS:")
        for risk in opp_risk['risks']:
            print(f"  • {risk}")
        
        print(f"\n💡 RECOMMENDATION: {opp_risk['investment_recommendation']}")
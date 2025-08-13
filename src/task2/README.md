# Task 2: Modular Time Series Forecasting

## Overview

Enhanced modular implementation of time series forecasting models with clean architecture, configuration management, and factory patterns.

## Architecture

```
task2/
├── base_model.py          # Abstract base class for all models
├── data_handler.py        # Centralized data operations
├── model_evaluator.py     # Standardized evaluation utilities
├── config.py             # Configuration management
├── model_factory.py      # Factory pattern for model creation
├── arima_model.py        # ARIMA implementation
├── lstm_model.py         # LSTM implementation  
├── model_comparison.py   # Model comparison utilities
├── forecasting_pipeline.py # Complete pipeline orchestrator
└── __init__.py           # Module exports and factory registration
```

## Key Features

### 🏗️ Modular Design
- **Base Classes**: Common interface for all forecasting models
- **Separation of Concerns**: Data handling, evaluation, and modeling separated
- **Factory Pattern**: Easy model instantiation and management
- **Configuration Management**: Centralized parameter control

### 📊 Data Handling
- **Centralized Data Operations**: Single source for data loading and preprocessing
- **Automatic Scaling**: Built-in MinMax scaling for LSTM models
- **Sequence Generation**: Automated sequence creation for time series models

### 🔧 Configuration System
- **Type-Safe Configuration**: Dataclass-based configuration with validation
- **Default Settings**: Sensible defaults with easy customization
- **Hierarchical Structure**: Separate configs for ARIMA, LSTM, and data

### 🏭 Factory Pattern
- **Dynamic Model Creation**: Create models by name with consistent interface
- **Extensible**: Easy to add new model types
- **Configuration Integration**: Automatic config injection

## Quick Start

```python
# Ultra-simple usage
from task2 import forecast
results = forecast('TSLA')

# Quick metrics only
from task2 import quick_forecast
metrics = quick_forecast('TSLA')
print(f"ARIMA MAE: {metrics['ARIMA_MAE']:.4f}")

# Pipeline with custom config
from task2 import ForecastingPipeline, ConfigManager
config = ConfigManager.get_default_config()
config.lstm.epochs = 100

pipeline = ForecastingPipeline(config=config)
results = pipeline.run('TSLA')
```

## Usage Patterns

### Simple Functions
```python
# One-liner forecasting
from task2 import forecast, compare_models
results = forecast('TSLA', models=['ARIMA'], epochs=30)
best_model = compare_models('TSLA')
```

### Pipeline Interface
```python
from task2 import ForecastingPipeline
pipeline = ForecastingPipeline()

# Run all models
results = pipeline.run('TSLA')

# Run specific models
results = pipeline.run('TSLA', models=['ARIMA'], plot=False)
```

### Individual Models
```python
from task2 import ModelFactory, ConfigManager
config = ConfigManager.get_default_config()
arima_model = ModelFactory.create_model('ARIMA', config)

data = arima_model.load_data()
train_data, test_data = arima_model.prepare_data(data)
arima_model.fit(train_data)
predictions = arima_model.predict(len(test_data))
```

## Configuration Options

### ARIMA Configuration
```python
@dataclass
class ARIMAConfig:
    max_p: int = 5          # Maximum AR order
    max_d: int = 2          # Maximum differencing
    max_q: int = 5          # Maximum MA order
    alpha: float = 0.05     # Stationarity test significance
```

### LSTM Configuration
```python
@dataclass 
class LSTMConfig:
    sequence_length: int = 60        # Input sequence length
    epochs: int = 50                 # Training epochs
    batch_size: int = 32            # Batch size
    learning_rate: float = 0.001    # Learning rate
    dropout_rate: float = 0.2       # Dropout rate
    lstm_units: List[int] = [50, 50, 50]  # LSTM layer units
```

## Model Comparison

```python
from task2 import ModelComparison

comparison = ModelComparison()
comparison.add_model_results('ARIMA', arima_predictions, actual_values)
comparison.add_model_results('LSTM', lstm_predictions, actual_values)

# Generate comprehensive report
report = comparison.generate_comparison_report()

# Visualizations
comparison.plot_predictions_comparison()
comparison.plot_metrics_comparison()
comparison.plot_residuals_analysis()
```

## Benefits of Modular Design

### 🔄 Reusability
- Components can be used independently
- Easy to extend with new models
- Consistent interfaces across models

### 🧪 Testability  
- Each module can be tested in isolation
- Mock dependencies for unit testing
- Clear separation of concerns

### 🔧 Maintainability
- Changes isolated to specific modules
- Easy to debug and troubleshoot
- Clear code organization

### 📈 Extensibility
- Add new models by implementing BaseForecaster
- Register with factory for automatic integration
- Extend configuration as needed

## Running Examples

```bash
# Run the modular demo
python scripts/task2_modular_demo.py

# Quick Python usage
python -c "from task2 import quick_forecast; print(quick_forecast('TSLA'))"
```

## Migration Guide

The refactored code maintains backward compatibility while providing new modular features:

### Old Way
```python
from src.task2.arima_model import ARIMAForecaster
arima = ARIMAForecaster(['TSLA'])
```

### New Way  
```python
from task2 import ModelFactory, ConfigManager
config = ConfigManager.get_default_config()
arima = ModelFactory.create_model('ARIMA', config, symbols=['TSLA'])
```

Both approaches work, but the new way provides better configuration management and consistency.
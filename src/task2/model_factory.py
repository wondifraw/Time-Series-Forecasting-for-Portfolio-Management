"""
Model Factory for Time Series Forecasting
Implements factory pattern for easy model instantiation and management
"""

from typing import Dict, Any, Optional
from .base_model import BaseForecaster
from .config import ModelConfig, ConfigManager

class ModelFactory:
    """Factory for creating forecasting models."""
    
    _models = {}
    
    @classmethod
    def register_model(cls, name: str, model_class):
        """Register a new model type."""
        cls._models[name] = model_class
    
    @classmethod
    def create_model(cls, model_type: str, config: Optional[ModelConfig] = None, 
                    **kwargs) -> BaseForecaster:
        """Create a model instance."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if config is None:
            config = ConfigManager.get_default_config()
        
        model_class = cls._models[model_type]
        return model_class(config=config, **kwargs)
    
    @classmethod
    def list_available_models(cls) -> list:
        """List all available model types."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model type."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
        return {
            'name': model_type,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'docstring': model_class.__doc__
        }
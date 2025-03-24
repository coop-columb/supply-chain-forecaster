from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


class BaseModel(ABC):
    """Base class for all forecasting models."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize a new forecasting model.

        Args:
            name: Unique name for the model instance
            config: Dictionary of model configuration parameters
        """
        self.name = name
        self.config = config
        self.model = None
        self.is_fitted = False
        self.metadata = {
            "creation_time": datetime.now().isoformat(),
            "last_fit_time": None,
            "model_type": self.__class__.__name__,
        }

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseModel":
        """
        Fit the model to the provided data.

        Args:
            X: Feature data
            y: Target data (may be None for some models like Prophet that use a single DataFrame)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate predictions for the provided data.

        Args:
            X: Feature data for prediction

        Returns:
            Forecasted values
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the underlying model.

        Returns:
            Dictionary of model parameters
        """
        pass

    def save(self, path: str) -> str:
        """
        Save the model to disk.

        Args:
            path: Directory path to save the model

        Returns:
            Full path to the saved model file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save model that has not been fitted")

        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}.joblib"
        full_path = os.path.join(path, filename)

        # Save model with its metadata
        model_data = {
            "model": self.model,
            "metadata": self.metadata,
            "config": self.config,
            "class_name": self.__class__.__name__,
        }

        joblib.dump(model_data, full_path)
        return full_path

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Load a model from disk.

        Args:
            path: Path to the saved model file

        Returns:
            Loaded model instance
        """
        model_data = joblib.load(path)

        # Check if the loaded model class matches
        if model_data["class_name"] != cls.__name__:
            raise ValueError(
                f"Loaded model is of type {model_data['class_name']}, not {cls.__name__}"
            )

        # Create a new instance
        instance = cls(
            name=model_data["metadata"].get("model_name", "loaded_model"),
            config=model_data["config"],
        )

        # Restore model state
        instance.model = model_data["model"]
        instance.metadata = model_data["metadata"]
        instance.is_fitted = True

        return instance

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if the model supports it.

        Returns:
            DataFrame with feature importance or None if not supported
        """
        return None

    def __str__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"

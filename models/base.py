"""Base model classes for the supply chain forecaster."""

import datetime
import inspect
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

import joblib
import numpy as np
import pandas as pd

from config import config
from utils import ModelError, NotFoundError, get_logger, safe_execute

logger = get_logger(__name__)


class ModelBase(ABC):
    """Base class for all models."""

    def __init__(self, name: str = None, **kwargs):
        """
        Initialize the model.
        
        Args:
            name: Name of the model.
            **kwargs: Additional model parameters.
        """
        self.name = name or self.__class__.__name__
        self.model = None
        self.features = None
        self.target = None
        self.params = kwargs
        self.metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "model_type": self.__class__.__name__,
            "parameters": self.params,
        }
        logger.info(f"Initialized {self.name} model")

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ModelBase":
        """
        Fit the model to the data.
        
        Args:
            X: Feature dataframe.
            y: Target series.
            **kwargs: Additional fitting parameters.
        
        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions using the model.
        
        Args:
            X: Feature dataframe.
            **kwargs: Additional prediction parameters.
        
        Returns:
            Predicted values.
        """
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series, metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Feature dataframe.
            y: Target series.
            metrics: List of metrics to compute.
        
        Returns:
            Dictionary of metric values.
        """
        from utils.evaluation import calculate_metrics
        
        if self.model is None:
            raise ModelError("Model has not been fitted", model_name=self.name)
        
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate metrics
        results = calculate_metrics(y, predictions, metrics)
        
        return results

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model. If None, uses the default directory.
        
        Returns:
            Path to the saved model.
        """
        if self.model is None:
            raise ModelError("Model has not been fitted", model_name=self.name)
        
        # Use default path if not provided
        if path is None:
            path = config.TRAINING_MODELS_DIR / f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        else:
            path = Path(path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        self.metadata.update({
            "saved_at": datetime.datetime.now().isoformat(),
            "features": list(self.features) if self.features is not None else None,
            "target": self.target,
            "path": str(path),
        })
        
        # Create a dictionary with everything to save
        model_data = {
            "model": self.model,
            "features": self.features,
            "target": self.target,
            "params": self.params,
            "metadata": self.metadata,
        }
        
        # Save the model
        logger.info(f"Saving {self.name} model to {path}")
        joblib.dump(model_data, path)
        
        # Save metadata separately for easier access
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelBase":
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model.
        
        Returns:
            Loaded model instance.
        """
        path = Path(path)
        if not path.exists():
            raise NotFoundError("Model file", str(path))
        
        logger.info(f"Loading model from {path}")
        
        # Load the model data
        model_data = joblib.load(path)
        
        # Create a new instance of the model
        model_instance = cls(**model_data["params"])
        
        # Set model attributes
        model_instance.model = model_data["model"]
        model_instance.features = model_data["features"]
        model_instance.target = model_data["target"]
        model_instance.metadata = model_data["metadata"]
        
        logger.info(f"Loaded {model_instance.name} model from {path}")
        
        return model_instance


class ModelRegistry:
    """Registry for tracking and managing models."""
    
    _models: Dict[str, Type[ModelBase]] = {}
    
    @classmethod
    def register(cls, model_class: Type[ModelBase]) -> Type[ModelBase]:
        """
        Register a model class.
        
        Args:
            model_class: Model class to register.
        
        Returns:
            The registered model class.
        """
        if not inspect.isclass(model_class) or not issubclass(model_class, ModelBase):
            raise TypeError(f"{model_class} is not a subclass of ModelBase")
        
        model_name = model_class.__name__
        cls._models[model_name] = model_class
        logger.info(f"Registered model: {model_name}")
        return model_class
    
    @classmethod
    def get_model(cls, model_name: str) -> Type[ModelBase]:
        """
        Get a registered model class by name.
        
        Args:
            model_name: Name of the model class.
        
        Returns:
            The registered model class.
        """
        if model_name not in cls._models:
            raise NotFoundError("Model", model_name)
        return cls._models[model_name]
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all registered model names.
        
        Returns:
            List of registered model names.
        """
        return list(cls._models.keys())
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> ModelBase:
        """
        Create an instance of a registered model.
        
        Args:
            model_name: Name of the model class.
            **kwargs: Additional model parameters.
        
        Returns:
            An instance of the specified model.
        """
        model_class = cls.get_model(model_name)
        return model_class(**kwargs)
    
    @classmethod
    def find_models(
        cls, directory: Optional[Union[str, Path]] = None, pattern: str = "*.joblib"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Find saved models in a directory.
        
        Args:
            directory: Directory to search for models. If None, uses the default directory.
            pattern: File pattern to match.
        
        Returns:
            Dictionary of model metadata.
        """
        directory = Path(directory) if directory else config.TRAINING_MODELS_DIR
        if not directory.exists():
            logger.warning(f"Model directory {directory} does not exist")
            return {}
        
        models = {}
        
        # Find all model files
        model_files = list(directory.glob(pattern))
        logger.info(f"Found {len(model_files)} model files in {directory}")
        
        for model_file in model_files:
            metadata_file = model_file.with_suffix(".json")
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    models[model_file.stem] = metadata
                except Exception as e:
                    logger.error(f"Error loading metadata for {model_file}: {str(e)}")
            else:
                try:
                    model_data = joblib.load(model_file)
                    if isinstance(model_data, dict) and "metadata" in model_data:
                        models[model_file.stem] = model_data["metadata"]
                    else:
                        logger.warning(f"No metadata found for {model_file}")
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {str(e)}")
        
        return models
    
    @classmethod
    def deploy_model(
        cls, model_path: Union[str, Path], target_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Deploy a model to the deployment directory.
        
        Args:
            model_path: Path to the model to deploy.
            target_dir: Directory to deploy the model to. If None, uses the default directory.
        
        Returns:
            Path to the deployed model.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise NotFoundError("Model file", str(model_path))
        
        target_dir = Path(target_dir) if target_dir else config.DEPLOYMENT_MODELS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Deploy the model
        target_path = target_dir / f"{model_path.stem}.joblib"
        logger.info(f"Deploying model from {model_path} to {target_path}")
        
        # Copy the model file
        import shutil
        shutil.copy2(model_path, target_path)
        
        # Copy the metadata file if it exists
        metadata_path = model_path.with_suffix(".json")
        if metadata_path.exists():
            shutil.copy2(metadata_path, target_path.with_suffix(".json"))
        
        logger.info(f"Model deployed to {target_path}")
        
        return target_path

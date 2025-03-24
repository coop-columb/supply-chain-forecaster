import importlib
import inspect
import os
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from models.base import BaseModel


class ModelRegistry:
    """Registry for managing and accessing different model implementations."""

    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, Type[BaseModel]] = {}

    def register(self, model_class: Type[BaseModel]) -> None:
        """
        Register a model class with the registry.

        Args:
            model_class: Model class to register
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class.__name__} must be a subclass of BaseModel")

        model_name = model_class.__name__
        self._models[model_name] = model_class

    def get_model_class(self, model_name: str) -> Type[BaseModel]:
        """
        Get a model class by name.

        Args:
            model_name: Name of the model class

        Returns:
            Model class

        Raises:
            KeyError: If model_name is not registered
        """
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found in registry")
        return self._models[model_name]

    def create_model(
        self, model_name: str, instance_name: str, config: Dict[str, Any]
    ) -> BaseModel:
        """
        Create a new instance of a registered model.

        Args:
            model_name: Name of the model class
            instance_name: Name for the model instance
            config: Configuration for the model instance

        Returns:
            Instantiated model
        """
        model_class = self.get_model_class(model_name)
        return model_class(name=instance_name, config=config)

    def list_available_models(self) -> List[str]:
        """
        List all registered model classes.

        Returns:
            List of model class names
        """
        return list(self._models.keys())

    def auto_discover(self, package_path: Optional[str] = None) -> None:
        """
        Automatically discover and register model implementations.

        Args:
            package_path: Path to package containing model implementations.
                          If None, uses the 'models.training' package.
        """
        if package_path is None:
            package_name = "models.training"
        else:
            package_name = package_path

        try:
            package = importlib.import_module(package_name)
        except ImportError:
            print(f"Unable to import package {package_name}")
            return

        # Get the directory for the package
        if hasattr(package, "__path__"):
            for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
                full_name = f"{package_name}.{name}"
                try:
                    module = importlib.import_module(full_name)

                    # Look for BaseModel subclasses in the module
                    for item_name, item in inspect.getmembers(module):
                        if (
                            inspect.isclass(item)
                            and issubclass(item, BaseModel)
                            and item is not BaseModel
                        ):
                            self.register(item)
                except ImportError:
                    print(f"Failed to import {full_name}")

    def __len__(self) -> int:
        """Get the number of registered models."""
        return len(self._models)


# Singleton instance of the model registry
model_registry = ModelRegistry()

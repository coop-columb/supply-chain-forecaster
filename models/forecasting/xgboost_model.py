"""XGBoost forecasting model for the supply chain forecaster."""

import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from models.base import ModelBase, ModelRegistry
from utils import get_logger, safe_execute

logger = get_logger(__name__)


@ModelRegistry.register
class XGBoostModel(ModelBase):
    """XGBoost regression model for time series forecasting."""

    def __init__(
        self,
        name: str = "XGBoostModel",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        objective: str = "reg:squarederror",
        booster: str = "gbtree",
        tree_method: str = "auto",
        **kwargs,
    ):
        """
        Initialize the XGBoost model.

        Args:
            name: Name of the model.
            n_estimators: Number of gradient boosted trees.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate.
            subsample: Subsample ratio of the training instances.
            colsample_bytree: Subsample ratio of columns when constructing each tree.
            objective: Objective function.
            booster: Booster type.
            tree_method: Tree construction algorithm.
            **kwargs: Additional XGBoost parameters.
        """
        try:
            import xgboost
        except ImportError:
            logger.error(
                "XGBoost not installed. Please install with pip install xgboost"
            )
            raise

        super().__init__(
            name=name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            booster=booster,
            tree_method=tree_method,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[tuple]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ) -> "XGBoostModel":
        """
        Fit the XGBoost model to the data.

        Args:
            X: Feature dataframe.
            y: Target series.
            eval_set: Evaluation set for early stopping.
            early_stopping_rounds: Number of rounds for early stopping.
            verbose: Whether to print training progress.
            **kwargs: Additional fitting parameters.

        Returns:
            Self for method chaining.
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error(
                "XGBoost not installed. Please install with pip install xgboost"
            )
            raise

        logger.info(f"Fitting XGBoost model {self.name}")

        # Store feature names and target name
        self.features = list(X.columns)
        self.target = y.name if y.name else "target"

        # Initialize model
        model = xgb.XGBRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            subsample=self.params["subsample"],
            colsample_bytree=self.params["colsample_bytree"],
            objective=self.params["objective"],
            booster=self.params["booster"],
            tree_method=self.params["tree_method"],
            **kwargs,
        )

        # Fit the model
        model.fit(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        self.model = model

        # Update metadata
        self.metadata.update(
            {
                "fitted_at": datetime.datetime.now().isoformat(),
                "data_shape": X.shape,
                "target_mean": float(y.mean()),
                "target_std": float(y.std()),
                "feature_importance": dict(zip(X.columns, model.feature_importances_)),
            }
        )

        # Add best iteration if early stopping was used
        if hasattr(model, "best_iteration"):
            self.metadata["best_iteration"] = model.best_iteration

        logger.info(f"Successfully fitted XGBoost model {self.name}")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the XGBoost model.

        Args:
            X: Feature dataframe.
            **kwargs: Additional prediction parameters.

        Returns:
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")

        logger.info(f"Making predictions with XGBoost model {self.name}")

        # Make predictions
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.

        Returns:
            Dataframe with feature importance.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")

        importance = self.model.feature_importances_
        importance_df = pd.DataFrame(
            {
                "Feature": self.features,
                "Importance": importance,
            }
        )
        importance_df = importance_df.sort_values("Importance", ascending=False)

        return importance_df

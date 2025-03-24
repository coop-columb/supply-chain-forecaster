"""LSTM forecasting model for the supply chain forecaster."""

import datetime
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from models.base import ModelBase, ModelRegistry
from utils import get_logger, safe_execute

logger = get_logger(__name__)


@ModelRegistry.register
class LSTMModel(ModelBase):
    """LSTM neural network model for time series forecasting."""

    def __init__(
        self,
        name: str = "LSTMModel",
        units: List[int] = [64, 32],
        dropout: float = 0.2,
        recurrent_dropout: float = 0.0,
        activation: str = "relu",
        optimizer: str = "adam",
        loss: str = "mse",
        metrics: List[str] = ["mae"],
        batch_size: int = 32,
        epochs: int = 50,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10,
        sequence_length: int = 10,
        **kwargs,
    ):
        """
        Initialize the LSTM model.
        
        Args:
            name: Name of the model.
            units: List of LSTM layer sizes.
            dropout: Dropout rate.
            recurrent_dropout: Recurrent dropout rate.
            activation: Activation function.
            optimizer: Optimizer.
            loss: Loss function.
            metrics: List of metrics to track.
            batch_size: Batch size for training.
            epochs: Number of epochs to train.
            validation_split: Fraction of data to use for validation.
            early_stopping: Whether to use early stopping.
            patience: Number of epochs with no improvement for early stopping.
            sequence_length: Sequence length for LSTM input.
            **kwargs: Additional LSTM parameters.
        """
        try:
            import tensorflow as tf
        except ImportError:
            logger.error(
                "TensorFlow not installed. Please install with pip install tensorflow"
            )
            raise
        
        super().__init__(
            name=name,
            units=units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            activation=activation,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            early_stopping=early_stopping,
            patience=patience,
            sequence_length=sequence_length,
            **kwargs,
        )
        
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        
        # Store preprocessing parameters
        self.scaler = None

    def _create_sequences(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input.
        
        Args:
            X: Feature dataframe.
            y: Target series (optional).
        
        Returns:
            Tuple of input sequences and target sequences (if y is provided).
        """
        seq_length = self.params["sequence_length"]
        
        # Convert to numpy arrays
        X_array = X.values
        
        n_samples = len(X_array)
        n_features = X_array.shape[1]
        
        # Create sequences
        X_sequences = []
        
        for i in range(n_samples - seq_length):
            X_sequences.append(X_array[i:i+seq_length])
        
        X_sequences = np.array(X_sequences)
        
        # Create target sequences if y is provided
        if y is not None:
            y_array = y.values
            y_sequences = y_array[seq_length:]
            return X_sequences, y_sequences
        else:
            return X_sequences, None

    def _build_model(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of the input data (sequence_length, n_features).
        
        Returns:
            Compiled LSTM model.
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.params["units"][0],
            activation=self.params["activation"],
            dropout=self.params["dropout"],
            recurrent_dropout=self.params["recurrent_dropout"],
            return_sequences=len(self.params["units"]) > 1,
            input_shape=input_shape,
        ))
        
        # Additional LSTM layers
        for i, units in enumerate(self.params["units"][1:]):
            return_sequences = i < len(self.params["units"]) - 2
            model.add(LSTM(
                units=units,
                activation=self.params["activation"],
                dropout=self.params["dropout"],
                recurrent_dropout=self.params["recurrent_dropout"],
                return_sequences=return_sequences,
            ))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=self.params["optimizer"],
            loss=self.params["loss"],
            metrics=self.params["metrics"],
        )
        
        return model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale_data: bool = True,
        **kwargs,
    ) -> "LSTMModel":
        """
        Fit the LSTM model to the data.
        
        Args:
            X: Feature dataframe.
            y: Target series.
            scale_data: Whether to scale the data.
            **kwargs: Additional fitting parameters.
        
        Returns:
            Self for method chaining.
        """
        try:
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            logger.error(
                "TensorFlow or scikit-learn not installed. "
                "Please install with pip install tensorflow scikit-learn"
            )
            raise
        
        logger.info(f"Fitting LSTM model {self.name}")
        
        # Store feature names and target name
        self.features = list(X.columns)
        self.target = y.name if y.name else "target"
        
        # Scale data if requested
        X_scaled = X.copy()
        y_scaled = y.copy()
        
        if scale_data:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            X_scaled = pd.DataFrame(
                self.scaler_X.fit_transform(X),
                columns=X.columns,
                index=X.index,
            )
            
            y_scaled = pd.Series(
                self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(),
                index=y.index,
                name=y.name,
            )
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        if len(X_seq) == 0:
            raise ValueError(
                f"Not enough data to create sequences with sequence_length={self.params['sequence_length']}"
            )
        
        logger.info(f"Created {len(X_seq)} sequences of length {self.params['sequence_length']}")
        
        # Build model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        model = self._build_model(input_shape)
        
        # Create callbacks
        callbacks = []
        
        if self.params["early_stopping"]:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.params["patience"],
                restore_best_weights=True,
            )
            callbacks.append(early_stopping)
        
        # Train model
        history = model.fit(
            X_seq, y_seq,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            validation_split=self.params["validation_split"],
            callbacks=callbacks,
            verbose=1,
            **kwargs,
        )
        
        self.model = model
        
        # Update metadata
        self.metadata.update({
            "fitted_at": datetime.datetime.now().isoformat(),
            "data_shape": X.shape,
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
            "train_loss": float(history.history["loss"][-1]),
            "train_metrics": {m: float(history.history[m][-1]) for m in self.params["metrics"]},
            "val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
            "val_metrics": {f"val_{m}": float(history.history[f"val_{m}"][-1]) for m in self.params["metrics"] if f"val_{m}" in history.history},
            "epochs_trained": len(history.history["loss"]),
        })
        
        logger.info(f"Successfully fitted LSTM model {self.name}")
        
        return self

    def predict(
        self,
        X: pd.DataFrame,
        steps_ahead: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the LSTM model.
        
        Args:
            X: Feature dataframe.
            steps_ahead: Number of steps to predict ahead.
            **kwargs: Additional prediction parameters.
        
        Returns:
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")
        
        logger.info(f"Making predictions with LSTM model {self.name}")
        
        # Scale data if scaler exists
        X_scaled = X.copy()
        
        if hasattr(self, "scaler_X") and self.scaler_X is not None:
            X_scaled = pd.DataFrame(
                self.scaler_X.transform(X),
                columns=X.columns,
                index=X.index,
            )
        
        seq_length = self.params["sequence_length"]
        
        if steps_ahead == 1 and len(X) >= seq_length:
            # Simple prediction using the last sequence
            X_seq, _ = self._create_sequences(X_scaled)
            
            if len(X_seq) == 0:
                # If we can't create full sequences, use the last partial sequence
                X_array = X_scaled.values
                if len(X_array) < seq_length:
                    # Pad with zeros if needed
                    padding = np.zeros((seq_length - len(X_array), X_array.shape[1]))
                    X_seq = np.array([np.vstack((padding, X_array))])
                else:
                    X_seq = np.array([X_array[-seq_length:]])
            
            predictions = self.model.predict(X_seq)
            
            # Unscale predictions if scaler exists
            if hasattr(self, "scaler_y") and self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions).flatten()
            else:
                predictions = predictions.flatten()
            
            # Extend predictions to match input length
            if len(predictions) < len(X):
                padding = np.full(len(X) - len(predictions), np.nan)
                predictions = np.concatenate([padding, predictions])
            
            return predictions
        
        elif steps_ahead > 1:
            # Multi-step forecasting
            predictions = []
            
            # Get the initial sequence
            X_array = X_scaled.values
            
            if len(X_array) < seq_length:
                # Pad with zeros if needed
                padding = np.zeros((seq_length - len(X_array), X_array.shape[1]))
                current_seq = np.vstack((padding, X_array))
            else:
                current_seq = X_array[-seq_length:]
            
            # Generate predictions one step at a time
            for _ in range(steps_ahead):
                # Reshape for prediction
                X_seq = np.array([current_seq])
                
                # Predict next value
                next_value = self.model.predict(X_seq)[0, 0]
                predictions.append(next_value)
                
                # Update sequence
                # This assumes the target is the first feature, adjust as needed
                current_seq = np.vstack((current_seq[1:], [X_array[-1]]))
                current_seq[-1, 0] = next_value
            
            # Unscale predictions if scaler exists
            predictions = np.array(predictions).reshape(-1, 1)
            if hasattr(self, "scaler_y") and self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions).flatten()
            else:
                predictions = predictions.flatten()
            
            return predictions
        
        else:
            raise ValueError("Not enough data to create sequences for prediction")

    def save(self, path: Optional[Union[str, os.PathLike]] = None) -> os.PathLike:
        """
        Save the LSTM model to disk.
        
        Args:
            path: Path to save the model. If None, uses the default directory.
        
        Returns:
            Path to the saved model.
        """
        import tensorflow as tf

        # Use default path if not provided
        if path is None:
            path = self.metadata.get("path")
            if path is None:
                path = os.path.join(
                    "models",
                    "training",
                    f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save TensorFlow model separately
        if self.model is not None:
            tf_model_path = f"{path}_tf_model"
            self.model.save(tf_model_path)
            logger.info(f"Saved TensorFlow model to {tf_model_path}")
            
            # Update metadata with TensorFlow model path
            self.metadata["tf_model_path"] = tf_model_path
        
        # Save everything else using joblib
        return super().save(path)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "LSTMModel":
        """
        Load an LSTM model from disk.
        
        Args:
            path: Path to the saved model.
        
        Returns:
            Loaded model instance.
        """
        import tensorflow as tf

        # Load metadata and other components
        model_instance = super().load(path)
        
        # Load TensorFlow model if available
        tf_model_path = model_instance.metadata.get("tf_model_path")
        if tf_model_path and os.path.exists(tf_model_path):
            model_instance.model = tf.keras.models.load_model(tf_model_path)
            logger.info(f"Loaded TensorFlow model from {tf_model_path}")
        else:
            logger.warning(f"TensorFlow model path not found or invalid: {tf_model_path}")
        
        return model_instance

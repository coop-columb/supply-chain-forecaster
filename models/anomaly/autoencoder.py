"""Autoencoder anomaly detection for the supply chain forecaster."""

import datetime
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from models.base import ModelBase, ModelRegistry
from utils import get_logger, safe_execute

logger = get_logger(__name__)


@ModelRegistry.register
class AutoencoderDetector(ModelBase):
    """Autoencoder-based anomaly detection model."""

    def __init__(
        self,
        name: str = "AutoencoderDetector",
        encoding_dim: int = 8,
        hidden_dims: List[int] = [32, 16],
        activation: str = "relu",
        optimizer: str = "adam",
        loss: str = "mse",
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10,
        contamination: float = 0.01,
        **kwargs,
    ):
        """
        Initialize the autoencoder anomaly detector.

        Args:
            name: Name of the model.
            encoding_dim: Dimension of the encoding layer.
            hidden_dims: Dimensions of hidden layers.
            activation: Activation function.
            optimizer: Optimizer.
            loss: Loss function.
            epochs: Number of epochs to train.
            batch_size: Batch size for training.
            validation_split: Fraction of data to use for validation.
            early_stopping: Whether to use early stopping.
            patience: Number of epochs with no improvement for early stopping.
            contamination: Expected proportion of anomalies.
            **kwargs: Additional parameters.
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
            encoding_dim=encoding_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            optimizer=optimizer,
            loss=loss,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping=early_stopping,
            patience=patience,
            contamination=contamination,
            **kwargs,
        )

        # Set random seed for reproducibility
        tf.random.set_seed(42)

        # Store preprocessing parameters
        self.scaler = None
        self.threshold = None

    def _build_model(self, input_dim: int) -> Tuple:
        """
        Build the autoencoder model.

        Args:
            input_dim: Dimension of the input data.

        Returns:
            Tuple of (encoder model, autoencoder model).
        """
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Dropout, Input
        from tensorflow.keras.models import Model

        # Input layer
        input_layer = Input(shape=(input_dim,))

        # Encoder layers
        x = input_layer

        for hidden_dim in self.params["hidden_dims"]:
            x = Dense(hidden_dim, activation=self.params["activation"])(x)

        # Bottleneck layer
        encoded = Dense(
            self.params["encoding_dim"], activation=self.params["activation"]
        )(x)

        # Decoder layers
        x = encoded

        for hidden_dim in reversed(self.params["hidden_dims"]):
            x = Dense(hidden_dim, activation=self.params["activation"])(x)

        # Output layer
        decoded = Dense(input_dim, activation="linear")(x)

        # Define encoder and autoencoder models
        encoder = Model(input_layer, encoded, name="encoder")
        autoencoder = Model(input_layer, decoded, name="autoencoder")

        # Compile autoencoder
        autoencoder.compile(
            optimizer=self.params["optimizer"],
            loss=self.params["loss"],
        )

        return encoder, autoencoder

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        scale_data: bool = True,
        **kwargs,
    ) -> "AutoencoderDetector":
        """
        Fit the autoencoder model to the data.

        Args:
            X: Feature dataframe.
            y: Ignored (kept for API consistency).
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

        logger.info(f"Fitting autoencoder model {self.name}")

        # Store feature names
        self.features = list(X.columns)

        # Scale data if requested
        X_scaled = X.copy()

        if scale_data:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index,
            )

        # Build model
        input_dim = X.shape[1]
        encoder, autoencoder = self._build_model(input_dim)

        # Prepare callbacks
        callbacks = []

        if self.params["early_stopping"]:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.params["patience"],
                restore_best_weights=True,
            )
            callbacks.append(early_stopping)

        # Train model
        history = autoencoder.fit(
            X_scaled,
            X_scaled,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=self.params["validation_split"],
            callbacks=callbacks,
            verbose=1,
            **kwargs,
        )

        # Store models
        self.encoder = encoder
        self.model = autoencoder

        # Compute reconstruction errors
        reconstructions = autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled.values - reconstructions), axis=1)

        # Set threshold for anomaly detection
        self.threshold = np.percentile(mse, 100 * (1 - self.params["contamination"]))

        # Update metadata
        self.metadata.update(
            {
                "fitted_at": datetime.datetime.now().isoformat(),
                "data_shape": X.shape,
                "input_dim": input_dim,
                "training_loss": float(history.history["loss"][-1]),
                "validation_loss": (
                    float(history.history["val_loss"][-1])
                    if "val_loss" in history.history
                    else None
                ),
                "epochs_trained": len(history.history["loss"]),
                "threshold": float(self.threshold),
                "mean_reconstruction_error": float(np.mean(mse)),
            }
        )

        logger.info(f"Successfully fitted autoencoder model {self.name}")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
        return_scores: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict anomalies in the data.

        Args:
            X: Feature dataframe.
            threshold: Threshold for anomaly detection (overrides fitted threshold).
            return_scores: Whether to return anomaly scores.
            **kwargs: Additional prediction parameters.

        Returns:
            Array of anomaly predictions (1 for normal, -1 for anomaly) and optionally scores.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")

        logger.info(f"Predicting anomalies with autoencoder model {self.name}")

        # Use provided threshold or fitted threshold
        threshold = threshold or self.threshold

        # Scale data if scaler exists
        X_scaled = X.copy()

        if hasattr(self, "scaler") and self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index,
            )

        # Get reconstructions and calculate errors
        reconstructions = self.model.predict(X_scaled)
        mse = np.mean(np.square(X_scaled.values - reconstructions), axis=1)

        # Determine anomalies
        predictions = np.where(mse > threshold, -1, 1)

        if return_scores:
            return predictions, mse
        else:
            return predictions

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        """
        Encode data using the autoencoder's encoder.

        Args:
            X: Feature dataframe.

        Returns:
            Encoded data.
        """
        if not hasattr(self, "encoder") or self.encoder is None:
            raise ValueError("Encoder not available")

        # Scale data if scaler exists
        X_scaled = X.copy()

        if hasattr(self, "scaler") and self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index,
            )

        return self.encoder.predict(X_scaled)

    def get_anomalies(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
        include_scores: bool = True,
        include_reconstructions: bool = False,
    ) -> pd.DataFrame:
        """
        Get anomalies with their scores.

        Args:
            X: Feature dataframe.
            threshold: Threshold for anomaly detection.
            include_scores: Whether to include anomaly scores in the result.
            include_reconstructions: Whether to include reconstructed values.

        Returns:
            Dataframe with anomalies and their scores.
        """
        predictions, scores = self.predict(X, threshold=threshold, return_scores=True)

        # Create result dataframe
        result = X.copy()
        result["is_anomaly"] = predictions == -1

        if include_scores:
            result["anomaly_score"] = scores

        if include_reconstructions:
            # Scale data if scaler exists
            X_scaled = X.copy()

            if hasattr(self, "scaler") and self.scaler is not None:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns,
                    index=X.index,
                )

            # Get reconstructions
            reconstructions = self.model.predict(X_scaled)

            # Inverse transform if scaler exists
            if hasattr(self, "scaler") and self.scaler is not None:
                reconstructions = self.scaler.inverse_transform(reconstructions)

            # Add reconstructed values to result
            for i, col in enumerate(X.columns):
                result[f"{col}_reconstructed"] = reconstructions[:, i]

        # Filter to only anomalies
        anomalies = result[result["is_anomaly"]].copy()

        logger.info(f"Found {len(anomalies)} anomalies out of {len(X)} points")

        return anomalies

    def save(self, path: Optional[Union[str, os.PathLike]] = None) -> os.PathLike:
        """
        Save the autoencoder model to disk.

        Args:
            path: Path to save the model. If None, uses the default directory.

        Returns:
            Path to the saved model.
        """
        import joblib
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

        # Save TensorFlow models separately
        if hasattr(self, "model") and self.model is not None:
            tf_model_path = f"{path}_autoencoder"
            self.model.save(tf_model_path)
            logger.info(f"Saved autoencoder model to {tf_model_path}")

            # Update metadata with autoencoder model path
            self.metadata["autoencoder_model_path"] = tf_model_path

        if hasattr(self, "encoder") and self.encoder is not None:
            tf_encoder_path = f"{path}_encoder"
            self.encoder.save(tf_encoder_path)
            logger.info(f"Saved encoder model to {tf_encoder_path}")

            # Update metadata with encoder model path
            self.metadata["encoder_model_path"] = tf_encoder_path

        # Save scaler if exists
        if hasattr(self, "scaler") and self.scaler is not None:
            scaler_path = f"{path}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")

            # Update metadata with scaler path
            self.metadata["scaler_path"] = scaler_path

        # Save threshold
        if hasattr(self, "threshold") and self.threshold is not None:
            self.metadata["threshold"] = float(self.threshold)

        # Save everything else using joblib
        return super().save(path)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "AutoencoderDetector":
        """
        Load an autoencoder model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded model instance.
        """
        import joblib
        import tensorflow as tf

        # Load metadata and other components
        model_instance = super().load(path)

        # Load TensorFlow autoencoder if available
        autoencoder_path = model_instance.metadata.get("autoencoder_model_path")
        if autoencoder_path and os.path.exists(autoencoder_path):
            model_instance.model = tf.keras.models.load_model(autoencoder_path)
            logger.info(f"Loaded autoencoder model from {autoencoder_path}")
        else:
            logger.warning(
                f"Autoencoder model path not found or invalid: {autoencoder_path}"
            )

        # Load TensorFlow encoder if available
        encoder_path = model_instance.metadata.get("encoder_model_path")
        if encoder_path and os.path.exists(encoder_path):
            model_instance.encoder = tf.keras.models.load_model(encoder_path)
            logger.info(f"Loaded encoder model from {encoder_path}")
        else:
            logger.warning(f"Encoder model path not found or invalid: {encoder_path}")

        # Load scaler if available
        scaler_path = model_instance.metadata.get("scaler_path")
        if scaler_path and os.path.exists(scaler_path):
            model_instance.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            logger.warning(f"Scaler path not found or invalid: {scaler_path}")

        # Load threshold
        model_instance.threshold = model_instance.metadata.get("threshold")

        return model_instance

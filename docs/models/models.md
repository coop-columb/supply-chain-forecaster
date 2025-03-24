# Model Documentation

This document provides detailed information about the forecasting and anomaly detection models available in the Supply Chain Forecaster system.

## Forecasting Models

The system supports several forecasting models, each with their own strengths and suitable use cases.

### Prophet Model

**Class**: `ProphetModel`

Prophet is a forecasting procedure implemented by Facebook designed to handle time series data with strong seasonal effects and holidays.

#### Key Features

- Handles missing data and outliers
- Automatically detects changepoints in the trend
- Supports holiday effects
- Produces uncertainty intervals

#### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| seasonality_mode | Type of seasonality | "additive" | "additive", "multiplicative" |
| changepoint_prior_scale | Flexibility of the trend | 0.05 | 0.001 to 0.5 |
| seasonality_prior_scale | Flexibility of the seasonality | 10.0 | 0.01 to 100.0 |
| holidays_prior_scale | Flexibility of the holidays | 10.0 | 0.01 to 100.0 |
| daily_seasonality | Whether to include daily seasonality | False | True, False |
| weekly_seasonality | Whether to include weekly seasonality | True | True, False |
| yearly_seasonality | Whether to include yearly seasonality | True | True, False |
| add_country_holidays | Country code to include holidays for | None | "US", "UK", etc. |

#### Best For

- Medium to long-term forecasting of time series with strong seasonal patterns
- Data with multiple seasonalities (e.g., daily, weekly, yearly)
- Data with irregular patterns due to holidays and events
- Forecasting with additional regressors

#### Example Usage

```python
model = ProphetModel(
    name="demand_forecast",
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05,
    yearly_seasonality=True,
    weekly_seasonality=True,
    add_country_holidays="US"
)

# Fit the model
model.fit(X_train, y_train, date_col="date")

# Generate forecast
forecast = model.predict(X_test, steps=30, return_conf_int=True)
```

### ARIMA Model

**Class**: `ARIMAModel`

ARIMA (AutoRegressive Integrated Moving Average) is a classical time series forecasting model that combines autoregression, differencing, and moving averages.

#### Key Features

- Captures linear relationships in time series data
- Supports seasonal patterns with SARIMA extension
- Provides statistical properties of the forecast
- Can perform automatic parameter selection with auto_arima

#### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| order | ARIMA order (p, d, q) | (1, 1, 1) | p: 0 to 5, d: 0 to 2, q: 0 to 5 |
| seasonal_order | Seasonal ARIMA order (P, D, Q, s) | None | P: 0 to 2, D: 0 to 1, Q: 0 to 2, s: period |
| trend | Trend component | None | "n", "c", "t", "ct" |
| enforce_stationarity | Whether to enforce stationarity | True | True, False |
| enforce_invertibility | Whether to enforce invertibility | True | True, False |
| auto_arima | Whether to use auto_arima | False | True, False |

#### Best For

- Short to medium-term forecasting of stationary time series
- Time series with clear patterns that can be captured by linear models
- Situations where interpretability and statistical properties are important
- Data with regular seasonal patterns

#### Example Usage

```python
model = ARIMAModel(
    name="inventory_forecast",
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 12),
    trend="c"
)

# Fit the model
model.fit(X_train, y_train, date_col="date")

# Generate forecast
forecast = model.predict(X_test, steps=30)
```

### XGBoost Model

**Class**: `XGBoostModel`

XGBoost is a powerful gradient boosting algorithm that can be adapted for time series forecasting by using lagged features.

#### Key Features

- Handles non-linear relationships
- Captures complex feature interactions
- Provides feature importance
- Robust to outliers and missing values

#### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| n_estimators | Number of boosting stages | 100 | 10 to 1000 |
| max_depth | Maximum depth of trees | 6 | 1 to 15 |
| learning_rate | Learning rate | 0.1 | 0.001 to 0.5 |
| subsample | Subsample ratio | 1.0 | 0.1 to 1.0 |
| colsample_bytree | Column subsample ratio | 1.0 | 0.1 to 1.0 |
| objective | Objective function | "reg:squarederror" | "reg:squarederror", "reg:absoluteerror" |
| booster | Booster type | "gbtree" | "gbtree", "gblinear", "dart" |
| tree_method | Tree construction algorithm | "auto" | "auto", "exact", "approx", "hist" |

#### Best For

- Complex time series with multiple input features
- Scenarios where feature importance is needed
- Forecasting problems where non-linear relationships are important
- Data with a moderate to large number of features

#### Example Usage

```python
model = XGBoostModel(
    name="sales_forecast",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05
)

# Fit the model
model.fit(X_train, y_train)

# Generate forecast
forecast = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance()
```

### LSTM Model

**Class**: `LSTMModel`

Long Short-Term Memory (LSTM) is a type of recurrent neural network capable of learning long-term dependencies in time series data.

#### Key Features

- Captures complex temporal dependencies
- Handles variable-length sequences
- Can model multiple time scales
- Powerful for complex time series with non-linear patterns

#### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| units | List of LSTM layer sizes | [64, 32] | Positive integers |
| dropout | Dropout rate | 0.2 | 0.0 to 0.5 |
| recurrent_dropout | Recurrent dropout rate | 0.0 | 0.0 to 0.5 |
| activation | Activation function | "relu" | "relu", "tanh", "sigmoid" |
| optimizer | Optimizer | "adam" | "adam", "sgd", "rmsprop" |
| loss | Loss function | "mse" | "mse", "mae" |
| batch_size | Batch size | 32 | 8 to 256 |
| epochs | Number of epochs | 50 | 10 to 500 |
| sequence_length | Sequence length for LSTM input | 10 | 1 to 100 |

#### Best For

- Complex time series with long-term dependencies
- Data with non-linear patterns that simpler models cannot capture
- Situations where high forecast accuracy is more important than interpretability
- Time series with multiple input features and complex interactions

#### Example Usage

```python
model = LSTMModel(
    name="complex_forecast",
    units=[128, 64],
    dropout=0.3,
    sequence_length=14,
    epochs=100
)

# Fit the model
model.fit(X_train, y_train, scale_data=True)

# Generate forecast
forecast = model.predict(X_test, steps_ahead=30)
```

## Anomaly Detection Models

The system supports several anomaly detection models, each with their own strengths and suitable use cases.

### Isolation Forest Detector

**Class**: `IsolationForestDetector`

Isolation Forest is an algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

#### Key Features

- Effective for high-dimensional data
- Fast training and prediction
- Works well with limited samples
- Based on the principle that anomalies are easier to isolate

#### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| n_estimators | Number of isolation trees | 100 | 10 to 1000 |
| max_samples | Number of samples to draw | "auto" | "auto" or a positive value |
| contamination | Expected proportion of anomalies | "auto" | "auto" or 0.001 to 0.5 |
| max_features | Number of features to consider | 1.0 | 0.1 to 1.0 |
| bootstrap | Whether to bootstrap samples | False | True, False |
| n_jobs | Number of parallel jobs | -1 | -1 or positive integer |
| random_state | Random seed | 42 | Any integer |

#### Best For

- High-dimensional data where distance-based methods struggle
- Datasets with a mix of categorical and numerical features
- Efficient detection of global anomalies
- Scenarios where training time is a concern

#### Example Usage

```python
model = IsolationForestDetector(
    name="inventory_anomaly",
    n_estimators=100,
    contamination=0.05
)

# Fit the model
model.fit(X_train)

# Detect anomalies
predictions, scores = model.predict(X_test, return_scores=True)

# Get anomalies with details
anomalies = model.get_anomalies(X_test)
```

### Statistical Detector

**Class**: `StatisticalDetector`

The Statistical Detector uses statistical methods such as Z-score, IQR, or MAD to identify outliers in the data.

#### Key Features

- Simple and interpretable
- Fast computation
- Effective for univariate anomaly detection
- Different statistical methods available

#### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| method | Detection method | "zscore" | "zscore", "iqr", "mad" |
| threshold | Threshold for anomaly detection | 3.0 | 1.0 to 10.0 |
| target_column | Target column for univariate detection | None | Any column name |

#### Best For

- Simple univariate or multivariate anomaly detection
- Cases where interpretability is important
- Preliminary analysis before applying more complex methods
- Detecting outliers in individual metrics

#### Example Usage

```python
model = StatisticalDetector(
    name="demand_anomaly",
    method="iqr",
    threshold=2.5,
    target_column="demand"
)

# Fit the model
model.fit(X_train)

# Detect anomalies
predictions, scores = model.predict(X_test, return_scores=True)

# Get anomalies with details
anomalies = model.get_anomalies(X_test)
```

### Autoencoder Detector

**Class**: `AutoencoderDetector`

The Autoencoder Detector uses a neural network to learn a compressed representation of the normal data and identifies points that do not reconstruct well as anomalies.

#### Key Features

- Captures complex patterns in the data
- Works well with high-dimensional data
- Can detect subtle anomalies
- Learns a compact representation of normal data

#### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| encoding_dim | Dimension of the encoding layer | 8 | 2 to 64 |
| hidden_dims | Dimensions of hidden layers | [32, 16] | Lists of positive integers |
| activation | Activation function | "relu" | "relu", "sigmoid", "tanh" |
| optimizer | Optimizer | "adam" | "adam", "sgd", "rmsprop" |
| loss | Loss function | "mse" | "mse", "mae" |
| epochs | Number of epochs | 50 | 10 to 500 |
| batch_size | Batch size | 32 | 8 to 256 |
| contamination | Expected proportion of anomalies | 0.01 | 0.001 to 0.5 |

#### Best For

- Complex data with non-linear patterns
- High-dimensional data with intricate relationships
- Detecting subtle anomalies that statistical methods might miss
- Scenarios where a deep learning approach is warranted

#### Example Usage

```python
model = AutoencoderDetector(
    name="complex_anomaly",
    encoding_dim=8,
    hidden_dims=[64, 32],
    contamination=0.02,
    epochs=100
)

# Fit the model
model.fit(X_train, scale_data=True)

# Detect anomalies
predictions, scores = model.predict(X_test, return_scores=True)

# Get anomalies with details
anomalies = model.get_anomalies(X_test, include_reconstructions=True)

# Get encoded representation
encoded_data = model.encode(X_test)
```

## Model Selection Guidelines

Here are some guidelines for selecting the appropriate forecasting and anomaly detection models:

### Forecasting Model Selection

- **Prophet**: Choose when dealing with time series that have strong seasonality, holidays, and changepoints. Good for medium to long-term forecasting.
- **ARIMA**: Choose for stationary or easily differenced time series with linear relationships. Good for short to medium-term forecasting.
- **XGBoost**: Choose when there are many input features and complex relationships. Good for scenarios where feature importance is needed.
- **LSTM**: Choose for complex, non-linear time series with long-term dependencies. Good when high accuracy is the priority and interpretability is less important.

### Anomaly Detection Model Selection

- **Isolation Forest**: Choose for high-dimensional data or when training speed is important. Good for identifying global anomalies.
- **Statistical Detector**: Choose for simple, interpretable anomaly detection on individual metrics. Good for preliminary analysis.
- **Autoencoder Detector**: Choose for complex data with subtle anomalies. Good when a deep learning approach is warranted.

## Cross-Validation Strategies

The system supports different cross-validation strategies for time series data:

### Expanding Window

The expanding window strategy uses an increasing amount of data for training as it moves forward in time. It's useful for stable time series where more data generally leads to better forecasts.

Parameters:
- initial_window: The size of the first training window
- step_size: The number of steps to move forward for each fold
- horizon: The forecasting horizon

### Sliding Window

The sliding window strategy uses a fixed window size for training, sliding forward in time. It's useful for time series with changing patterns where recent data is more relevant.

Parameters:
- initial_window: The size of the training window
- step_size: The number of steps to move forward for each fold
- horizon: The forecasting horizon
- max_train_size: Maximum number of samples used for training

## Evaluation Metrics

The system calculates several metrics to evaluate forecasting and anomaly detection models:

### Forecasting Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Square root of the average squared difference between predictions and actual values
- **MAPE (Mean Absolute Percentage Error)**: Average percentage difference between predictions and actual values
- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Symmetric version of MAPE that handles zero values better
- **R² (R-squared)**: Proportion of the variance in the dependent variable that is predictable from the independent variable(s)

### Anomaly Detection Metrics

- **Precision**: Proportion of predicted anomalies that are actual anomalies
- **Recall**: Proportion of actual anomalies that are correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve, measuring the trade-off between true positive rate and false positive rate
- **Anomaly Rate**: Proportion of data points identified as anomalies

## Next Steps

After understanding the available models, explore the [Usage Guide](../usage/usage.md) for practical examples of how to use them in the system.
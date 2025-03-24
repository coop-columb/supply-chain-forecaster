# API Documentation

This document provides detailed information about the Supply Chain Forecaster API endpoints, request/response formats, and example usage.

## Base URL

The API is accessible at:

```
http://localhost:8000
```

## API Overview

The API is organized into the following sections:

- **Health Check**: Endpoints for monitoring API health
- **Authentication**: Endpoints for API key management and user info
- **Models**: Endpoints for managing trained models
- **Forecasting**: Endpoints for training forecasting models and generating predictions
- **Predictions**: Endpoints for making predictions with trained models
- **Anomalies**: Endpoints for anomaly detection

## Swagger Documentation

The API provides interactive Swagger documentation at:

```
http://localhost:8000/docs
```

This interface allows you to explore the API and test endpoints directly.

## Health Check Endpoints

### Get Health Status

```
GET /health
```

Returns the current health status of the API.

Response:
```json
{
  "status": "ok"
}
```

### Get Readiness Status

```
GET /health/readiness
```

Returns the readiness status of the API.

Response:
```json
{
  "status": "ready"
}
```

### Get Liveness Status

```
GET /health/liveness
```

Returns the liveness status of the API.

Response:
```json
{
  "status": "alive"
}
```

### Get API Version

```
GET /version
```

Returns the current API version.

Response:
```json
{
  "version": "0.1.0"
}
```

## Model Management Endpoints

### List Available Models

```
GET /models/
```

Returns a list of available model types.

Response:
```json
{
  "available_models": [
    "ProphetModel",
    "ARIMAModel",
    "XGBoostModel",
    "LSTMModel",
    "IsolationForestDetector",
    "StatisticalDetector",
    "AutoencoderDetector"
  ]
}
```

### List Trained Models

```
GET /models/?trained=true
```

Returns a list of trained models.

Response:
```json
{
  "trained_models": {
    "ProphetModel_20230815_123456": {
      "model_type": "ProphetModel",
      "created_at": "2023-08-15T12:34:56",
      "parameters": {
        "seasonality_mode": "additive",
        "changepoint_prior_scale": 0.05
      }
    }
  }
}
```

### List Deployed Models

```
GET /models/?deployed=true
```

Returns a list of deployed models.

Response:
```json
{
  "deployed_models": {
    "ProphetModel_20230815_123456": {
      "model_type": "ProphetModel",
      "created_at": "2023-08-15T12:34:56",
      "parameters": {
        "seasonality_mode": "additive",
        "changepoint_prior_scale": 0.05
      }
    }
  }
}
```

### Get Model Details

```
GET /models/{model_name}?model_type={model_type}&from_deployment={true|false}
```

Returns details about a specific model.

**Parameters:**
- `model_name`: Name of the model
- `model_type`: Type of the model
- `from_deployment`: Whether to load from deployment (default: true)

**Response:**

```json
{
  "name": "ProphetModel_20230815_123456",
  "type": "ProphetModel",
  "metadata": {
    "created_at": "2023-08-15T12:34:56",
    "fitted_at": "2023-08-15T12:40:23",
    "parameters": {
      "seasonality_mode": "additive",
      "changepoint_prior_scale": 0.05
    },
    "train_metrics": {
      "mae": 10.25,
      "rmse": 15.72,
      "mape": 8.45
    }
  }
}
```

### Deploy a Model

```
POST /models/{model_name}/deploy
```

Deploys a trained model for production use.

**Parameters:**
- `model_name`: Name of the model to deploy

**Response:**
```json
{
  "status": "success",
  "message": "Model 'ProphetModel_20230815_123456' deployed successfully",
  "metadata": {
    "created_at": "2023-08-15T12:34:56",
    "deployed_at": "2023-08-15T13:45:30"
  }
}
```

### Delete a Model

```
DELETE /models/{model_name}?from_deployment={true|false}
```

Deletes a model.

**Parameters:**
- `model_name`: Name of the model to delete
- `from_deployment`: Whether to delete from deployment (default: false)

**Response:**
```json
{
  "status": "success",
  "message": "Model 'ProphetModel_20230815_123456' deleted successfully"
}
```

## Authentication Endpoints

These endpoints are available when authentication is enabled in the configuration.

### Authentication Methods

The API supports two authentication methods:

1. **HTTP Basic Authentication**: Username and password authentication
2. **API Key Authentication**: Using API keys in the `X-API-Key` header

### Get Current User Info

```
GET /auth/me
```

Returns information about the currently authenticated user.

**Headers:**
- `Authorization`: Basic authentication header
- Or `X-API-Key`: API key

**Response:**
```json
{
  "username": "admin",
  "full_name": "Administrator",
  "email": "admin@example.com",
  "auth_type": "basic",
  "roles": ["admin"]
}
```

### Create a New API Key

```
POST /auth/keys
```

Creates a new API key. Requires administrator privileges.

**Headers:**
- `Authorization`: Basic authentication header

**Request:**
```json
{
  "name": "My Service API Key",
  "expires_days": 90,
  "scope": "read:forecasts write:forecasts"
}
```

**Response:**
```json
{
  "key": "a1b2c3d4e5f6...",
  "name": "My Service API Key",
  "created_at": 1679012345,
  "expires_at": 1687012345,
  "scope": "read:forecasts write:forecasts"
}
```

### List API Keys

```
GET /auth/keys
```

Lists all API keys without showing the full key values. Requires administrator privileges.

**Headers:**
- `Authorization`: Basic authentication header

**Response:**
```json
[
  {
    "id": "a1b2c3d4...",
    "name": "My Service API Key",
    "created_at": 1679012345,
    "expires_at": 1687012345,
    "last_used": 1680012345,
    "scope": "read:forecasts write:forecasts"
  },
  {
    "id": "e5f6g7h8...",
    "name": "Monitoring Key",
    "created_at": 1678012345,
    "expires_at": null,
    "last_used": 1680011111,
    "scope": "read:health"
  }
]
```

### Revoke an API Key

```
DELETE /auth/keys/{key_id}
```

Revokes (deletes) an API key. Requires administrator privileges.

**Headers:**
- `Authorization`: Basic authentication header

**Parameters:**
- `key_id`: ID (prefix) of the API key to revoke

**Response:**
```json
{
  "status": "success",
  "message": "API key a1b2c3d4... revoked"
}
```

## Forecasting Endpoints

### Train a Forecasting Model

```
POST /forecasting/train
```

Trains a forecasting model on the provided data.

**Request:**
- `file`: CSV file containing the data
- `params`: JSON string with training parameters

Example params:
```json
{
  "model_type": "ProphetModel",
  "model_name": "MyProphetModel",
  "feature_columns": ["date", "temperature", "promotion"],
  "target_column": "demand",
  "date_column": "date",
  "model_params": {
    "seasonality_mode": "additive",
    "changepoint_prior_scale": 0.05
  },
  "save_model": true
}
```

**Response:**
```json
{
  "status": "success",
  "model_name": "MyProphetModel",
  "model_type": "ProphetModel",
  "metrics": {
    "mae": 10.25,
    "rmse": 15.72,
    "mape": 8.45
  },
  "message": "Model 'MyProphetModel' trained successfully"
}
```

### Generate a Forecast

```
POST /forecasting/forecast
```

Generates a forecast using a trained model.

**Request:**
- `file`: CSV file containing the feature data
- `params`: JSON string with forecast parameters

Example params:
```json
{
  "model_name": "MyProphetModel",
  "model_type": "ProphetModel",
  "feature_columns": ["date", "temperature", "promotion"],
  "date_column": "date",
  "steps": 30,
  "return_conf_int": true,
  "from_deployment": true
}
```

**Response:**
```json
{
  "status": "success",
  "model_name": "MyProphetModel",
  "steps": 30,
  "result": {
    "forecast": [105.2, 106.7, 108.1, ...],
    "lower_bound": [95.3, 96.8, 98.2, ...],
    "upper_bound": [115.1, 116.6, 118.0, ...],
    "dates": ["2023-08-16T00:00:00", "2023-08-17T00:00:00", ...]
  }
}
```

### Perform Cross-Validation

```
POST /forecasting/cross-validate
```

Performs time series cross-validation on a model.

**Request:**
- `file`: CSV file containing the data
- `params`: JSON string with cross-validation parameters

Example params:
```json
{
  "model_type": "ProphetModel",
  "feature_columns": ["date", "temperature", "promotion"],
  "target_column": "demand",
  "date_column": "date",
  "model_params": {
    "seasonality_mode": "additive",
    "changepoint_prior_scale": 0.05
  },
  "strategy": "expanding",
  "initial_window": 30,
  "step_size": 7,
  "horizon": 7
}
```

**Response:**
```json
{
  "status": "success",
  "model_type": "ProphetModel",
  "cv_results": {
    "model_type": "ProphetModel",
    "model_params": {
      "seasonality_mode": "additive",
      "changepoint_prior_scale": 0.05
    },
    "cv_strategy": "expanding",
    "initial_window": 30,
    "step_size": 7,
    "horizon": 7,
    "num_folds": 12,
    "avg_metrics": {
      "mae": 12.45,
      "rmse": 18.32,
      "mape": 9.76
    },
    "fold_metrics": {
      "mae": [11.2, 12.5, 13.7, ...],
      "rmse": [16.8, 18.2, 19.9, ...],
      "mape": [8.5, 9.8, 10.9, ...]
    }
  }
}
```

## Prediction Endpoints

### Make Predictions

```
POST /predictions/
```

Makes predictions using a trained model.

**Request:**
- `file`: CSV file containing the feature data
- `params`: JSON string with prediction parameters

Example params:
```json
{
  "model_name": "MyXGBoostModel",
  "model_type": "XGBoostModel",
  "feature_columns": ["temperature", "promotion", "day_of_week"],
  "from_deployment": true
}
```

**Response:**
```json
{
  "status": "success",
  "model_name": "MyXGBoostModel",
  "predictions": [105.2, 106.7, 108.1, ...]
}
```

## Anomaly Detection Endpoints

### Train an Anomaly Detection Model

```
POST /anomalies/train
```

Trains an anomaly detection model on the provided data.

**Request:**
- `file`: CSV file containing the data
- `params`: JSON string with training parameters

Example params:
```json
{
  "model_type": "IsolationForestDetector",
  "model_name": "MyIsolationForest",
  "feature_columns": ["demand", "inventory", "lead_time"],
  "model_params": {
    "n_estimators": 100,
    "contamination": 0.05
  },
  "save_model": true
}
```

**Response:**
```json
{
  "status": "success",
  "model_name": "MyIsolationForest",
  "model_type": "IsolationForestDetector",
  "metrics": {
    "anomaly_rate": 0.05,
    "num_anomalies": 25
  },
  "message": "Anomaly detection model 'MyIsolationForest' trained successfully"
}
```

### Detect Anomalies

```
POST /anomalies/detect
```

Detects anomalies in the provided data.

**Request:**
- `file`: CSV file containing the data
- `params`: JSON string with detection parameters

Example params:
```json
{
  "model_name": "MyIsolationForest",
  "model_type": "IsolationForestDetector",
  "feature_columns": ["demand", "inventory", "lead_time"],
  "threshold": null,
  "from_deployment": true,
  "return_details": true
}
```

**Response:**
```json
{
  "status": "success",
  "model_name": "MyIsolationForest",
  "results": {
    "model_name": "MyIsolationForest",
    "model_type": "IsolationForestDetector",
    "data_points": 500,
    "anomaly_count": 25,
    "anomaly_rate": 0.05,
    "threshold": -0.5,
    "anomalies": [
      {
        "index": 45,
        "demand": 250.5,
        "inventory": 50.2,
        "lead_time": 12,
        "is_anomaly": true,
        "anomaly_score": 0.85
      },
      // More anomalies...
    ]
  }
}
```

## Error Handling

The API returns standard HTTP status codes and JSON error responses.

### Example Error Response

```json
{
  "error": "Model not found",
  "details": {
    "model_name": "NonExistentModel"
  }
}
```

## API Usage Examples

Here are some examples of how to use the API with curl:

### Authentication Examples

#### Basic Authentication

```bash
# Get user information with basic auth
curl -X GET "http://localhost:8000/auth/me" \
  -u "admin:adminpassword"

# Create a new API key
curl -X POST "http://localhost:8000/auth/keys" \
  -u "admin:adminpassword" \
  -H "Content-Type: application/json" \
  -d '{"name":"Production Service Key","expires_days":90,"scope":"read:* write:forecasts"}'
```

#### API Key Authentication

```bash
# Get user information with API key
curl -X GET "http://localhost:8000/auth/me" \
  -H "X-API-Key: your-api-key-here"

# Using API key for forecast generation
curl -X POST "http://localhost:8000/forecasting/forecast" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@future_data.csv" \
  -F "params={\"model_name\":\"DemandForecast\",\"model_type\":\"ProphetModel\",\"feature_columns\":[\"date\",\"temperature\",\"promotion\"],\"date_column\":\"date\",\"steps\":30,\"return_conf_int\":true}"
```

### Training a Prophet Model

```bash
curl -X POST "http://localhost:8000/forecasting/train" \
  -F "file=@data.csv" \
  -F "params={\"model_type\":\"ProphetModel\",\"model_name\":\"DemandForecast\",\"feature_columns\":[\"date\",\"temperature\",\"promotion\"],\"target_column\":\"demand\",\"date_column\":\"date\",\"model_params\":{\"seasonality_mode\":\"additive\"},\"save_model\":true}"
```

### Generating a Forecast

```bash
curl -X POST "http://localhost:8000/forecasting/forecast" \
  -F "file=@future_data.csv" \
  -F "params={\"model_name\":\"DemandForecast\",\"model_type\":\"ProphetModel\",\"feature_columns\":[\"date\",\"temperature\",\"promotion\"],\"date_column\":\"date\",\"steps\":30,\"return_conf_int\":true}"
```

### Detecting Anomalies

```bash
curl -X POST "http://localhost:8000/anomalies/detect" \
  -F "file=@data.csv" \
  -F "params={\"model_name\":\"InventoryAnomaly\",\"model_type\":\"IsolationForestDetector\",\"feature_columns\":[\"demand\",\"inventory\",\"lead_time\"],\"return_details\":true}"
```

## Next Steps

After familiarizing yourself with the API, explore the [Model Documentation](../models/models.md) for more information about the available models and their parameters.
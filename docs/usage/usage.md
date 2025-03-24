# Usage Guide

This guide provides detailed instructions on how to use the Supply Chain Forecaster system.

## Dashboard Usage

The dashboard provides an intuitive interface for working with supply chain data, training models, and generating forecasts.

### Accessing the Dashboard

Open your web browser and navigate to:

```
http://localhost:8050
```

The default port is 8050, but this may be different if you configured it differently.

### Dashboard Features

The dashboard is organized into several sections:

- **Home**: Overview of the system and quick links to key features.
- **Data Exploration**: Upload and analyze your supply chain data.
- **Forecasting**: Train forecasting models and generate predictions.
- **Anomaly Detection**: Train anomaly detection models and identify unusual patterns.
- **Model Management**: View, deploy, and manage trained models.

### Data Exploration

1. Click on the "Data Exploration" link in the navigation bar.
2. Upload a CSV file containing your supply chain data by dragging and dropping or selecting a file.
3. Once uploaded, you'll see a preview of the data and various visualizations.
4. Analyze the time series patterns and correlations in your data.

### Forecasting

#### Training a Forecasting Model

1. Click on the "Forecasting" link in the navigation bar.
2. Select the "Train Model" tab.
3. Upload a CSV file containing your historical data.
4. Configure the model:
   - Select a model type (Prophet, ARIMA, XGBoost, LSTM)
   - Enter a model name
   - Configure model-specific parameters
   - Select feature columns, target column, and date column
5. Click "Train Model" to start the training process.
6. Once training is complete, you'll see the training metrics and results.

#### Generating Forecasts

1. Click on the "Forecasting" link in the navigation bar.
2. Select the "Generate Forecast" tab.
3. Upload a CSV file containing your feature data for the forecast period.
4. Configure the forecast:
   - Select a trained model
   - Select feature columns and date column
   - Set the forecast horizon
5. Click "Generate Forecast" to create the forecast.
6. View the forecast chart showing historical and predicted values.

#### Cross-Validation

1. Click on the "Forecasting" link in the navigation bar.
2. Select the "Cross-Validation" tab.
3. Upload a CSV file containing your historical data.
4. Configure the cross-validation:
   - Select a model type and parameters
   - Select feature columns, target column, and date column
   - Set cross-validation parameters
5. Click "Run Cross-Validation" to start the process.
6. View the cross-validation metrics across different time periods.

### Anomaly Detection

#### Training an Anomaly Detection Model

1. Click on the "Anomaly Detection" link in the navigation bar.
2. Select the "Train Model" tab.
3. Upload a CSV file containing your historical data.
4. Configure the model:
   - Select a model type (Isolation Forest, Statistical, Autoencoder)
   - Enter a model name
   - Configure model-specific parameters
   - Select feature columns
5. Click "Train Model" to start the training process.
6. Once training is complete, you'll see the training metrics and results.

#### Detecting Anomalies

1. Click on the "Anomaly Detection" link in the navigation bar.
2. Select the "Detect Anomalies" tab.
3. Upload a CSV file containing your data for anomaly detection.
4. Configure the detection:
   - Select a trained model
   - Select feature columns and date column (for visualization)
   - Set the threshold (optional)
5. Click "Detect Anomalies" to identify outliers.
6. View the anomaly chart highlighting unusual data points.

### Model Management

#### Viewing Models

1. Click on the "Model Management" link in the navigation bar.
2. Select either the "Trained Models" or "Deployed Models" tab.
3. Click "Refresh Model List" to see the latest models.
4. Click on a model in the list to view its details.

#### Deploying Models

1. In the "Model Management" page, select a trained model.
2. Click "Deploy Model" in the model details section.
3. The model will be deployed for production use.

#### Deleting Models

1. In the "Model Management" page, select a model.
2. Click "Delete Model" in the model details section to remove it.

## API Usage

The API provides programmatic access to all system features.

### Accessing the API

The API is available at:

```
http://localhost:8000
```

The default port is 8000, but this may be different if you configured it differently.

### API Documentation

The API provides detailed Swagger documentation at:

```
http://localhost:8000/docs
```

This interactive documentation lets you explore and test all API endpoints.

### API Authentication

The API currently does not have authentication enabled. For production use, it's recommended to implement an authentication mechanism.

### API Examples

See the [API Documentation](../api/api.md) for detailed examples of API calls.

## Next Steps

After familiarizing yourself with the system, explore the [API Documentation](../api/api.md) and [Model Documentation](../models/models.md) for more advanced usage scenarios.
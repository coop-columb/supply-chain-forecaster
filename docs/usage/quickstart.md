# Quick Start Guide

This guide provides a fast path to get up and running with the Supply Chain Forecaster. Follow these steps to install, configure, and start using the system for basic forecasting and anomaly detection.

## Prerequisites

- Docker and Docker Compose installed
- Git installed
- Python 3.10+ (if running locally without Docker)

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/supply-chain-forecaster.git
   cd supply-chain-forecaster
   ```

2. Create a configuration file:
   ```bash
   cp .env.example .env
   ```

3. Start the development environment:
   ```bash
   docker-compose up -d api dashboard
   ```

4. Access the Dashboard:
   - Open your browser and navigate to [http://localhost:8050](http://localhost:8050)
   - The API is available at [http://localhost:8000](http://localhost:8000)

### Local Installation

For detailed local installation instructions, see the [Installation Guide](../installation/installation.md).

## Sample Workflow: Demand Forecasting

Here's a quick example of how to generate a forecast using the Dashboard:

1. **Prepare Your Data**

   Create a CSV file with the following columns:
   - `date`: Date in YYYY-MM-DD format
   - `demand`: The target value to forecast
   - Any additional features (e.g., `price`, `promotions`, etc.)

   Example `sample_data.csv`:
   ```
   date,demand,price,promotion
   2023-01-01,100,15.99,0
   2023-01-02,120,15.99,0
   2023-01-03,80,14.99,1
   ...
   ```

2. **Upload Data and Train a Model**

   1. Navigate to the **Forecasting** page
   2. Select the **Train Model** tab
   3. Upload your CSV file
   4. Configure the model:
      - Model Type: `Prophet`
      - Model Name: `DemandForecast`
      - Feature Columns: Select your feature columns
      - Target Column: `demand`
      - Date Column: `date`
   5. Click **Train Model**

3. **Generate a Forecast**

   1. Once the model is trained, go to the **Generate Forecast** tab
   2. Select your trained model `DemandForecast`
   3. Set the forecast horizon (e.g., 30 days)
   4. Click **Generate Forecast**
   5. View the forecast chart showing predictions with confidence intervals

## Sample Workflow: Anomaly Detection

1. **Prepare Your Data**

   Create a CSV file with numerical columns representing your supply chain metrics.

2. **Train an Anomaly Detection Model**

   1. Navigate to the **Anomaly Detection** page
   2. Select the **Train Model** tab
   3. Upload your CSV file
   4. Configure the model:
      - Model Type: `IsolationForest`
      - Model Name: `SupplyChainAnomalies`
      - Feature Columns: Select all relevant metrics
   5. Click **Train Model**

3. **Detect Anomalies**

   1. Go to the **Detect Anomalies** tab
   2. Select your trained model `SupplyChainAnomalies`
   3. Upload data for analysis
   4. Click **Detect Anomalies**
   5. View the results showing detected anomalies

## Next Steps

After completing this quick start guide, explore these resources:

1. [Dashboard Walkthrough](dashboard_walkthrough.md) - Detailed guide to the dashboard
2. [Common Workflows](common_workflows.md) - Step-by-step guides for common tasks
3. [API Examples](api_examples.md) - Examples of using the API programmatically

For comprehensive documentation, see:
- [Full Usage Guide](usage.md)
- [API Documentation](../api/api.md)
- [Model Documentation](../models/models.md)
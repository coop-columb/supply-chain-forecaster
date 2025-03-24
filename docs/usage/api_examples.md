# API Usage Examples

This guide provides practical examples of using the Supply Chain Forecaster API with various programming languages and for different use cases.

## Table of Contents

1. [Python Client Examples](#python-client-examples)
2. [JavaScript Client Examples](#javascript-client-examples)
3. [Command-Line Examples](#command-line-examples)
4. [Advanced Usage Scenarios](#advanced-usage-scenarios)
5. [Integration Examples](#integration-examples)
6. [Error Handling](#error-handling)

## Python Client Examples

### Setting Up

First, install the required packages:

```bash
pip install requests pandas matplotlib
```

### Basic API Client

Create a simple Python client for interacting with the API:

```python
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt

class SupplyChainClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def health_check(self):
        """Check if API is operational"""
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        return response.json()
    
    def list_models(self, trained=False, deployed=False):
        """List available models"""
        params = {"trained": trained, "deployed": deployed}
        response = requests.get(
            f"{self.base_url}/models/", 
            params=params, 
            headers=self.headers
        )
        return response.json()
    
    def train_forecasting_model(self, data_file, model_params):
        """Train a forecasting model"""
        files = {"file": open(data_file, "rb")}
        data = {"params": json.dumps(model_params)}
        
        response = requests.post(
            f"{self.base_url}/forecasting/train",
            files=files,
            data=data,
            headers=self.headers
        )
        return response.json()
    
    def generate_forecast(self, data_file, forecast_params):
        """Generate a forecast using a trained model"""
        files = {"file": open(data_file, "rb")}
        data = {"params": json.dumps(forecast_params)}
        
        response = requests.post(
            f"{self.base_url}/forecasting/forecast",
            files=files,
            data=data,
            headers=self.headers
        )
        return response.json()
    
    def detect_anomalies(self, data_file, detection_params):
        """Detect anomalies in the provided data"""
        files = {"file": open(data_file, "rb")}
        data = {"params": json.dumps(detection_params)}
        
        response = requests.post(
            f"{self.base_url}/anomalies/detect",
            files=files,
            data=data,
            headers=self.headers
        )
        return response.json()
```

### Example: Training a Forecasting Model

```python
client = SupplyChainClient()

# Define model parameters
model_params = {
    "model_type": "ProphetModel",
    "model_name": "DemandForecast_2023Q1",
    "feature_columns": ["date", "temperature", "promotion"],
    "target_column": "demand",
    "date_column": "date",
    "model_params": {
        "seasonality_mode": "additive",
        "changepoint_prior_scale": 0.05
    },
    "save_model": True
}

# Train the model
result = client.train_forecasting_model("data/processed/demand_data.csv", model_params)
print(f"Model training result: {result}")
print(f"Training metrics: {result.get('metrics', {})}")
```

### Example: Generating a Forecast

```python
client = SupplyChainClient()

# Define forecast parameters
forecast_params = {
    "model_name": "DemandForecast_2023Q1",
    "model_type": "ProphetModel",
    "feature_columns": ["date", "temperature", "promotion"],
    "date_column": "date",
    "steps": 30,
    "return_conf_int": True,
    "from_deployment": True
}

# Generate forecast
forecast = client.generate_forecast(
    "data/processed/future_features.csv", 
    forecast_params
)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(forecast["result"]["dates"], forecast["result"]["forecast"], label="Forecast")
plt.fill_between(
    forecast["result"]["dates"],
    forecast["result"]["lower_bound"],
    forecast["result"]["upper_bound"],
    alpha=0.3,
    label="95% Confidence Interval"
)
plt.title("Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("demand_forecast.png")
plt.show()
```

### Example: Detecting Anomalies

```python
client = SupplyChainClient()

# Define anomaly detection parameters
detection_params = {
    "model_name": "InventoryAnomaly_IsoForest",
    "model_type": "IsolationForestDetector",
    "feature_columns": ["demand", "inventory", "lead_time"],
    "threshold": 0.05,
    "from_deployment": True,
    "return_details": True
}

# Detect anomalies
anomalies = client.detect_anomalies(
    "data/processed/inventory_data.csv", 
    detection_params
)

# Print summary
print(f"Data points analyzed: {anomalies['results']['data_points']}")
print(f"Anomalies detected: {anomalies['results']['anomaly_count']}")
print(f"Anomaly rate: {anomalies['results']['anomaly_rate']:.2%}")

# Plot anomalies if date information is available
if "anomalies" in anomalies["results"]:
    # Convert to DataFrame for easier handling
    anomaly_df = pd.DataFrame(anomalies["results"]["anomalies"])
    if "date" in anomaly_df.columns:
        plt.figure(figsize=(12, 6))
        plt.scatter(
            anomaly_df["date"], 
            anomaly_df["demand"],
            c=anomaly_df["is_anomaly"].map({True: "red", False: "blue"}),
            label="Data Points"
        )
        plt.title("Demand Anomalies")
        plt.xlabel("Date")
        plt.ylabel("Demand")
        plt.legend(["Normal", "Anomaly"])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("demand_anomalies.png")
        plt.show()
```

### Example: Batch Processing

```python
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import os

class BatchProcessingClient(SupplyChainClient):
    def process_batch(self, data_files, forecast_params, output_dir):
        """Process multiple files in parallel"""
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    self.generate_forecast, 
                    data_file, 
                    forecast_params
                ): os.path.basename(data_file) 
                for data_file in data_files
            }
            
            for future in futures:
                file_name = futures[future]
                try:
                    result = future.result()
                    results.append({
                        "file": file_name,
                        "status": result["status"],
                        "forecast": result["result"]["forecast"] if "result" in result else None
                    })
                    
                    # Save individual results
                    output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_forecast.csv")
                    if "result" in result and "forecast" in result["result"]:
                        forecast_df = pd.DataFrame({
                            "date": result["result"]["dates"],
                            "forecast": result["result"]["forecast"],
                            "lower_bound": result["result"]["lower_bound"],
                            "upper_bound": result["result"]["upper_bound"]
                        })
                        forecast_df.to_csv(output_file, index=False)
                        
                except Exception as e:
                    results.append({
                        "file": file_name,
                        "status": "error",
                        "error": str(e)
                    })
        
        # Create summary report
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(os.path.join(output_dir, "batch_summary.csv"), index=False)
        
        print(f"Batch processing completed in {time.time() - start_time:.2f} seconds")
        return results
```

### Example: Automated Retraining

```python
import schedule
import time
import datetime

def retrain_models(client, data_path, model_configs):
    """Retrain models with latest data"""
    print(f"Starting model retraining at {datetime.datetime.now()}")
    
    for config in model_configs:
        try:
            # Update model name with timestamp
            config["model_name"] = f"{config['base_name']}_{datetime.datetime.now().strftime('%Y%m%d')}"
            
            # Train model
            result = client.train_forecasting_model(data_path, config)
            
            if result["status"] == "success":
                print(f"Successfully trained model: {config['model_name']}")
                
                # Deploy if specified
                if config.get("auto_deploy", False):
                    deploy_result = client.deploy_model(config["model_name"])
                    print(f"Deployment result: {deploy_result}")
            else:
                print(f"Failed to train model: {config['model_name']}")
                print(f"Error: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"Exception during training of {config.get('base_name', 'unknown')}: {str(e)}")
    
    print(f"Completed model retraining at {datetime.datetime.now()}")

# Example usage
client = SupplyChainClient()
model_configs = [
    {
        "base_name": "WeeklyDemandForecast",
        "model_type": "ProphetModel",
        "feature_columns": ["date", "temperature", "promotion"],
        "target_column": "demand",
        "date_column": "date",
        "model_params": {"seasonality_mode": "additive"},
        "save_model": True,
        "auto_deploy": True
    },
    {
        "base_name": "DailyInventoryForecast",
        "model_type": "XGBoostModel",
        "feature_columns": ["date", "demand", "lead_time"],
        "target_column": "inventory",
        "date_column": "date",
        "model_params": {"n_estimators": 100},
        "save_model": True,
        "auto_deploy": False
    }
]

# Schedule weekly retraining
schedule.every().monday.at("02:00").do(
    retrain_models, 
    client=client,
    data_path="data/processed/latest_data.csv",
    model_configs=model_configs
)

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

## JavaScript Client Examples

### Basic Node.js Client

```javascript
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

class SupplyChainClient {
  constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
    this.baseUrl = baseUrl;
    this.headers = {};
    if (apiKey) {
      this.headers['Authorization'] = `Bearer ${apiKey}`;
    }
  }

  async healthCheck() {
    const response = await axios.get(`${this.baseUrl}/health`, { headers: this.headers });
    return response.data;
  }

  async listModels(trained = false, deployed = false) {
    const response = await axios.get(`${this.baseUrl}/models/`, {
      params: { trained, deployed },
      headers: this.headers
    });
    return response.data;
  }

  async trainForecastingModel(dataFilePath, modelParams) {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(dataFilePath));
    formData.append('params', JSON.stringify(modelParams));

    const response = await axios.post(
      `${this.baseUrl}/forecasting/train`,
      formData,
      {
        headers: {
          ...this.headers,
          ...formData.getHeaders()
        }
      }
    );
    return response.data;
  }

  async generateForecast(dataFilePath, forecastParams) {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(dataFilePath));
    formData.append('params', JSON.stringify(forecastParams));

    const response = await axios.post(
      `${this.baseUrl}/forecasting/forecast`,
      formData,
      {
        headers: {
          ...this.headers,
          ...formData.getHeaders()
        }
      }
    );
    return response.data;
  }
}

// Example usage
async function runExample() {
  const client = new SupplyChainClient();
  
  try {
    // Check API health
    const health = await client.healthCheck();
    console.log('API Health:', health);
    
    // Train a model
    const modelParams = {
      model_type: 'ProphetModel',
      model_name: 'JSDemandForecast',
      feature_columns: ['date', 'temperature', 'promotion'],
      target_column: 'demand',
      date_column: 'date',
      model_params: {
        seasonality_mode: 'additive',
        changepoint_prior_scale: 0.05
      },
      save_model: true
    };
    
    const trainingResult = await client.trainForecastingModel(
      'data/processed/demand_data.csv',
      modelParams
    );
    console.log('Training Result:', trainingResult);
    
    // Generate a forecast
    const forecastParams = {
      model_name: 'JSDemandForecast',
      model_type: 'ProphetModel',
      feature_columns: ['date', 'temperature', 'promotion'],
      date_column: 'date',
      steps: 30,
      return_conf_int: true
    };
    
    const forecastResult = await client.generateForecast(
      'data/processed/future_features.csv',
      forecastParams
    );
    console.log('Forecast Generated:', forecastResult.status);
    console.log('Forecast Length:', forecastResult.result.forecast.length);
    
    // Save results to file
    fs.writeFileSync(
      'forecast_results.json',
      JSON.stringify(forecastResult, null, 2)
    );
    console.log('Results saved to file');
    
  } catch (error) {
    console.error('Error:', error.response ? error.response.data : error.message);
  }
}

runExample();
```

## Command-Line Examples

### Using curl

**Check API Health:**
```bash
curl -X GET "http://localhost:8000/health"
```

**List Available Models:**
```bash
curl -X GET "http://localhost:8000/models/"
```

**List Trained Models:**
```bash
curl -X GET "http://localhost:8000/models/?trained=true"
```

**Train a Forecasting Model:**
```bash
curl -X POST "http://localhost:8000/forecasting/train" \
  -F "file=@data/processed/demand_data.csv" \
  -F "params={\"model_type\":\"ProphetModel\",\"model_name\":\"CLIDemandForecast\",\"feature_columns\":[\"date\",\"temperature\",\"promotion\"],\"target_column\":\"demand\",\"date_column\":\"date\",\"model_params\":{\"seasonality_mode\":\"additive\"},\"save_model\":true}"
```

**Generate a Forecast:**
```bash
curl -X POST "http://localhost:8000/forecasting/forecast" \
  -F "file=@data/processed/future_features.csv" \
  -F "params={\"model_name\":\"CLIDemandForecast\",\"model_type\":\"ProphetModel\",\"feature_columns\":[\"date\",\"temperature\",\"promotion\"],\"date_column\":\"date\",\"steps\":30,\"return_conf_int\":true}" \
  -o forecast_results.json
```

**Detect Anomalies:**
```bash
curl -X POST "http://localhost:8000/anomalies/detect" \
  -F "file=@data/processed/inventory_data.csv" \
  -F "params={\"model_name\":\"InventoryAnomaly\",\"model_type\":\"IsolationForestDetector\",\"feature_columns\":[\"demand\",\"inventory\",\"lead_time\"],\"return_details\":true}" \
  -o anomaly_results.json
```

### Using a Shell Script

Create a bash script for common operations:

```bash
#!/bin/bash
# supply_chain_api.sh

API_URL="http://localhost:8000"
API_KEY=""  # Add your API key if using authentication

# Function to check API health
check_health() {
  echo "Checking API health..."
  curl -s -X GET "${API_URL}/health"
  echo ""
}

# Function to train a model
train_model() {
  if [ "$#" -ne 3 ]; then
    echo "Usage: $0 train <data_file> <model_type> <model_name>"
    exit 1
  fi
  
  DATA_FILE=$1
  MODEL_TYPE=$2
  MODEL_NAME=$3
  
  echo "Training ${MODEL_TYPE} model '${MODEL_NAME}' with data from ${DATA_FILE}..."
  curl -s -X POST "${API_URL}/forecasting/train" \
    -F "file=@${DATA_FILE}" \
    -F "params={\"model_type\":\"${MODEL_TYPE}\",\"model_name\":\"${MODEL_NAME}\",\"feature_columns\":[\"date\",\"temperature\",\"promotion\"],\"target_column\":\"demand\",\"date_column\":\"date\",\"save_model\":true}"
  echo ""
}

# Function to generate a forecast
generate_forecast() {
  if [ "$#" -ne 3 ]; then
    echo "Usage: $0 forecast <data_file> <model_type> <model_name>"
    exit 1
  fi
  
  DATA_FILE=$1
  MODEL_TYPE=$2
  MODEL_NAME=$3
  
  echo "Generating forecast with model '${MODEL_NAME}' using data from ${DATA_FILE}..."
  curl -s -X POST "${API_URL}/forecasting/forecast" \
    -F "file=@${DATA_FILE}" \
    -F "params={\"model_name\":\"${MODEL_NAME}\",\"model_type\":\"${MODEL_TYPE}\",\"feature_columns\":[\"date\",\"temperature\",\"promotion\"],\"date_column\":\"date\",\"steps\":30}" \
    -o "${MODEL_NAME}_forecast.json"
  echo "Forecast saved to ${MODEL_NAME}_forecast.json"
}

# Parse command
case "$1" in
  health)
    check_health
    ;;
  train)
    train_model "$2" "$3" "$4"
    ;;
  forecast)
    generate_forecast "$2" "$3" "$4"
    ;;
  *)
    echo "Usage: $0 {health|train|forecast} [arguments]"
    exit 1
    ;;
esac
```

Make the script executable:
```bash
chmod +x supply_chain_api.sh
```

Use the script:
```bash
# Check health
./supply_chain_api.sh health

# Train a model
./supply_chain_api.sh train data/processed/demand_data.csv ProphetModel WeeklyDemand

# Generate a forecast
./supply_chain_api.sh forecast data/processed/future_features.csv ProphetModel WeeklyDemand
```

## Advanced Usage Scenarios

### Ensemble Forecasting

This example combines predictions from multiple models to create an ensemble forecast:

```python
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class EnsembleForecaster:
    def __init__(self, client, model_configs):
        self.client = client
        self.model_configs = model_configs
    
    def generate_ensemble_forecast(self, feature_data_file, steps=30, weights=None):
        """Generate forecasts from multiple models and combine them"""
        forecasts = []
        models = []
        
        # Generate individual forecasts
        for config in self.model_configs:
            model_type = config["model_type"]
            model_name = config["model_name"]
            
            print(f"Generating forecast with {model_name}...")
            forecast_params = {
                "model_name": model_name,
                "model_type": model_type,
                "feature_columns": config["feature_columns"],
                "date_column": config["date_column"],
                "steps": steps,
                "return_conf_int": True
            }
            
            try:
                result = self.client.generate_forecast(feature_data_file, forecast_params)
                if result["status"] == "success":
                    forecasts.append(result["result"]["forecast"])
                    models.append(model_name)
                    print(f"  Success - generated forecast of length {len(result['result']['forecast'])}")
                else:
                    print(f"  Failed - {result.get('message', 'Unknown error')}")
            except Exception as e:
                print(f"  Exception: {str(e)}")
        
        # Apply weights or use equal weighting
        if weights is None:
            weights = [1/len(forecasts)] * len(forecasts)
        
        # Ensure we have forecasts to combine
        if not forecasts:
            raise ValueError("No successful forecasts were generated")
        
        # Create ensemble forecast
        ensemble_forecast = np.zeros(len(forecasts[0]))
        for i, forecast in enumerate(forecasts):
            ensemble_forecast += np.array(forecast) * weights[i]
        
        # Get dates from the last successful forecast
        dates = result["result"]["dates"]
        
        # Create ensemble confidence intervals (average of all models)
        lower_bounds = []
        upper_bounds = []
        for i, forecast in enumerate(forecasts):
            lower_bounds.append(result["result"]["lower_bound"])
            upper_bounds.append(result["result"]["upper_bound"])
        
        ensemble_lower = np.array(lower_bounds).mean(axis=0)
        ensemble_upper = np.array(upper_bounds).mean(axis=0)
        
        # Return ensemble results
        return {
            "dates": dates,
            "forecast": ensemble_forecast.tolist(),
            "lower_bound": ensemble_lower.tolist(),
            "upper_bound": ensemble_upper.tolist(),
            "models": models,
            "weights": weights
        }
    
    def plot_ensemble_forecast(self, ensemble_result, title="Ensemble Forecast"):
        """Plot the ensemble forecast with confidence intervals"""
        plt.figure(figsize=(12, 6))
        plt.plot(ensemble_result["dates"], ensemble_result["forecast"], 'b-', label="Ensemble Forecast")
        plt.fill_between(
            ensemble_result["dates"],
            ensemble_result["lower_bound"],
            ensemble_result["upper_bound"],
            alpha=0.3,
            color='b',
            label="Confidence Interval"
        )
        
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Create a model weight annotation
        weight_text = "Model Weights:\n"
        for model, weight in zip(ensemble_result["models"], ensemble_result["weights"]):
            weight_text += f"{model}: {weight:.2f}\n"
        
        plt.annotate(
            weight_text,
            xy=(0.02, 0.02),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
        )
        
        plt.savefig("ensemble_forecast.png")
        plt.show()
```

### Real-time Anomaly Monitoring

This example shows how to set up a continuous anomaly monitoring system:

```python
import pandas as pd
import time
from datetime import datetime
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='anomaly_monitor.log'
)
logger = logging.getLogger('AnomalyMonitor')

class AnomalyMonitor:
    def __init__(self, client, config):
        self.client = client
        self.model_name = config["model_name"]
        self.model_type = config["model_type"]
        self.feature_columns = config["feature_columns"]
        self.threshold = config.get("threshold", 0.05)
        self.check_interval = config.get("check_interval", 3600)  # in seconds
        self.data_source = config["data_source"]
        self.alert_config = config.get("alert_config", {})
        self.last_check_time = None
        self.last_anomalies = []
    
    def load_data(self):
        """Load data from the specified source"""
        # This could load from a database, API, etc.
        logger.info(f"Loading data from {self.data_source}")
        data = pd.read_csv(self.data_source)
        return data
    
    def check_for_anomalies(self):
        """Check for new anomalies in the data"""
        try:
            # Load and prepare data
            data = self.load_data()
            current_time = datetime.now()
            
            # If this is not the first check, filter to only new data
            if self.last_check_time:
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data = data[data['timestamp'] > self.last_check_time]
                elif 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data[data['date'] > self.last_check_time]
            
            if data.empty:
                logger.info("No new data to check")
                return []
            
            # Save filtered data to temp file
            temp_file = f"temp_data_{int(time.time())}.csv"
            data.to_csv(temp_file, index=False)
            
            # Set up anomaly detection params
            detection_params = {
                "model_name": self.model_name,
                "model_type": self.model_type,
                "feature_columns": self.feature_columns,
                "threshold": self.threshold,
                "return_details": True
            }
            
            # Detect anomalies
            logger.info(f"Checking {len(data)} data points for anomalies")
            result = self.client.detect_anomalies(temp_file, detection_params)
            
            # Clean up temp file
            import os
            os.remove(temp_file)
            
            # Process results
            if result["status"] == "success" and "anomalies" in result["results"]:
                anomalies = [
                    a for a in result["results"]["anomalies"] 
                    if a["is_anomaly"]
                ]
                
                logger.info(f"Found {len(anomalies)} anomalies out of {len(data)} data points")
                
                # Update last check time
                self.last_check_time = current_time
                self.last_anomalies = anomalies
                
                # If anomalies found, send alerts
                if anomalies and self.alert_config.get("enabled", False):
                    self.send_alerts(anomalies)
                
                return anomalies
            else:
                logger.error(f"Error in anomaly detection: {result.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.exception(f"Exception during anomaly check: {str(e)}")
            return []
    
    def send_alerts(self, anomalies):
        """Send alerts when anomalies are detected"""
        if not self.alert_config.get("email", {}).get("enabled", False):
            return
        
        try:
            # Create email content
            email_config = self.alert_config["email"]
            subject = f"Supply Chain Anomaly Alert: {len(anomalies)} anomalies detected"
            
            body = f"""
            <html>
            <body>
            <h2>Supply Chain Anomaly Alert</h2>
            <p>The system has detected {len(anomalies)} anomalies in recent data.</p>
            <h3>Top Anomalies:</h3>
            <table border="1">
            <tr>
            """
            
            # Add table headers for each column in the anomalies
            for key in anomalies[0].keys():
                if key != "is_anomaly":
                    body += f"<th>{key}</th>"
            body += "</tr>"
            
            # Add rows for top 5 anomalies (or fewer if less than 5)
            for anomaly in anomalies[:5]:
                body += "<tr>"
                for key, value in anomaly.items():
                    if key != "is_anomaly":
                        body += f"<td>{value}</td>"
                body += "</tr>"
            
            body += """
            </table>
            <p>Please investigate these anomalies as they may indicate supply chain disruptions.</p>
            </body>
            </html>
            """
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = email_config["from"]
            msg['To'] = ", ".join(email_config["to"])
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            if email_config.get("use_tls", False):
                server.starttls()
            if email_config.get("username") and email_config.get("password"):
                server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Sent anomaly alert email to {', '.join(email_config['to'])}")
        
        except Exception as e:
            logger.exception(f"Failed to send alert email: {str(e)}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        logger.info(f"Starting anomaly monitoring with {self.model_name}, checking every {self.check_interval} seconds")
        
        try:
            while True:
                self.check_for_anomalies()
                logger.info(f"Sleeping for {self.check_interval} seconds")
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.exception(f"Monitoring stopped due to error: {str(e)}")
```

## Integration Examples

### Integration with ERP System

This example shows how to integrate the forecasting API with an ERP system for automated inventory planning:

```python
import pandas as pd
import sqlalchemy
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ERP_Integration")

class ERPIntegration:
    def __init__(self, api_client, db_config, forecast_config):
        self.client = api_client
        self.db_engine = sqlalchemy.create_engine(
            f"postgresql://{db_config['username']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        self.forecast_config = forecast_config
    
    def extract_data(self):
        """Extract relevant data from ERP system"""
        logger.info("Extracting data from ERP system")
        
        query = """
        SELECT 
            p.product_id,
            p.product_name,
            i.warehouse_id,
            w.warehouse_name,
            i.date,
            i.quantity as inventory,
            COALESCE(s.quantity, 0) as sales,
            COALESCE(pr.is_promotion, 0) as promotion
        FROM 
            inventory i
        JOIN 
            products p ON i.product_id = p.product_id
        JOIN 
            warehouses w ON i.warehouse_id = w.warehouse_id
        LEFT JOIN 
            sales s ON i.product_id = s.product_id AND i.date = s.date
        LEFT JOIN 
            promotions pr ON i.product_id = pr.product_id AND i.date BETWEEN pr.start_date AND pr.end_date
        WHERE 
            i.date >= NOW() - INTERVAL '180 days'
        ORDER BY 
            p.product_id, i.warehouse_id, i.date
        """
        
        data = pd.read_sql(query, self.db_engine)
        logger.info(f"Extracted {len(data)} records")
        
        return data
    
    def prepare_forecast_data(self, data):
        """Prepare data for forecasting"""
        logger.info("Preparing forecast data")
        
        # Pivot data to create separate files for each product/warehouse
        products = data['product_id'].unique()
        warehouses = data['warehouse_id'].unique()
        
        prepared_files = []
        
        for product_id in products:
            for warehouse_id in warehouses:
                # Filter data for this product/warehouse
                product_data = data[
                    (data['product_id'] == product_id) & 
                    (data['warehouse_id'] == warehouse_id)
                ]
                
                if len(product_data) >= 30:  # Only forecast if enough data
                    # Create a temp file
                    filename = f"temp_forecast_{product_id}_{warehouse_id}.csv"
                    product_data.to_csv(filename, index=False)
                    
                    prepared_files.append({
                        'filename': filename,
                        'product_id': product_id,
                        'product_name': product_data['product_name'].iloc[0],
                        'warehouse_id': warehouse_id,
                        'warehouse_name': product_data['warehouse_name'].iloc[0]
                    })
        
        logger.info(f"Prepared {len(prepared_files)} files for forecasting")
        return prepared_files
    
    def generate_forecasts(self, prepared_files):
        """Generate forecasts for each product/warehouse"""
        logger.info("Generating forecasts")
        
        forecast_results = []
        
        for file_info in prepared_files:
            logger.info(f"Forecasting for product {file_info['product_id']} at warehouse {file_info['warehouse_id']}")
            
            # Configure forecast parameters
            forecast_params = {
                **self.forecast_config,
                "model_name": self.forecast_config["model_name"],
                "model_type": self.forecast_config["model_type"],
                "feature_columns": self.forecast_config["feature_columns"],
                "date_column": "date",
                "steps": 30  # Forecast for next 30 days
            }
            
            try:
                # Generate forecast
                result = self.client.generate_forecast(file_info['filename'], forecast_params)
                
                if result["status"] == "success":
                    # Add product/warehouse info to result
                    result['product_id'] = file_info['product_id']
                    result['product_name'] = file_info['product_name']
                    result['warehouse_id'] = file_info['warehouse_id']
                    result['warehouse_name'] = file_info['warehouse_name']
                    
                    forecast_results.append(result)
                    logger.info(f"Forecast successful for {file_info['product_name']}")
                else:
                    logger.error(f"Forecast failed: {result.get('message', 'Unknown error')}")
            
            except Exception as e:
                logger.exception(f"Exception during forecasting: {str(e)}")
            
            finally:
                # Clean up temp file
                import os
                try:
                    os.remove(file_info['filename'])
                except:
                    pass
        
        logger.info(f"Generated {len(forecast_results)} forecasts")
        return forecast_results
    
    def update_erp_system(self, forecast_results):
        """Update ERP system with forecast results"""
        logger.info("Updating ERP system with forecasts")
        
        for result in forecast_results:
            try:
                # Prepare data for database
                forecast_data = []
                today = datetime.now().date()
                
                for i, (forecast_value, date_str) in enumerate(zip(
                    result['result']['forecast'],
                    result['result']['dates']
                )):
                    forecast_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S').date()
                    
                    forecast_data.append({
                        'product_id': result['product_id'],
                        'warehouse_id': result['warehouse_id'],
                        'forecast_date': forecast_date,
                        'forecast_value': forecast_value,
                        'lower_bound': result['result']['lower_bound'][i],
                        'upper_bound': result['result']['upper_bound'][i],
                        'days_ahead': (forecast_date - today).days,
                        'generated_at': datetime.now(),
                        'model_name': result['model_name']
                    })
                
                # Convert to DataFrame
                df = pd.DataFrame(forecast_data)
                
                # Insert into database
                df.to_sql(
                    'demand_forecasts', 
                    self.db_engine, 
                    if_exists='append', 
                    index=False
                )
                
                logger.info(f"Updated forecast for product {result['product_id']} at warehouse {result['warehouse_id']}")
            
            except Exception as e:
                logger.exception(f"Failed to update ERP system: {str(e)}")
    
    def run_integration(self):
        """Run the full integration process"""
        logger.info("Starting ERP integration process")
        
        try:
            # Extract data from ERP
            erp_data = self.extract_data()
            
            # Prepare data for forecasting
            prepared_files = self.prepare_forecast_data(erp_data)
            
            # Generate forecasts
            forecast_results = self.generate_forecasts(prepared_files)
            
            # Update ERP with forecasts
            self.update_erp_system(forecast_results)
            
            logger.info("Integration completed successfully")
            return {
                "status": "success",
                "forecasts_generated": len(forecast_results),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.exception(f"Integration failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
```

## Error Handling

### Python Error Handling Example

```python
def robust_api_call(client_method, *args, **kwargs):
    """Make a robust API call with error handling and retries"""
    import time
    
    max_retries = kwargs.pop('max_retries', 3)
    retry_delay = kwargs.pop('retry_delay', 2)
    
    for attempt in range(max_retries):
        try:
            # Make the API call
            result = client_method(*args, **kwargs)
            
            # Check for API-level errors
            if isinstance(result, dict) and result.get('status') == 'error':
                error_message = result.get('message', 'Unknown API error')
                print(f"API Error: {error_message}")
                
                # Check if error is retryable
                if 'timeout' in error_message.lower() or 'temporary' in error_message.lower():
                    print(f"Retryable error, attempt {attempt+1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Non-retryable API error
                    return {
                        'status': 'error',
                        'message': error_message,
                        'retried': attempt
                    }
            
            # Success
            return result
            
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying, attempt {attempt+1}/{max_retries}")
                time.sleep(retry_delay)
            else:
                return {
                    'status': 'error',
                    'message': f"Connection failed after {max_retries} attempts: {str(e)}",
                    'retried': attempt
                }
        
        except requests.exceptions.Timeout as e:
            print(f"Timeout error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying, attempt {attempt+1}/{max_retries}")
                time.sleep(retry_delay)
            else:
                return {
                    'status': 'error',
                    'message': f"Request timed out after {max_retries} attempts: {str(e)}",
                    'retried': attempt
                }
        
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return {
                'status': 'error',
                'message': f"Unexpected error: {str(e)}",
                'retried': attempt
            }
    
    # This should not be reached
    return {
        'status': 'error',
        'message': 'Unknown error in robust_api_call',
        'retried': max_retries
    }

# Example usage
client = SupplyChainClient()
result = robust_api_call(
    client.generate_forecast,
    "data/processed/future_features.csv",
    {
        "model_name": "DemandForecast",
        "model_type": "ProphetModel",
        "feature_columns": ["date", "temperature", "promotion"],
        "date_column": "date",
        "steps": 30
    },
    max_retries=5,
    retry_delay=3
)
```

## Next Steps

After exploring these API examples, you can:

1. **Extend the clients**: Add support for additional API endpoints and features
2. **Build custom dashboards**: Integrate the API into custom dashboards or business intelligence tools
3. **Implement automated workflows**: Set up scheduled forecasting and reporting
4. **Deploy in production**: Configure authentication, rate limiting, and high availability

For more information, see:
- [Complete API documentation](../api/api.md)
- [Model documentation](../models/models.md)
- [Monitoring and observability guide](../deployment/monitoring.md)
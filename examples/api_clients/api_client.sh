#!/bin/bash
# Supply Chain Forecaster API Client
# Usage: ./api_client.sh {health|train|forecast} [arguments]

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
    -F "params={\"model_type\":\"${MODEL_TYPE}\",\"model_name\":\"${MODEL_NAME}\",\"feature_columns\":[\"date\",\"price\",\"promotion\"],\"target_column\":\"demand\",\"date_column\":\"date\",\"save_model\":true}"
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
    -F "params={\"model_name\":\"${MODEL_NAME}\",\"model_type\":\"${MODEL_TYPE}\",\"feature_columns\":[\"date\",\"price\",\"promotion\"],\"date_column\":\"date\",\"steps\":30}" \
    -o "${MODEL_NAME}_forecast.json"
  echo "Forecast saved to ${MODEL_NAME}_forecast.json"
}

# Function to detect anomalies
detect_anomalies() {
  if [ "$#" -ne 3 ]; then
    echo "Usage: $0 anomalies <data_file> <model_type> <model_name>"
    exit 1
  fi
  
  DATA_FILE=$1
  MODEL_TYPE=$2
  MODEL_NAME=$3
  
  echo "Detecting anomalies with model '${MODEL_NAME}' using data from ${DATA_FILE}..."
  curl -s -X POST "${API_URL}/anomalies/detect" \
    -F "file=@${DATA_FILE}" \
    -F "params={\"model_name\":\"${MODEL_NAME}\",\"model_type\":\"${MODEL_TYPE}\",\"feature_columns\":[\"demand\",\"inventory\",\"lead_time\"]}" \
    -o "${MODEL_NAME}_anomalies.json"
  echo "Anomalies saved to ${MODEL_NAME}_anomalies.json"
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
  anomalies)
    detect_anomalies "$2" "$3" "$4"
    ;;
  *)
    echo "Usage: $0 {health|train|forecast|anomalies} [arguments]"
    echo "Examples:"
    echo "  $0 health"
    echo "  $0 train ../sample_data/sample_demand_data.csv ProphetModel DemandForecast"
    echo "  $0 forecast ../sample_data/sample_demand_data.csv ProphetModel DemandForecast"
    echo "  $0 anomalies ../sample_data/sample_inventory_data.csv IsolationForestDetector InventoryAnomalies"
    exit 1
    ;;
esac
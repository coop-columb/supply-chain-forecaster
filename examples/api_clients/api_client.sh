#!/bin/bash
# Supply Chain Forecaster API Client
# Usage: ./api_client.sh {health|train|forecast|auth|keys} [arguments]

API_URL="http://localhost:8000"
API_KEY=""  # Add your API key if using authentication
USERNAME=""  # Add your username if using basic authentication
PASSWORD=""  # Add your password if using basic authentication

# Set up authentication headers
AUTH_HEADER=""
if [ -n "$API_KEY" ]; then
  # API key authentication
  AUTH_HEADER="-H \"X-API-Key: $API_KEY\""
elif [ -n "$USERNAME" ] && [ -n "$PASSWORD" ]; then
  # Basic authentication
  AUTH_HEADER="-u \"$USERNAME:$PASSWORD\""
fi

# Function to check API health
check_health() {
  echo "Checking API health..."
  eval "curl -s -X GET \"${API_URL}/health\" $AUTH_HEADER"
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
  eval "curl -s -X POST \"${API_URL}/forecasting/train\" \
    $AUTH_HEADER \
    -F \"file=@${DATA_FILE}\" \
    -F \"params={\\\"model_type\\\":\\\"${MODEL_TYPE}\\\",\\\"model_name\\\":\\\"${MODEL_NAME}\\\",\\\"feature_columns\\\":[\\\"date\\\",\\\"price\\\",\\\"promotion\\\"],\\\"target_column\\\":\\\"demand\\\",\\\"date_column\\\":\\\"date\\\",\\\"save_model\\\":true}\""
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
  eval "curl -s -X POST \"${API_URL}/forecasting/forecast\" \
    $AUTH_HEADER \
    -F \"file=@${DATA_FILE}\" \
    -F \"params={\\\"model_name\\\":\\\"${MODEL_NAME}\\\",\\\"model_type\\\":\\\"${MODEL_TYPE}\\\",\\\"feature_columns\\\":[\\\"date\\\",\\\"price\\\",\\\"promotion\\\"],\\\"date_column\\\":\\\"date\\\",\\\"steps\\\":30}\" \
    -o \"${MODEL_NAME}_forecast.json\""
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
  eval "curl -s -X POST \"${API_URL}/anomalies/detect\" \
    $AUTH_HEADER \
    -F \"file=@${DATA_FILE}\" \
    -F \"params={\\\"model_name\\\":\\\"${MODEL_NAME}\\\",\\\"model_type\\\":\\\"${MODEL_TYPE}\\\",\\\"feature_columns\\\":[\\\"demand\\\",\\\"inventory\\\",\\\"lead_time\\\"]}\" \
    -o \"${MODEL_NAME}_anomalies.json\""
  echo "Anomalies saved to ${MODEL_NAME}_anomalies.json"
}

# Authentication functions
get_user_info() {
  echo "Getting current user information..."
  eval "curl -s -X GET \"${API_URL}/auth/me\" $AUTH_HEADER"
  echo ""
}

create_api_key() {
  if [ "$#" -lt 1 ]; then
    echo "Usage: $0 keys create <name> [expires_days] [scope]"
    exit 1
  fi
  
  KEY_NAME=$1
  EXPIRES_DAYS=${2:-"null"}
  SCOPE=${3:-"null"}
  
  if [ "$EXPIRES_DAYS" != "null" ]; then
    EXPIRES_DAYS="$EXPIRES_DAYS"
  fi
  
  if [ "$SCOPE" != "null" ]; then
    SCOPE="\"$SCOPE\""
  fi
  
  echo "Creating API key '$KEY_NAME'..."
  eval "curl -s -X POST \"${API_URL}/auth/keys\" \
    $AUTH_HEADER \
    -H \"Content-Type: application/json\" \
    -d '{\"name\":\"$KEY_NAME\",\"expires_days\":$EXPIRES_DAYS,\"scope\":$SCOPE}'"
  echo ""
}

list_api_keys() {
  echo "Listing API keys..."
  eval "curl -s -X GET \"${API_URL}/auth/keys\" $AUTH_HEADER"
  echo ""
}

revoke_api_key() {
  if [ "$#" -ne 1 ]; then
    echo "Usage: $0 keys revoke <key_id>"
    exit 1
  fi
  
  KEY_ID=$1
  
  echo "Revoking API key '$KEY_ID'..."
  eval "curl -s -X DELETE \"${API_URL}/auth/keys/$KEY_ID\" $AUTH_HEADER"
  echo ""
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
  auth)
    get_user_info
    ;;
  keys)
    case "$2" in
      create)
        create_api_key "$3" "$4" "$5"
        ;;
      list)
        list_api_keys
        ;;
      revoke)
        revoke_api_key "$3"
        ;;
      *)
        echo "Usage: $0 keys {create|list|revoke} [arguments]"
        echo "Examples:"
        echo "  $0 keys list"
        echo "  $0 keys create \"Production Service\" 90 \"read:* write:forecasts\""
        echo "  $0 keys revoke a1b2c3d4"
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Usage: $0 {health|train|forecast|anomalies|auth|keys} [arguments]"
    echo "Examples:"
    echo "  $0 health"
    echo "  $0 train ../sample_data/sample_demand_data.csv ProphetModel DemandForecast"
    echo "  $0 forecast ../sample_data/sample_demand_data.csv ProphetModel DemandForecast"
    echo "  $0 anomalies ../sample_data/sample_inventory_data.csv IsolationForestDetector InventoryAnomalies"
    echo "  $0 auth"
    echo "  $0 keys list"
    exit 1
    ;;
esac
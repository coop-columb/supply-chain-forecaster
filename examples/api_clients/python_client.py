import base64
import json

import matplotlib.pyplot as plt
import pandas as pd
import requests


class SupplyChainClient:
    def __init__(
        self,
        base_url="http://localhost:8000",
        api_key=None,
        username=None,
        password=None,
    ):
        self.base_url = base_url
        self.headers = {}

        # Set up authentication headers
        if api_key:
            # API key authentication
            self.headers["X-API-Key"] = api_key
        elif username and password:
            # Basic authentication
            auth_str = f"{username}:{password}"
            encoded_auth = base64.b64encode(auth_str.encode()).decode()
            self.headers["Authorization"] = f"Basic {encoded_auth}"

    def health_check(self):
        """Check if API is operational"""
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        return response.json()

    def list_models(self, trained=False, deployed=False):
        """List available models"""
        params = {"trained": trained, "deployed": deployed}
        response = requests.get(
            f"{self.base_url}/models/", params=params, headers=self.headers
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
            headers=self.headers,
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
            headers=self.headers,
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
            headers=self.headers,
        )
        return response.json()

    # Authentication endpoints
    def get_current_user(self):
        """Get information about the currently authenticated user"""
        response = requests.get(f"{self.base_url}/auth/me", headers=self.headers)
        return response.json()

    def create_api_key(self, name, expires_days=None, scope=None):
        """Create a new API key (admin only)"""
        data = {"name": name, "expires_days": expires_days, "scope": scope}

        response = requests.post(
            f"{self.base_url}/auth/keys", json=data, headers=self.headers
        )
        return response.json()

    def list_api_keys(self):
        """List all API keys (admin only)"""
        response = requests.get(f"{self.base_url}/auth/keys", headers=self.headers)
        return response.json()

    def revoke_api_key(self, key_id):
        """Revoke (delete) an API key (admin only)"""
        response = requests.delete(
            f"{self.base_url}/auth/keys/{key_id}", headers=self.headers
        )
        return response.json()


# Example usage
if __name__ == "__main__":
    # Example 1: No authentication
    client = SupplyChainClient()

    # Check if API is working
    health = client.health_check()
    print(f"API health: {health}")

    # Example 2: Using basic authentication
    admin_client = SupplyChainClient(username="admin", password="adminpassword")

    # Get authenticated user information
    user_info = admin_client.get_current_user()
    print(f"Authenticated as: {user_info}")

    # Create a new API key
    try:
        api_key_data = admin_client.create_api_key(
            name="Production Service",
            expires_days=90,
            scope="read:forecasts write:forecasts",
        )
        print(f"Created API key: {api_key_data}")

        # Store the key securely - it's only shown once!
        api_key = api_key_data["key"]

        # List all API keys
        keys = admin_client.list_api_keys()
        print(f"Available API keys: {keys}")
    except Exception as e:
        print(f"Authentication might not be enabled: {e}")

    # Example 3: Using API key authentication (if available)
    try:
        key_client = SupplyChainClient(api_key=api_key)

        # List available models
        models = key_client.list_models(trained=True)
        print(f"Available models: {models}")
    except:
        # Fall back to unauthenticated client
        models = client.list_models(trained=True)
        print(f"Available models: {models}")

    # Example forecast parameters
    forecast_params = {
        "model_name": "DemandForecast",
        "model_type": "ProphetModel",
        "feature_columns": ["date", "price", "promotion"],
        "date_column": "date",
        "steps": 30,
        "return_conf_int": True,
    }

    # Generate a forecast
    forecast = client.generate_forecast("sample_demand_data.csv", forecast_params)

    # Plot results
    if forecast["status"] == "success":
        plt.figure(figsize=(12, 6))
        plt.plot(
            forecast["result"]["dates"],
            forecast["result"]["forecast"],
            label="Forecast",
        )
        plt.fill_between(
            forecast["result"]["dates"],
            forecast["result"]["lower_bound"],
            forecast["result"]["upper_bound"],
            alpha=0.3,
            label="95% Confidence Interval",
        )
        plt.title("Demand Forecast")
        plt.xlabel("Date")
        plt.ylabel("Demand")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("demand_forecast.png")
        plt.show()

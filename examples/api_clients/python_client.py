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

# Example usage
if __name__ == "__main__":
    client = SupplyChainClient()
    
    # Check if API is working
    health = client.health_check()
    print(f"API health: {health}")
    
    # List available models
    models = client.list_models(trained=True)
    print(f"Available models: {models}")
    
    # Example forecast parameters
    forecast_params = {
        "model_name": "DemandForecast",
        "model_type": "ProphetModel",
        "feature_columns": ["date", "price", "promotion"],
        "date_column": "date",
        "steps": 30,
        "return_conf_int": True
    }
    
    # Generate a forecast
    forecast = client.generate_forecast(
        "sample_demand_data.csv", 
        forecast_params
    )
    
    # Plot results
    if forecast["status"] == "success":
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
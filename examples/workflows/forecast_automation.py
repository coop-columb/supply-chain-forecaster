#!/usr/bin/env python3
"""
Forecast Automation Example

This script demonstrates how to automate the forecasting process
for regular model retraining and forecast generation.
"""

import os
import sys
import logging
import argparse
import schedule
import time
import datetime
from pathlib import Path

# Add parent directory to Python path to import the API client
sys.path.append(str(Path(__file__).parent.parent))
from api_clients.python_client import SupplyChainClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('forecast_automation.log')
    ]
)
logger = logging.getLogger('ForecastAutomation')

def retrain_models(client, data_path, model_configs):
    """Retrain models with latest data"""
    logger.info(f"Starting model retraining at {datetime.datetime.now()}")
    
    for config in model_configs:
        try:
            # Update model name with timestamp
            config["model_name"] = f"{config['base_name']}_{datetime.datetime.now().strftime('%Y%m%d')}"
            
            # Train model
            logger.info(f"Training model: {config['model_name']}")
            result = client.train_forecasting_model(data_path, config)
            
            if result["status"] == "success":
                logger.info(f"Successfully trained model: {config['model_name']}")
                
                # Deploy if specified
                if config.get("auto_deploy", False):
                    deploy_result = client.deploy_model(config["model_name"])
                    logger.info(f"Deployment result: {deploy_result}")
            else:
                logger.error(f"Failed to train model: {config['model_name']}")
                logger.error(f"Error: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.exception(f"Exception during training of {config.get('base_name', 'unknown')}: {str(e)}")
    
    logger.info(f"Completed model retraining at {datetime.datetime.now()}")
    return True

def generate_forecasts(client, data_path, forecast_configs):
    """Generate forecasts using trained models"""
    logger.info(f"Starting forecast generation at {datetime.datetime.now()}")
    
    for config in forecast_configs:
        try:
            # Generate forecast
            logger.info(f"Generating forecast using model: {config['model_name']}")
            result = client.generate_forecast(data_path, config)
            
            if result["status"] == "success":
                logger.info(f"Successfully generated forecast with model: {config['model_name']}")
                
                # Save forecast results
                output_file = f"forecast_{config['model_name']}_{datetime.datetime.now().strftime('%Y%m%d')}.json"
                with open(output_file, 'w') as f:
                    import json
                    json.dump(result, f, indent=2)
                logger.info(f"Saved forecast to {output_file}")
            else:
                logger.error(f"Failed to generate forecast with model: {config['model_name']}")
                logger.error(f"Error: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.exception(f"Exception during forecast with {config.get('model_name', 'unknown')}: {str(e)}")
    
    logger.info(f"Completed forecast generation at {datetime.datetime.now()}")
    return True

def setup_schedules(client, data_paths, model_configs, forecast_configs):
    """Set up scheduled tasks for model retraining and forecasting"""
    # Schedule weekly model retraining (Monday at 2 AM)
    schedule.every().monday.at("02:00").do(
        retrain_models, 
        client=client,
        data_path=data_paths["training"],
        model_configs=model_configs
    )
    
    # Schedule daily forecast generation (every day at 3 AM)
    schedule.every().day.at("03:00").do(
        generate_forecasts, 
        client=client,
        data_path=data_paths["forecast"],
        forecast_configs=forecast_configs
    )
    
    logger.info("Scheduled tasks set up:")
    logger.info("- Model retraining: Every Monday at 2:00 AM")
    logger.info("- Forecast generation: Every day at 3:00 AM")

def run_scheduler():
    """Run the scheduler loop"""
    logger.info("Starting scheduler. Press Ctrl+C to exit.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")

def run_once(client, data_paths, model_configs, forecast_configs, task):
    """Run a specified task once"""
    if task == "retrain":
        retrain_models(client, data_paths["training"], model_configs)
    elif task == "forecast":
        generate_forecasts(client, data_paths["forecast"], forecast_configs)
    else:
        logger.error(f"Unknown task: {task}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Forecast Automation Tool")
    parser.add_argument("--mode", choices=["schedule", "retrain", "forecast"], default="schedule",
                        help="Mode to run: schedule (default), retrain, or forecast")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="API URL (default: http://localhost:8000)")
    parser.add_argument("--api-key", default=None,
                        help="API key for authentication (optional)")
    args = parser.parse_args()
    
    # Initialize API client
    client = SupplyChainClient(base_url=args.api_url, api_key=args.api_key)
    
    # Data paths configuration
    data_paths = {
        "training": "../sample_data/sample_demand_data.csv",  # Path to training data
        "forecast": "../sample_data/sample_demand_data.csv"   # Path to data for generating forecasts
    }
    
    # Model configuration
    model_configs = [
        {
            "base_name": "WeeklyDemandForecast",
            "model_type": "ProphetModel",
            "feature_columns": ["date", "price", "promotion"],
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
    
    # Forecast configuration
    forecast_configs = [
        {
            "model_name": "WeeklyDemandForecast_Latest",
            "model_type": "ProphetModel",
            "feature_columns": ["date", "price", "promotion"],
            "date_column": "date",
            "steps": 30,
            "return_conf_int": True,
            "from_deployment": True
        }
    ]
    
    # Run in specified mode
    if args.mode == "schedule":
        setup_schedules(client, data_paths, model_configs, forecast_configs)
        run_scheduler()
    else:
        run_once(client, data_paths, model_configs, forecast_configs, args.mode)

if __name__ == "__main__":
    main()
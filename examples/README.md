# Supply Chain Forecaster Examples

This directory contains example files and templates to help users get started with the Supply Chain Forecaster.

## Directory Structure

```
examples/
├── sample_data/            # Sample CSV data files for testing
├── api_clients/            # Example API client implementations
└── workflows/              # Example workflow scripts
```

## Sample Data

The `sample_data/` directory contains example CSV files that can be used for testing the forecasting and anomaly detection features:

- `sample_demand_data.csv`: Example demand data with date, demand, price, and promotion columns
- `sample_inventory_data.csv`: Example inventory data with product, warehouse, inventory levels, and lead time

## API Clients

The `api_clients/` directory contains example implementations of clients for the Supply Chain Forecaster API:

- `python_client.py`: Python client using the requests library
- `js_client.js`: JavaScript client using axios
- `api_client.sh`: Bash script using curl

## Workflows

The `workflows/` directory contains example scripts for automating common tasks:

- `forecast_automation.py`: Script for automating regular forecasting
- `anomaly_monitor.py`: Script for continuous anomaly monitoring
- `erp_integration.py`: Example of integrating with an ERP system

## Usage

These examples are meant to be customized for your specific needs. Copy the relevant example files to your project and modify them as needed.

For detailed instructions on using these examples, refer to the [API Examples](../docs/usage/api_examples.md) and [Common Workflows](../docs/usage/common_workflows.md) documentation.
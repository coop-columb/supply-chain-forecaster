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
      './sample_data/sample_demand_data.csv',
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
      './sample_data/sample_demand_data.csv',
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

// Run the example
runExample();
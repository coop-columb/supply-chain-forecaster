const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

class SupplyChainClient {
  constructor(baseUrl = 'http://localhost:8000', apiKey = null, username = null, password = null) {
    this.baseUrl = baseUrl;
    this.headers = {};
    
    // Set up authentication headers
    if (apiKey) {
      // API key authentication
      this.headers['X-API-Key'] = apiKey;
    } else if (username && password) {
      // Basic authentication
      const auth = Buffer.from(`${username}:${password}`).toString('base64');
      this.headers['Authorization'] = `Basic ${auth}`;
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

  // Authentication methods
  async getCurrentUser() {
    const response = await axios.get(
      `${this.baseUrl}/auth/me`,
      { headers: this.headers }
    );
    return response.data;
  }

  async createApiKey(name, expiresDay = null, scope = null) {
    const data = {
      name,
      expires_days: expiresDay,
      scope
    };

    const response = await axios.post(
      `${this.baseUrl}/auth/keys`,
      data,
      { headers: this.headers }
    );
    return response.data;
  }

  async listApiKeys() {
    const response = await axios.get(
      `${this.baseUrl}/auth/keys`,
      { headers: this.headers }
    );
    return response.data;
  }

  async revokeApiKey(keyId) {
    const response = await axios.delete(
      `${this.baseUrl}/auth/keys/${keyId}`,
      { headers: this.headers }
    );
    return response.data;
  }
}

// Example usage
async function runExample() {
  // Example 1: No authentication
  const client = new SupplyChainClient();
  
  try {
    // Check API health
    const health = await client.healthCheck();
    console.log('API Health:', health);
    
    // Example 2: Basic Authentication
    console.log('\nTrying Basic Authentication:');
    const adminClient = new SupplyChainClient(
      'http://localhost:8000',
      null, // no API key
      'admin', // username
      'adminpassword' // password
    );
    
    try {
      // Get authenticated user info
      const userInfo = await adminClient.getCurrentUser();
      console.log('Authenticated as:', userInfo);
      
      // Create a new API key
      const apiKeyData = await adminClient.createApiKey(
        'JS Service Key',
        90, // expires in 90 days
        'read:forecasts write:forecasts'
      );
      console.log('Created API key:', apiKeyData);
      
      // Store the key securely - it's only shown once!
      const apiKey = apiKeyData.key;
      
      // List all API keys
      const keys = await adminClient.listApiKeys();
      console.log('Available API keys:', keys);
      
      // Example 3: API Key Authentication
      console.log('\nTrying API Key Authentication:');
      const keyClient = new SupplyChainClient(
        'http://localhost:8000',
        apiKey
      );
      
      // List available models with API key auth
      const models = await keyClient.listModels(true);
      console.log('Available models:', models);
    } catch (authError) {
      console.log('Auth might not be enabled:', authError.message);
    }
    
    // Continue with regular API calls
    console.log('\nRegular API Usage:');
    
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
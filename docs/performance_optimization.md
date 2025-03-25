# Performance Optimization Guide

This guide covers the performance optimization strategy for the Supply Chain Forecaster system, including profiling techniques, implemented optimizations, common bottlenecks, and further optimization approaches.

## Performance Profiling

The system includes built-in profiling tools to help identify performance bottlenecks in various components:

- API endpoints
- Model training and inference
- Dashboard loading and rendering
- Data processing operations

### Running with Profiling Enabled

To enable profiling in your development environment:

```bash
# Run the API with profiling enabled
./scripts/run_with_profiling.sh

# Or set environment variables manually
export ENABLE_PROFILING=true
export PROFILING_SAMPLE_RATE=1.0
python -m api.main
```

### Collecting Profiling Data

Performance data is collected automatically when profiling is enabled. You can:

1. Access real-time profiling data through the API endpoint: `http://localhost:8000/profiling/stats`
2. Run the profiling script to collect more detailed data: `python scripts/profile_application.py`
3. Generate visualizations from profiling data: `python scripts/analyze_profiling.py`

## Implemented Optimizations

The following optimizations have been implemented in the system:

### Model Caching

We've implemented two levels of caching to improve model performance:

1. **Model Instance Caching**: Model instances are now cached in memory using an LRU (Least Recently Used) cache, eliminating redundant model loading from disk.

```python
# Model caching in ModelService.load_model
cached_model = model_cache.get(cache_key)
if cached_model is not None:
    logger.info(f"Loaded model '{model_name}' from cache")
    return cached_model
```

2. **Prediction Result Caching**: Model predictions are cached with expiry times to avoid redundant computations for the same input data.

```python
# Prediction caching decorator
@memoize_with_expiry()
def predict(self, X: pd.DataFrame, steps_ahead: int = 1, **kwargs):
    # Prediction logic here
```

### Optimized LSTM Sequence Creation

The LSTM model's sequence creation algorithm has been optimized using vectorized NumPy operations, replacing the previous loop-based implementation:

```python
# Vectorized implementation
indices = np.arange(n_samples - seq_length + 1)
X_sequences = np.zeros((len(indices), seq_length, n_features))

for i, start_idx in enumerate(indices):
    X_sequences[i] = X_array[start_idx:start_idx + seq_length]
```

### Auto ARIMA Optimization

The ARIMA model's parameter selection has been optimized to be faster and more efficient:

1. Parallel processing using all available CPU cores (`n_jobs=-1`)
2. Stepwise parameter search instead of exhaustive grid search
3. Early stopping based on information criterion
4. Development mode optimizations for faster experimentation

### Profiling Instrumentation

All critical model operations now include profiling instrumentation to measure performance:

```python
with profile_time(f"lstm_predict_{self.name}", "model"):
    # Model prediction logic
```

### Configurable Caching

Caching behavior can be configured through environment variables or config files:

- `ENABLE_MODEL_CACHING`: Enable/disable model instance caching
- `MODEL_CACHE_SIZE`: Maximum number of models to cache
- `ENABLE_RESPONSE_CACHING`: Enable/disable prediction result caching
- `RESPONSE_CACHE_TTL_SECONDS`: Time-to-live for cached predictions

## Common Performance Bottlenecks

Based on our analysis, these are the most common performance bottlenecks:

### API Performance

- **CSV File Loading**: Loading large CSV files can cause memory spikes
- **Model Loading**: Loading models from disk for each request is slow
- **Data Serialization**: Converting large datasets to/from JSON can be inefficient

### Model Performance

- **Model Training**: Training deep learning models (LSTM) can be slow
- **Inference Latency**: Real-time prediction with complex models
- **Feature Engineering**: Preprocessing steps taking too long

### Dashboard Performance

- **Data Loading**: Loading large datasets for visualization
- **Rendering**: Complex charts with many data points
- **Callback Chains**: Multiple dependent callbacks creating a waterfall effect

## Optimization Strategies

### API Optimization

1. **Model Caching**:
   ```python
   # Cache models in memory using LRU cache
   @lru_cache(maxsize=10)
   def get_model(model_name, model_type):
       return load_model(model_name, model_type)
   ```

2. **Chunked File Processing**:
   ```python
   # Process large files in chunks
   def process_large_file(file_path):
       for chunk in pd.read_csv(file_path, chunksize=10000):
           process_chunk(chunk)
   ```

3. **Asynchronous Processing**:
   ```python
   # Use async for I/O-bound operations
   async def fetch_data(urls):
       tasks = [fetch_url(url) for url in urls]
       return await asyncio.gather(*tasks)
   ```

### Model Optimization

1. **Batch Inference**:
   ```python
   # Process multiple predictions at once
   def batch_predict(model, batch_data):
       return model.predict(batch_data)
   ```

2. **Model Quantization**:
   ```python
   # Reduce model size and improve inference speed
   def quantize_model(model):
       return tf.quantization.quantize_model(model)
   ```

3. **Feature Selection**:
   ```python
   # Use only the most important features
   def select_features(data, feature_importance):
       threshold = 0.01
       important_features = [f for f, imp in feature_importance.items() if imp > threshold]
       return data[important_features]
   ```

### Dashboard Optimization

1. **Data Downsampling**:
   ```python
   # Downsample large datasets for visualization
   def downsample_for_chart(df, max_points=1000):
       if len(df) > max_points:
           return df.iloc[::len(df)//max_points]
       return df
   ```

2. **Lazy Loading**:
   ```python
   # Load components only when needed
   @app.callback(
       Output("component-container", "children"),
       [Input("tab", "value")]
   )
   def load_tab_content(tab):
       if tab == "tab1":
           return create_tab1_content()
       return []
   ```

3. **Client-Side Callbacks**:
   ```javascript
   // Use client-side callbacks for simple UI interactions
   app.clientside_callback(
       """
       function(value) {
           return value;
       }
       """,
       Output("output-id", "children"),
       Input("input-id", "value"),
   )
   ```

## Measuring Optimization Results

After implementing optimizations, it's important to measure their impact:

1. Run the profiling script: `python scripts/profile_application.py`
2. Run the model optimization test script: `python scripts/run_model_optimization_tests.py`
3. Compare the results before and after optimization
4. Use the built-in Prometheus metrics to monitor long-term performance trends

The optimization test script generates a report in `docs/performance/optimization_results.png` and `docs/performance/optimization_summary.txt` that quantifies the performance improvements from our optimizations:

- LSTM sequence creation speed improvements
- ARIMA prediction performance metrics
- Model caching speedup factors
- Memory usage improvements

## Further Resources

- [FastAPI Performance Tips](https://fastapi.tiangolo.com/advanced/performance/)
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)
- [Dash Performance Optimization](https://dash.plotly.com/performance)
- [Pandas Optimization Techniques](https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html)
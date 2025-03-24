# Troubleshooting Guide

This guide provides solutions for common issues you might encounter when using the Supply Chain Forecaster system.

## Table of Contents

1. [Dashboard Issues](#dashboard-issues)
2. [API Issues](#api-issues)
3. [Model Training Problems](#model-training-problems)
4. [Forecasting Issues](#forecasting-issues)
5. [Anomaly Detection Issues](#anomaly-detection-issues)
6. [Data Preparation Issues](#data-preparation-issues)
7. [Deployment and Infrastructure Issues](#deployment-and-infrastructure-issues)

## Dashboard Issues

### Dashboard Not Loading

**Symptoms:**
- Blank page when accessing the dashboard URL
- Browser console errors
- "Connection refused" message

**Possible Causes and Solutions:**

1. **Dashboard service not running**
   - Check if the service is running: `docker-compose ps dashboard`
   - Start the service if it's not running: `docker-compose up -d dashboard`

2. **Port conflict**
   - Another application might be using port 8050
   - Check for port usage: `lsof -i :8050`
   - Change the dashboard port in `.env` or `docker-compose.yml`

3. **Network connectivity issues**
   - Ensure there are no firewall rules blocking access
   - Try accessing from localhost if possible: `http://localhost:8050`

4. **Browser cache issues**
   - Clear your browser cache and cookies
   - Try a different browser or an incognito/private window

### Dashboard Components Not Rendering

**Symptoms:**
- Dashboard loads but some components are missing
- Empty charts or tables
- JavaScript console errors

**Possible Causes and Solutions:**

1. **JavaScript errors**
   - Check the browser console for error messages
   - Ensure all required assets are loading correctly

2. **API connectivity issues**
   - Verify the API service is running: `docker-compose ps api`
   - Check the API is accessible: `curl http://localhost:8000/health`
   - Ensure the dashboard is configured with the correct API URL

3. **CSS or styling issues**
   - Try disabling browser extensions that might interfere with CSS
   - Ensure the browser is supported (Chrome, Firefox, Safari, Edge)

### File Upload Not Working

**Symptoms:**
- "No file selected" error despite selecting a file
- File upload appears to complete but data doesn't load
- Error messages after file selection

**Possible Causes and Solutions:**

1. **File too large**
   - Check if the file size exceeds the maximum allowed (default: 10MB)
   - For large files, consider pre-processing to reduce size

2. **Invalid file format**
   - Ensure the file is a valid CSV with the expected columns
   - Check for special characters or encoding issues in the file
   - Validate the CSV format using a tool like Excel or LibreOffice

3. **Permission issues**
   - Check that the container has write access to the upload directory
   - Verify the user permissions inside the container

## API Issues

### API Not Responding

**Symptoms:**
- Timeout when accessing API endpoints
- "Connection refused" errors
- Dashboard shows API connectivity errors

**Possible Causes and Solutions:**

1. **API service not running**
   - Check if the service is running: `docker-compose ps api`
   - Start the service if needed: `docker-compose up -d api`

2. **API service crashed**
   - Check the API logs for errors: `docker-compose logs api`
   - Look for error messages, stack traces, or out-of-memory errors
   - Restart the service: `docker-compose restart api`

3. **Port or networking issues**
   - Verify the API port is correctly exposed in Docker
   - Check for firewall rules or network segmentation issues
   - Try connecting from inside the Docker network

### API Returning Errors

**Symptoms:**
- HTTP 4xx or 5xx status codes
- Error response JSON with error messages
- Inconsistent behavior between endpoints

**Possible Causes and Solutions:**

1. **Invalid request parameters**
   - Check the API documentation for correct parameter format
   - Ensure all required parameters are provided
   - Validate JSON or form data formatting

2. **Authentication issues**
   - Verify API keys or authentication tokens if enabled
   - Check authorization settings for the endpoints
   - Ensure credentials haven't expired

3. **Server-side errors**
   - Check API logs for detailed error messages
   - Look for database connection issues or resource constraints
   - Verify that dependent services are running

### Slow API Response

**Symptoms:**
- API calls take a long time to complete
- Timeout errors for complex operations
- Increasing latency over time

**Possible Causes and Solutions:**

1. **Resource constraints**
   - Check CPU and memory usage
   - Consider scaling the API service with more resources
   - Optimize database queries or caching strategies

2. **Large data processing**
   - For operations with large datasets, increase timeout settings
   - Consider batch processing for large operations
   - Optimize code paths for better performance

3. **Concurrent requests**
   - Check if the API is handling too many simultaneous requests
   - Implement request throttling or rate limiting
   - Scale horizontally with multiple API instances

## Model Training Problems

### Training Fails to Start

**Symptoms:**
- "Training failed" message immediately after submission
- Error message about invalid parameters
- No log entries for training process

**Possible Causes and Solutions:**

1. **Invalid model configuration**
   - Check that all required parameters are provided
   - Verify parameter values are within acceptable ranges
   - Ensure feature columns exist in the dataset

2. **Incompatible data format**
   - Verify the data format matches the model's expectations
   - Check for missing or invalid values in required columns
   - Ensure date columns are in the correct format (YYYY-MM-DD)

3. **Resource allocation issues**
   - Check if there's enough memory available for training
   - Consider reducing dataset size for initial testing
   - Increase container memory limits if necessary

### Training Takes Too Long

**Symptoms:**
- Model training runs for much longer than expected
- Training process doesn't complete within reasonable time
- System becomes unresponsive during training

**Possible Causes and Solutions:**

1. **Dataset too large**
   - Consider sampling or aggregating the data
   - Reduce the training window or feature set
   - Use a more efficient algorithm for large datasets

2. **Complex model configuration**
   - Simplify model parameters for initial training
   - Reduce iterations, estimators, or epochs
   - Choose a more efficient model type for the data size

3. **Resource constraints**
   - Increase memory and CPU allocation
   - Optimize the training process with batching
   - Consider distributed training for very large models

### Training Produces Poor Results

**Symptoms:**
- High error metrics (RMSE, MAE, etc.)
- Unrealistic forecasts
- Model fails to capture patterns in the data

**Possible Causes and Solutions:**

1. **Data quality issues**
   - Check for outliers, missing values, or incorrect data
   - Verify date ranges and consistency
   - Ensure feature engineering is appropriate for the model

2. **Inappropriate model selection**
   - Try a different model type that might better fit the data
   - Adjust hyperparameters to better match the data characteristics
   - Consider ensemble approaches for complex patterns

3. **Insufficient training data**
   - Ensure enough historical data is available
   - Add relevant features that might improve prediction
   - Consider data augmentation techniques

## Forecasting Issues

### Unrealistic Forecasts

**Symptoms:**
- Forecasts show implausible values (negative demand, extreme growth)
- Seasonal patterns not reflected in forecasts
- Forecasts ignore known future events

**Possible Causes and Solutions:**

1. **Data preparation issues**
   - Check for data anomalies in the training set
   - Ensure seasonality and trends are properly captured
   - Add relevant external factors (holidays, promotions, etc.)

2. **Model limitations**
   - Different models handle seasonality and trends differently
   - Try a model better suited to your data characteristics
   - Add explicit seasonality components if available

3. **Feature importance**
   - Verify that important features are included
   - Check feature engineering and transformations
   - For XGBoost and LSTM models, review feature importance

### Forecast Confidence Intervals Too Wide

**Symptoms:**
- Very wide confidence intervals making forecasts less useful
- Upper and lower bounds span an impractically large range
- Intervals grow extremely wide over the forecast horizon

**Possible Causes and Solutions:**

1. **High data volatility**
   - Check for high variance in the historical data
   - Consider smoothing techniques for noisy data
   - Segment data if different patterns exist

2. **Model uncertainty**
   - Use models with better uncertainty quantification
   - Adjust model parameters affecting uncertainty estimates
   - For Prophet, reduce `uncertainty_samples` or adjust `interval_width`

3. **Insufficient data**
   - Ensure enough historical data points for reliable intervals
   - Add more relevant features to reduce uncertainty
   - Consider ensemble methods for more reliable intervals

### Forecasts Don't Capture Seasonality

**Symptoms:**
- Seasonal patterns present in historical data missing from forecasts
- Flat forecast despite clear cycles in the data
- Wrong seasonal timing in forecasts

**Possible Causes and Solutions:**

1. **Incorrect model configuration**
   - For Prophet, ensure seasonality is enabled (`yearly`, `weekly`, `daily`)
   - For ARIMA, include seasonal orders
   - For XGBoost, include engineered seasonal features

2. **Insufficient historical data**
   - Ensure data spans at least 2 full seasonal cycles
   - Check for gaps in the historical data
   - Verify date parsing is correct

3. **Seasonality overwhelmed by other factors**
   - Check for trend dominance masking seasonality
   - Review feature importance in the model
   - Consider isolating and modeling seasonality separately

## Anomaly Detection Issues

### Too Many Anomalies Detected

**Symptoms:**
- Large percentage of data points flagged as anomalies
- Normal variations incorrectly identified as anomalies
- Clustering of anomalies in normal periods

**Possible Causes and Solutions:**

1. **Threshold too sensitive**
   - Increase the anomaly threshold parameter
   - For Isolation Forest, decrease `contamination`
   - For Statistical detector, increase standard deviation multiplier

2. **Insufficient training data**
   - Ensure enough normal data for the model to learn patterns
   - Include diverse normal patterns in training data
   - Consider semi-supervised approaches if normal data is limited

3. **Inappropriate feature selection**
   - Review selected features for relevance to anomalies
   - Remove highly volatile features that aren't indicative of anomalies
   - Normalize or scale features appropriately

### Missing Important Anomalies

**Symptoms:**
- Known anomalies not detected by the system
- Only extreme anomalies are identified
- Subtle but important patterns missed

**Possible Causes and Solutions:**

1. **Threshold too conservative**
   - Decrease the anomaly threshold parameter
   - For Isolation Forest, increase `contamination`
   - Consider different anomaly scoring methods

2. **Feature limitations**
   - Add features that better capture the anomaly patterns
   - Consider domain-specific transformations
   - Create compound features that might better indicate anomalies

3. **Algorithm limitations**
   - Try different anomaly detection algorithms
   - Consider ensemble approaches combining multiple detectors
   - Use specialized algorithms for time series anomalies

### Inconsistent Anomaly Detection

**Symptoms:**
- Different results when running detection multiple times
- Random or unpredictable anomaly scores
- Anomalies detected in one run but missed in another

**Possible Causes and Solutions:**

1. **Random algorithm components**
   - For Isolation Forest, set a fixed random seed
   - Increase the number of estimators for more stability
   - Use ensemble methods to reduce variance

2. **Data ordering effects**
   - Ensure consistent data preprocessing and ordering
   - For methods sensitive to data order, standardize the approach
   - Consider deterministic algorithms for reproducibility

3. **Threshold instability**
   - Use percentile-based thresholds instead of fixed values
   - Calculate thresholds based on the full score distribution
   - Implement adaptive thresholding methods

## Data Preparation Issues

### CSV Import Errors

**Symptoms:**
- File upload fails with format errors
- Missing columns or rows after import
- Data types incorrectly interpreted

**Possible Causes and Solutions:**

1. **Delimiter issues**
   - Ensure the CSV uses commas as delimiters (or specify delimiter)
   - Check for inconsistent delimiters within the file
   - Verify quotes are properly used for text with commas

2. **Encoding problems**
   - Save CSV files with UTF-8 encoding
   - Check for and remove any BOM markers
   - Convert special characters that might cause issues

3. **Header issues**
   - Ensure the CSV has a header row
   - Check for duplicate column names
   - Avoid special characters in column names

### Date Parsing Issues

**Symptoms:**
- "Invalid date format" errors
- Dates interpreted incorrectly (e.g., month/day swapped)
- Date-related functions fail

**Possible Causes and Solutions:**

1. **Inconsistent date formats**
   - Use ISO format dates (YYYY-MM-DD) for compatibility
   - Ensure dates are consistently formatted throughout the file
   - Pre-process dates in Excel or another tool before importing

2. **Locale-specific formatting**
   - Be aware of regional date formats (MM/DD/YYYY vs. DD/MM/YYYY)
   - Explicitly specify date format when parsing
   - Convert all dates to ISO format before importing

3. **Time zone issues**
   - Use consistent time zones throughout the data
   - Consider converting all timestamps to UTC
   - Document the time zone used for analysis

### Missing or Invalid Data

**Symptoms:**
- Errors about required columns missing
- Unexpected NULL values in results
- Charts with gaps or discontinuities

**Possible Causes and Solutions:**

1. **Incomplete records**
   - Check for and fill missing values appropriately
   - Consider interpolation for time series data
   - Use backward or forward fill for categorical data

2. **Outliers and invalid values**
   - Screen for and address extreme values
   - Implement domain-specific validity checks
   - Consider windsorizing or transforming outliers

3. **Inconsistent data types**
   - Ensure numeric columns contain only numbers
   - Watch for mixed formats in date columns
   - Check for unexpected string values in numeric columns

## Deployment and Infrastructure Issues

### Container Startup Failures

**Symptoms:**
- Containers exit immediately after starting
- Docker Compose shows services as "Exit" status
- No logs produced or logs show critical errors

**Possible Causes and Solutions:**

1. **Configuration errors**
   - Check environment variables and configuration files
   - Verify volume mounts and permissions
   - Ensure required services are available (database, etc.)

2. **Resource constraints**
   - Check for memory limits and OOM (Out Of Memory) kills
   - Verify CPU allocation is sufficient
   - Monitor disk space and I/O bottlenecks

3. **Dependency issues**
   - Verify all required dependencies are installed
   - Check for version conflicts
   - Ensure external services are accessible

### Memory Leaks

**Symptoms:**
- Increasing memory usage over time
- Performance degradation with uptime
- Container restarts due to OOM errors

**Possible Causes and Solutions:**

1. **Resource cleanup issues**
   - Check for proper garbage collection
   - Monitor for large objects staying in memory
   - Look for caching that grows unbounded

2. **Connection or file handle leaks**
   - Ensure database connections are properly closed
   - Check for unclosed file handles
   - Verify network sockets are being released

3. **Memory-intensive operations**
   - Optimize algorithms that load large datasets
   - Implement batching for large data processing
   - Consider streaming approaches for large data

### Network Connectivity Issues

**Symptoms:**
- Services can't communicate with each other
- Intermittent connection failures
- Timeouts when accessing external resources

**Possible Causes and Solutions:**

1. **Docker network configuration**
   - Check Docker network settings and DNS resolution
   - Verify service names are correctly used for internal connections
   - Ensure ports are properly exposed

2. **Firewall or security group issues**
   - Check for restrictive firewall rules
   - Verify security groups allow necessary traffic
   - Test connectivity with simple tools like ping or curl

3. **Rate limiting or throttling**
   - Check for API rate limits being hit
   - Implement exponential backoff for retries
   - Monitor external service health and availability

## General Troubleshooting Steps

### Checking Logs

Logs are your primary tool for diagnosing issues. Here's how to access them:

1. **Docker container logs:**
   ```bash
   # View logs for a specific service
   docker-compose logs api
   
   # Follow logs in real-time
   docker-compose logs -f dashboard
   
   # View last 100 lines
   docker-compose logs --tail=100 api
   ```

2. **Application logs:**
   - API logs: Check the `/app/logs` directory in the API container
   - Dashboard logs: Check browser console for client-side issues

3. **System logs:**
   - Check host system logs for Docker issues: `/var/log/syslog` or `journalctl`
   - Monitor resource usage: `docker stats`

### Diagnostic Commands

Useful commands for diagnosing issues:

1. **Check service status:**
   ```bash
   docker-compose ps
   ```

2. **Check resource usage:**
   ```bash
   docker stats
   ```

3. **Verify network connectivity:**
   ```bash
   # From within a container
   docker-compose exec api ping dashboard
   
   # Check port availability
   nc -zv localhost 8000
   ```

4. **Examine container details:**
   ```bash
   docker inspect container_name
   ```

5. **Test API endpoints:**
   ```bash
   curl -v http://localhost:8000/health
   ```

### Common Quick Fixes

1. **Restart services:**
   ```bash
   docker-compose restart api dashboard
   ```

2. **Rebuild containers:**
   ```bash
   docker-compose build --no-cache api
   docker-compose up -d api
   ```

3. **Update dependencies:**
   ```bash
   docker-compose down
   git pull
   docker-compose pull
   docker-compose build --pull
   docker-compose up -d
   ```

4. **Reset database/state (caution - data loss):**
   ```bash
   docker-compose down -v  # Removes volumes
   docker-compose up -d
   ```

## Getting Help

If you've tried the troubleshooting steps and still have issues:

1. **Check existing issues:**
   - Review the project's GitHub issues for similar problems
   - Search the documentation for specific error messages

2. **Gather information:**
   - Collect logs showing the issue
   - Document steps to reproduce the problem
   - Note your environment details (OS, Docker version, etc.)

3. **Contact support:**
   - Open a GitHub issue with the information collected
   - Join the project's discussion forum or chat
   - Email the support team with detailed problem description
#!/usr/bin/env python
"""
Run model optimization tests to measure performance improvements.

This script loads test data and runs performance tests on both
the original and optimized model implementations.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from models.forecasting import ARIMAModel, LSTMModel, ProphetModel, XGBoostModel
from utils.profiling import profile_time, get_profiling_stats, reset_profiling_stats

# Configure logging
from utils import get_logger
logger = get_logger(__name__)

def load_test_data():
    """Load test data for model evaluation."""
    try:
        # Try to load sample data from examples
        sample_file = Path(__file__).parent.parent / "examples" / "sample_data" / "sample_demand_data.csv"
        df = pd.read_csv(sample_file)
        
        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        
        return df
    except Exception as e:
        # If sample data not available, generate synthetic data
        logger.warning(f"Could not load sample data: {e}. Generating synthetic data.")
        
        dates = pd.date_range(start="2022-01-01", periods=365)
        
        # Create synthetic time series with trend, seasonality, and noise
        trend = np.linspace(0, 10, len(dates))
        seasonality = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly seasonality
        seasonality += 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # Yearly seasonality
        noise = np.random.normal(0, 1, len(dates))
        
        demand = trend + seasonality + noise
        demand = np.maximum(demand, 0)  # Ensure non-negative values
        
        df = pd.DataFrame({
            "demand": demand,
            "store_id": 1,
            "product_id": 1,
        }, index=dates)
        
        # Add some features
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["year"] = df.index.year
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        return df

def test_lstm_sequence_creation(model, data, repetitions=10):
    """Test LSTM sequence creation performance."""
    X = data.drop("demand", axis=1)
    y = data["demand"]
    
    # Test original implementation
    original_time = 0
    for i in range(repetitions):
        start_time = time.time()
        X_seq, y_seq = model._create_sequences(X, y)
        original_time += time.time() - start_time
    
    return {
        "average_time_ms": (original_time / repetitions) * 1000,
        "sequences_created": len(X_seq),
        "sequence_length": model.params["sequence_length"],
        "feature_count": X.shape[1]
    }

def test_arima_prediction(model, data):
    """Test ARIMA model prediction performance."""
    X = data.drop("demand", axis=1)
    y = data["demand"]
    
    # First train the model
    with profile_time("arima_train", "model"):
        model.fit(X, y)
    
    # Test prediction
    with profile_time("arima_predict", "model"):
        predictions = model.predict(X, steps=30)
    
    return get_profiling_stats()["model"]

def test_model_caching(model_class, data, repetitions=5):
    """Test model caching performance."""
    X = data.drop("demand", axis=1)
    y = data["demand"]
    
    # Train the model
    model = model_class()
    model.fit(X, y)
    
    # Enable caching
    config.ENABLE_RESPONSE_CACHING = True
    
    # First prediction (cache miss)
    start_time = time.time()
    predictions = model.predict(X)
    first_prediction_time = (time.time() - start_time) * 1000
    
    # Subsequent predictions (cache hits)
    cache_hit_times = []
    for i in range(repetitions):
        start_time = time.time()
        predictions = model.predict(X)
        cache_hit_times.append((time.time() - start_time) * 1000)
    
    # Disable caching
    config.ENABLE_RESPONSE_CACHING = False
    
    return {
        "cache_miss_time_ms": first_prediction_time,
        "average_cache_hit_time_ms": sum(cache_hit_times) / len(cache_hit_times),
        "speedup_factor": first_prediction_time / (sum(cache_hit_times) / len(cache_hit_times))
    }

def run_all_tests():
    """Run all optimization tests and report results."""
    logger.info("Loading test data...")
    data = load_test_data()
    logger.info(f"Loaded data with shape {data.shape}")
    
    results = {}
    
    # Test LSTM sequence creation
    logger.info("Testing LSTM sequence creation optimization...")
    original_lstm = LSTMModel(sequence_length=10)
    lstm_results = test_lstm_sequence_creation(original_lstm, data)
    results["lstm_sequence_creation"] = lstm_results
    logger.info(f"LSTM sequence creation: {lstm_results['average_time_ms']:.2f}ms")
    
    # Test ARIMA prediction
    logger.info("Testing ARIMA prediction with optimization...")
    arima_model = ARIMAModel()
    reset_profiling_stats()
    arima_results = test_arima_prediction(arima_model, data)
    results["arima_prediction"] = arima_results
    logger.info(f"ARIMA prediction results: {arima_results}")
    
    # Test model caching
    logger.info("Testing model caching with LSTM...")
    caching_results = test_model_caching(LSTMModel, data)
    results["model_caching"] = caching_results
    logger.info(f"Model caching speedup: {caching_results['speedup_factor']:.2f}x")
    
    return results

def plot_results(results):
    """Plot optimization test results."""
    plt.figure(figsize=(12, 8))
    
    # Plot LSTM sequence creation times
    if "lstm_sequence_creation" in results:
        plt.subplot(2, 2, 1)
        lstm_results = results["lstm_sequence_creation"]
        plt.bar(["Original"], [lstm_results["average_time_ms"]])
        plt.title("LSTM Sequence Creation Time")
        plt.ylabel("Time (ms)")
    
    # Plot ARIMA prediction times
    if "arima_prediction" in results:
        plt.subplot(2, 2, 2)
        arima_results = results["arima_prediction"]
        if "arima_train" in arima_results and "arima_predict" in arima_results:
            times = [
                arima_results["arima_train"]["mean_ms"],
                arima_results["arima_predict"]["mean_ms"]
            ]
            plt.bar(["Training", "Prediction"], times)
            plt.title("ARIMA Training and Prediction Time")
            plt.ylabel("Time (ms)")
    
    # Plot model caching results
    if "model_caching" in results:
        plt.subplot(2, 2, 3)
        caching_results = results["model_caching"]
        plt.bar(["Cache Miss", "Cache Hit"], 
                [caching_results["cache_miss_time_ms"], caching_results["average_cache_hit_time_ms"]])
        plt.title(f"Model Caching Performance (Speedup: {caching_results['speedup_factor']:.2f}x)")
        plt.ylabel("Time (ms)")
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path(__file__).parent.parent / "docs" / "performance"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "optimization_results.png"
    plt.savefig(output_file)
    logger.info(f"Saved results plot to {output_file}")
    
    # Create summary text report
    report_file = output_dir / "optimization_summary.txt"
    with open(report_file, "w") as f:
        f.write("# Model Optimization Performance Report\n\n")
        
        if "lstm_sequence_creation" in results:
            lstm_results = results["lstm_sequence_creation"]
            f.write("## LSTM Sequence Creation\n")
            f.write(f"- Average time: {lstm_results['average_time_ms']:.2f}ms\n")
            f.write(f"- Sequences created: {lstm_results['sequences_created']}\n")
            f.write(f"- Sequence length: {lstm_results['sequence_length']}\n")
            f.write(f"- Feature count: {lstm_results['feature_count']}\n\n")
        
        if "arima_prediction" in results:
            arima_results = results["arima_prediction"]
            f.write("## ARIMA Model Performance\n")
            for operation, metrics in arima_results.items():
                f.write(f"### {operation}\n")
                for key, value in metrics.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
        
        if "model_caching" in results:
            caching_results = results["model_caching"]
            f.write("## Model Caching Performance\n")
            f.write(f"- Cache miss time: {caching_results['cache_miss_time_ms']:.2f}ms\n")
            f.write(f"- Average cache hit time: {caching_results['average_cache_hit_time_ms']:.2f}ms\n")
            f.write(f"- Speedup factor: {caching_results['speedup_factor']:.2f}x\n")
    
    logger.info(f"Saved summary report to {report_file}")

if __name__ == "__main__":
    logger.info("Running model optimization tests...")
    results = run_all_tests()
    plot_results(results)
    logger.info("Tests completed successfully!")
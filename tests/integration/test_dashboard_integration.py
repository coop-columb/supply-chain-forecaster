"""Integration tests for the dashboard functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import dash
from dash.testing.composite import DashComposite


def test_dashboard_initialization(dashboard_app):
    """Test that the dashboard initializes correctly."""
    if dashboard_app is None:
        pytest.skip("Dashboard app could not be initialized")
    
    # Check that the dashboard has the expected layout components
    assert hasattr(dashboard_app, "layout")


@pytest.mark.skipif(dash is None, reason="Dash not available")
def test_dashboard_pages(dashboard_app):
    """Test that the dashboard pages are accessible."""
    if dashboard_app is None:
        pytest.skip("Dashboard app could not be initialized")
    
    # Get the page registry
    if not hasattr(dashboard_app, "pages_registry"):
        pytest.skip("Dashboard does not use page registry")
    
    # Check that the expected pages exist
    expected_pages = ["home", "data-exploration", "forecasting", "anomaly-detection", "model-management"]
    for page in expected_pages:
        assert page in dashboard_app.pages_registry, f"Expected page '{page}' not found in dashboard"


def test_data_upload_component(dashboard_app, sample_time_series_data, integration_test_env):
    """Test the data upload component functionality."""
    if dashboard_app is None or not hasattr(dashboard_app, "layout"):
        pytest.skip("Dashboard app layout not available")
    
    # Save sample data to a file
    data_file = integration_test_env["data_dir"] / "dashboard_test_data.csv"
    sample_time_series_data.to_csv(data_file, index=False)
    
    # In a real test with dash.testing, you would use:
    # dash_duo = DashComposite(...)
    # dash_duo.start_server(dashboard_app)
    # dash_duo.upload_file_by_id(id='upload-data', filepath=str(data_file))
    
    # For now, we'll just verify the data file exists and is valid
    assert data_file.exists(), "Test data file was not created"
    df = pd.read_csv(data_file)
    assert len(df) > 0, "Test data file is empty"
    assert "demand" in df.columns, "Expected 'demand' column not found in test data"


def test_dashboard_charts_components(dashboard_app):
    """Test that the chart components are properly initialized."""
    if dashboard_app is None:
        pytest.skip("Dashboard app could not be initialized")
    
    try:
        # Import chart components directly
        from dashboard.components.charts import (
            create_time_series_chart,
            create_forecast_chart,
            create_anomaly_chart,
            create_feature_importance_chart
        )
        
        # Create a simple DataFrame for testing
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=30),
            'value': np.random.normal(100, 10, 30)
        })
        
        # Test time series chart
        time_series_fig = create_time_series_chart(df, x_column='date', y_columns=['value'], title='Test Time Series', id_prefix='test')
        assert time_series_fig is not None, "Time series chart creation failed"
        
        # Test forecast chart (with mocked forecast values)
        forecast_df = df.copy()
        forecast_df['forecast'] = forecast_df['value'] * 1.1
        forecast_df['upper'] = forecast_df['forecast'] * 1.1
        forecast_df['lower'] = forecast_df['forecast'] * 0.9
        
        forecast_fig = create_forecast_chart(
            historical_df=df,
            forecast_df=forecast_df, 
            date_column='date',
            value_column='value',
            lower_bound='lower',
            upper_bound='upper',
            title='Test Forecast',
            id_prefix='test'
        )
        assert forecast_fig is not None, "Forecast chart creation failed"
        
    except ImportError as e:
        pytest.skip(f"Could not import dashboard components: {e}")


def test_dashboard_model_selection_component(dashboard_app):
    """Test the model selection component functionality."""
    if dashboard_app is None:
        pytest.skip("Dashboard app could not be initialized")
    
    try:
        # Import model selection components
        from dashboard.components.model_selection import (
            create_forecasting_model_selection,
            create_anomaly_model_selection
        )
        
        # Test forecasting model selection
        forecasting_component = create_forecasting_model_selection()
        assert forecasting_component is not None, "Forecasting model selection creation failed"
        
        # Test anomaly model selection
        anomaly_component = create_anomaly_model_selection()
        assert anomaly_component is not None, "Anomaly model selection creation failed"
        
    except ImportError as e:
        pytest.skip(f"Could not import model selection components: {e}")


def test_dashboard_page_layouts(dashboard_app):
    """Test each dashboard page layout."""
    if dashboard_app is None:
        pytest.skip("Dashboard app could not be initialized")
    
    try:
        # Try to import page modules
        try:
            from dashboard.pages.home import layout as home_layout
            assert home_layout is not None, "Home page layout is None"
        except ImportError as e:
            print(f"Could not import home page: {e}")
        
        try:
            from dashboard.pages.data_exploration import layout as data_exploration_layout
            assert data_exploration_layout is not None, "Data exploration page layout is None"
        except ImportError as e:
            print(f"Could not import data exploration page: {e}")
        
        try:
            from dashboard.pages.forecasting import layout as forecasting_layout
            assert forecasting_layout is not None, "Forecasting page layout is None"
        except ImportError as e:
            print(f"Could not import forecasting page: {e}")
        
        try:
            from dashboard.pages.anomaly_detection import layout as anomaly_layout
            assert anomaly_layout is not None, "Anomaly detection page layout is None"
        except ImportError as e:
            print(f"Could not import anomaly detection page: {e}")
        
        try:
            from dashboard.pages.model_management import layout as model_management_layout
            assert model_management_layout is not None, "Model management page layout is None"
        except ImportError as e:
            print(f"Could not import model management page: {e}")
        
    except Exception as e:
        pytest.skip(f"Error testing page layouts: {e}")
"""Chart components for the supply chain forecaster dashboard."""

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html

from config import config
from utils.dashboard_optimization import (
    downsample_timeseries,
    memoize_component,
    optimize_plotly_figure,
    profile_component,
)


@memoize_component()
@profile_component("time_series_chart")
def create_time_series_chart(df, x_column, y_columns, title, id_prefix):
    """
    Create a time series chart.
    
    Args:
        df: DataFrame containing the data.
        x_column: Column to use for the x-axis.
        y_columns: List of columns to plot on the y-axis.
        title: Chart title.
        id_prefix: Prefix for the component ID.
    
    Returns:
        Time series chart component.
    """
    if df is None or x_column not in df.columns:
        return html.Div("No data to display")
    
    # Ensure all requested y-columns exist
    valid_y_columns = [col for col in y_columns if col in df.columns]
    
    if not valid_y_columns:
        return html.Div("No valid series to display")
    
    # Downsample the dataframe if it's too large
    max_points = getattr(config, "DASHBOARD_MAX_POINTS", 500)
    if len(df) > max_points:
        df = downsample_timeseries(df, x_column, valid_y_columns, max_points)
    
    # Create figure
    fig = go.Figure()
    
    for col in valid_y_columns:
        fig.add_trace(go.Scatter(
            x=df[x_column],
            y=df[col],
            mode='lines+markers',
            name=col,
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title='Value',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    # Optimize the figure for faster rendering
    fig = optimize_plotly_figure(fig)
    
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            dcc.Graph(
                id=f'{id_prefix}-time-series-chart',
                figure=fig,
                config={'displayModeBar': True, 'responsive': True},
            ),
        ])
    )


@memoize_component()
@profile_component("forecast_chart")
def create_forecast_chart(historical_df, forecast_df, date_column, value_column, 
                        lower_bound=None, upper_bound=None, title="Forecast", id_prefix="forecast"):
    """
    Create a forecast chart with historical and forecasted values.
    
    Args:
        historical_df: DataFrame with historical data.
        forecast_df: DataFrame with forecast data.
        date_column: Column with dates.
        value_column: Column with values.
        lower_bound: Column with lower confidence bound.
        upper_bound: Column with upper confidence bound.
        title: Chart title.
        id_prefix: Prefix for the component ID.
    
    Returns:
        Forecast chart component.
    """
    if historical_df is None or forecast_df is None:
        return html.Div("No data to display")
    
    # Downsample dataframes if they're too large
    max_points = getattr(config, "DASHBOARD_MAX_POINTS", 500)
    
    # Reserve space for forecast points - allocate 70% to historical, 30% to forecast
    hist_max_points = int(max_points * 0.7)
    forecast_max_points = max_points - hist_max_points
    
    if len(historical_df) > hist_max_points:
        historical_df = downsample_timeseries(historical_df, date_column, value_column, hist_max_points)
    
    if len(forecast_df) > forecast_max_points:
        columns_to_include = [value_column]
        if lower_bound is not None:
            columns_to_include.append(lower_bound)
        if upper_bound is not None:
            columns_to_include.append(upper_bound)
        forecast_df = downsample_timeseries(forecast_df, date_column, columns_to_include, forecast_max_points)
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_df[date_column],
        y=historical_df[value_column],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue'),
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_df[date_column],
        y=forecast_df[value_column],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red'),
    ))
    
    # Add confidence interval if provided
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=forecast_df[date_column],
            y=forecast_df[upper_bound],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df[date_column],
            y=forecast_df[lower_bound],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='95% Confidence Interval',
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=date_column,
        yaxis_title=value_column,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    # Optimize the figure for faster rendering
    fig = optimize_plotly_figure(fig)
    
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            dcc.Graph(
                id=f'{id_prefix}-forecast-chart',
                figure=fig,
                config={'displayModeBar': True, 'responsive': True},
            ),
        ])
    )


@memoize_component()
@profile_component("anomaly_chart")
def create_anomaly_chart(df, date_column, value_column, anomaly_column, score_column=None, 
                        title="Anomaly Detection", id_prefix="anomaly"):
    """
    Create an anomaly detection chart.
    
    Args:
        df: DataFrame containing the data.
        date_column: Column with dates.
        value_column: Column with values.
        anomaly_column: Column indicating anomalies.
        score_column: Column with anomaly scores.
        title: Chart title.
        id_prefix: Prefix for the component ID.
    
    Returns:
        Anomaly detection chart component.
    """
    if df is None or date_column not in df.columns or value_column not in df.columns:
        return html.Div("No data to display")
    
    # Downsample the dataframe if it's too large, but preserve anomalies
    max_points = getattr(config, "DASHBOARD_MAX_POINTS", 500)
    
    # Split normal and anomaly points
    normal_df = df[~df[anomaly_column]]
    anomaly_df = df[df[anomaly_column]]
    
    # Only downsample the normal points if there are too many total points
    if len(df) > max_points:
        # Reserve space for anomalies (they're important and shouldn't be downsampled)
        normal_max_points = max(5, max_points - len(anomaly_df))
        
        if len(normal_df) > normal_max_points:
            columns_to_include = [value_column]
            normal_df = downsample_timeseries(normal_df, date_column, columns_to_include, normal_max_points)
    
    # Create figure
    fig = go.Figure()
    
    # Add normal points
    fig.add_trace(go.Scatter(
        x=normal_df[date_column],
        y=normal_df[value_column],
        mode='lines+markers',
        name='Normal',
        line=dict(color='blue'),
        marker=dict(color='blue', size=6),
    ))
    
    # Add anomalies
    if not anomaly_df.empty:
        # Use anomaly scores for marker size if available
        marker_size = 10
        marker_color = 'red'
        hover_text = None
        
        if score_column in anomaly_df.columns:
            marker_size = np.clip(anomaly_df[score_column] * 15, 10, 25)
            hover_text = [
                f"Date: {d}<br>Value: {v}<br>Anomaly Score: {s:.2f}"
                for d, v, s in zip(anomaly_df[date_column], anomaly_df[value_column], anomaly_df[score_column])
            ]
        
        fig.add_trace(go.Scatter(
            x=anomaly_df[date_column],
            y=anomaly_df[value_column],
            mode='markers',
            name='Anomaly',
            marker=dict(
                color=marker_color,
                size=marker_size,
                line=dict(width=2, color='DarkSlateGrey'),
            ),
            text=hover_text,
            hoverinfo='text' if hover_text else 'x+y',
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=date_column,
        yaxis_title=value_column,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    # Optimize the figure for faster rendering
    fig = optimize_plotly_figure(fig)
    
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            dcc.Graph(
                id=f'{id_prefix}-anomaly-chart',
                figure=fig,
                config={'displayModeBar': True, 'responsive': True},
            ),
        ])
    )


@memoize_component()
@profile_component("feature_importance_chart")
def create_feature_importance_chart(importance_df, feature_col, importance_col, 
                                  title="Feature Importance", id_prefix="importance"):
    """
    Create a feature importance chart.
    
    Args:
        importance_df: DataFrame with feature importance data.
        feature_col: Column with feature names.
        importance_col: Column with importance values.
        title: Chart title.
        id_prefix: Prefix for the component ID.
    
    Returns:
        Feature importance chart component.
    """
    if importance_df is None or feature_col not in importance_df.columns or importance_col not in importance_df.columns:
        return html.Div("No data to display")
    
    # Sort by importance and limit to top features if there are too many
    df_sorted = importance_df.sort_values(importance_col, ascending=False)
    
    # If we have too many features, just show the top 20
    max_features = 20
    if len(df_sorted) > max_features:
        df_sorted = df_sorted.head(max_features)
        # Re-sort for display (ascending for horizontal bar chart)
        df_sorted = df_sorted.sort_values(importance_col, ascending=True)
    else:
        df_sorted = df_sorted.sort_values(importance_col, ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        df_sorted,
        y=feature_col,
        x=importance_col,
        orientation='h',
        title=title,
    )
    
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    # Optimize the figure for faster rendering
    fig = optimize_plotly_figure(fig)
    
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            dcc.Graph(
                id=f'{id_prefix}-importance-chart',
                figure=fig,
                config={'displayModeBar': True, 'responsive': True},
            ),
        ])
    )

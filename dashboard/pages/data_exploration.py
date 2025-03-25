"""Data exploration page for the supply chain forecaster dashboard."""

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, dcc, html

from dashboard.components.charts import create_time_series_chart
from dashboard.components.data_upload import create_upload_component
from utils.dashboard_optimization import memoize_component, profile_component


def create_data_exploration_layout():
    """
    Create the data exploration page layout.
    
    Returns:
        Data exploration page layout.
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("Data Exploration", className="mb-4"),
                html.P(
                    "Upload and explore your supply chain data. Visualize historical trends "
                    "and analyze key metrics to better understand your data.",
                    className="lead mb-4",
                ),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                create_upload_component("exploration"),
            ], width=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="exploration-summary-stats"),
            ], width=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="exploration-time-series-container"),
            ], width=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="exploration-correlation-container"),
            ], width=12),
        ]),
    ])


@memoize_component()
@profile_component("summary_stats")
def create_summary_stats(df):
    """
    Create summary statistics for the dataframe.
    
    Args:
        df: DataFrame to summarize.
    
    Returns:
        Summary statistics component.
    """
    if df is None or df.empty:
        return html.Div()
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        return html.Div("No numeric columns found in the data.")
    
    # Calculate summary statistics - limit to at most 15 columns for performance
    if len(numeric_cols) > 15:
        # Select columns based on variance (more interesting columns first)
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.index[:15].tolist()
    
    # Calculate summary statistics
    summary = df[numeric_cols].describe().round(2).reset_index()
    
    return dbc.Card([
        dbc.CardHeader(html.H5("Summary Statistics")),
        dbc.CardBody([
            dash_table.DataTable(
                data=summary.to_dict('records'),
                columns=[{'name': str(i), 'id': str(i)} for i in summary.columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                page_size=10,  # Paginate for better performance with large tables
            ),
        ]),
    ])


@memoize_component()
@profile_component("correlation_heatmap")
def create_correlation_heatmap(df):
    """
    Create correlation heatmap for numeric columns.
    
    Args:
        df: DataFrame to analyze.
    
    Returns:
        Correlation heatmap component.
    """
    if df is None or df.empty:
        return html.Div()
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        return html.Div("At least two numeric columns are required for correlation analysis.")
    
    # Limit to at most 20 columns for performance and readability
    if len(numeric_cols) > 20:
        # Select columns based on variance (more interesting columns first)
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.index[:20].tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr().round(2)
    
    # Create heatmap
    import plotly.express as px
    import plotly.graph_objs as go
    
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale=px.colors.diverging.RdBu_r,
        zmin=-1, zmax=1,
    )
    
    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    # Optimize the figure for faster rendering
    from utils.dashboard_optimization import optimize_plotly_figure
    fig = optimize_plotly_figure(fig)
    
    return dbc.Card([
        dbc.CardHeader(html.H5("Correlation Analysis")),
        dbc.CardBody([
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': True, 'responsive': True},
            ),
        ]),
    ])
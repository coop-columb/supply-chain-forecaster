"""Model selection components for the supply chain forecaster dashboard."""

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_model_selection(id_prefix, model_types=None, deployed_models=None):
    """
    Create model selection components.
    
    Args:
        id_prefix: Prefix for the component IDs.
        model_types: List of available model types.
        deployed_models: Dictionary of deployed models.
    
    Returns:
        Model selection components.
    """
    if model_types is None:
        model_types = []
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Model Type"),
                dcc.Dropdown(
                    id=f'{id_prefix}-model-type',
                    options=[{'label': model_type, 'value': model_type} for model_type in model_types],
                    placeholder="Select a model type",
                ),
            ], width=6),
            
            dbc.Col([
                html.Label("Model Name"),
                dcc.Input(
                    id=f'{id_prefix}-model-name',
                    type='text',
                    placeholder="Enter a model name",
                    className="form-control",
                ),
            ], width=6),
        ], className="mb-3"),
        
        html.Div(id=f'{id_prefix}-model-params-container'),
    ])


def create_deployed_model_selection(id_prefix, deployed_models=None):
    """
    Create deployed model selection component.
    
    Args:
        id_prefix: Prefix for the component ID.
        deployed_models: Dictionary of deployed models.
    
    Returns:
        Deployed model selection component.
    """
    if deployed_models is None:
        deployed_models = {}
    
    # Create options from deployed models
    options = []
    for model_id, model_info in deployed_models.items():
        model_type = model_info.get("model_type", "Unknown")
        label = f"{model_id} ({model_type})"
        options.append({"label": label, "value": model_id})
    
    return html.Div([
        html.Label("Select Deployed Model"),
        dcc.Dropdown(
            id=f'{id_prefix}-deployed-model',
            options=options,
            placeholder="Select a deployed model",
        ),
    ])


def create_model_parameters(model_type, id_prefix):
    """
    Create model parameter inputs based on model type.
    
    Args:
        model_type: Type of model.
        id_prefix: Prefix for the component IDs.
    
    Returns:
        Model parameter components.
    """
    if model_type == "ProphetModel":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Seasonality Mode"),
                    dcc.Dropdown(
                        id=f'{id_prefix}-param-seasonality-mode',
                        options=[
                            {'label': 'Additive', 'value': 'additive'},
                            {'label': 'Multiplicative', 'value': 'multiplicative'},
                        ],
                        value='additive',
                    ),
                ], width=6),
                
                dbc.Col([
                    html.Label("Changepoint Prior Scale"),
                    dcc.Input(
                        id=f'{id_prefix}-param-changepoint-prior-scale',
                        type='number',
                        value=0.05,
                        min=0.001,
                        max=0.5,
                        step=0.001,
                        className="form-control",
                    ),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Seasonality Prior Scale"),
                    dcc.Input(
                        id=f'{id_prefix}-param-seasonality-prior-scale',
                        type='number',
                        value=10.0,
                        min=0.01,
                        max=100.0,
                        step=0.01,
                        className="form-control",
                    ),
                ], width=6),
                
                dbc.Col([
                    html.Label("Holidays Prior Scale"),
                    dcc.Input(
                        id=f'{id_prefix}-param-holidays-prior-scale',
                        type='number',
                        value=10.0,
                        min=0.01,
                        max=100.0,
                        step=0.01,
                        className="form-control",
                    ),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Seasonality Components"),
                    dbc.Checklist(
                        id=f'{id_prefix}-param-seasonality',
                        options=[
                            {'label': 'Daily', 'value': 'daily'},
                            {'label': 'Weekly', 'value': 'weekly'},
                            {'label': 'Yearly', 'value': 'yearly'},
                        ],
                        value=['weekly', 'yearly'],
                    ),
                ], width=6),
                
                dbc.Col([
                    html.Label("Country Holidays"),
                    dcc.Dropdown(
                        id=f'{id_prefix}-param-country-holidays',
                        options=[
                            {'label': 'None', 'value': 'none'},
                            {'label': 'US', 'value': 'US'},
                            {'label': 'UK', 'value': 'UK'},
                            {'label': 'Canada', 'value': 'Canada'},
                            {'label': 'Germany', 'value': 'Germany'},
                            {'label': 'France', 'value': 'France'},
                        ],
                        value='none',
                    ),
                ], width=6),
            ], className="mb-3"),
        ])
    
    elif model_type == "ARIMAModel":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Auto ARIMA"),
                    dbc.Switch(
                        id=f'{id_prefix}-param-auto-arima',
                        value=False,
                    ),
                ], width=6),
            ], className="mb-3"),
            
            html.Div(
                dbc.Row([
                    dbc.Col([
                        html.Label("Order (p, d, q)"),
                        dbc.InputGroup([
                            dbc.Input(
                                id=f'{id_prefix}-param-p',
                                type='number',
                                value=1,
                                min=0,
                                max=5,
                                step=1,
                                placeholder="p",
                            ),
                            dbc.Input(
                                id=f'{id_prefix}-param-d',
                                type='number',
                                value=1,
                                min=0,
                                max=2,
                                step=1,
                                placeholder="d",
                            ),
                            dbc.Input(
                                id=f'{id_prefix}-param-q',
                                type='number',
                                value=1,
                                min=0,
                                max=5,
                                step=1,
                                placeholder="q",
                            ),
                        ]),
                    ], width=6),
                    
                    dbc.Col([
                        html.Label("Seasonal Order (P, D, Q, s)"),
                        dbc.InputGroup([
                            dbc.Input(
                                id=f'{id_prefix}-param-P',
                                type='number',
                                value=0,
                                min=0,
                                max=2,
                                step=1,
                                placeholder="P",
                            ),
                            dbc.Input(
                                id=f'{id_prefix}-param-D',
                                type='number',
                                value=0,
                                min=0,
                                max=1,
                                step=1,
                                placeholder="D",
                            ),
                            dbc.Input(
                                id=f'{id_prefix}-param-Q',
                                type='number',
                                value=0,
                                min=0,
                                max=2,
                                step=1,
                                placeholder="Q",
                            ),
                            dbc.Input(
                                id=f'{id_prefix}-param-s',
                                type='number',
                                value=12,
                                min=0,
                                step=1,
                                placeholder="s",
                            ),
                        ]),
                    ], width=6),
                ], className="mb-3"),
                id=f'{id_prefix}-manual-arima-params',
            ),
        ])
    
    elif model_type == "XGBoostModel":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Number of Estimators"),
                    dcc.Input(
                        id=f'{id_prefix}-param-n-estimators',
                        type='number',
                        value=100,
                        min=10,
                        max=1000,
                        step=10,
                        className="form-control",
                    ),
                ], width=6),
                
                dbc.Col([
                    html.Label("Learning Rate"),
                    dcc.Input(
                        id=f'{id_prefix}-param-learning-rate',
                        type='number',
                        value=0.1,
                        min=0.001,
                        max=0.5,
                        step=0.001,
                        className="form-control",
                    ),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Max Depth"),
                    dcc.Input(
                        id=f'{id_prefix}-param-max-depth',
                        type='number',
                        value=6,
                        min=1,
                        max=15,
                        step=1,
                        className="form-control",
                    ),
                ], width=6),
                
                dbc.Col([
                    html.Label("Subsample"),
                    dcc.Input(
                        id=f'{id_prefix}-param-subsample',
                        type='number',
                        value=1.0,
                        min=0.1,
                        max=1.0,
                        step=0.1,
                        className="form-control",
                    ),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Column Sample by Tree"),
                    dcc.Input(
                        id=f'{id_prefix}-param-colsample-bytree',
                        type='number',
                        value=1.0,
                        min=0.1,
                        max=1.0,
                        step=0.1,
                        className="form-control",
                    ),
                ], width=6),
                
                dbc.Col([
                    html.Label("Objective"),
                    dcc.Dropdown(
                        id=f'{id_prefix}-param-objective',
                        options=[
                            {'label': 'Squared Error', 'value': 'reg:squarederror'},
                            {'label': 'Absolute Error', 'value': 'reg:absoluteerror'},
                            {'label': 'Poisson', 'value': 'count:poisson'},
                        ],
                        value='reg:squarederror',
                    ),
                ], width=6),
            ], className="mb-3"),
        ])
    
    elif model_type == "IsolationForestDetector" or model_type == "StatisticalDetector" or model_type == "AutoencoderDetector":
        if model_type == "IsolationForestDetector":
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Number of Estimators"),
                        dcc.Input(
                            id=f'{id_prefix}-param-n-estimators',
                            type='number',
                            value=100,
                            min=10,
                            max=1000,
                            step=10,
                            className="form-control",
                        ),
                    ], width=6),
                    
                    dbc.Col([
                        html.Label("Contamination"),
                        dcc.Input(
                            id=f'{id_prefix}-param-contamination',
                            type='number',
                            value=0.1,
                            min=0.001,
                            max=0.5,
                            step=0.001,
                            className="form-control",
                        ),
                    ], width=6),
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Max Samples"),
                        dcc.Dropdown(
                            id=f'{id_prefix}-param-max-samples',
                            options=[
                                {'label': 'auto', 'value': 'auto'},
                                {'label': '10%', 'value': 0.1},
                                {'label': '20%', 'value': 0.2},
                                {'label': '50%', 'value': 0.5},
                                {'label': '100%', 'value': 1.0},
                            ],
                            value='auto',
                        ),
                    ], width=6),
                    
                    dbc.Col([
                        html.Label("Random State"),
                        dcc.Input(
                            id=f'{id_prefix}-param-random-state',
                            type='number',
                            value=42,
                            className="form-control",
                        ),
                    ], width=6),
                ], className="mb-3"),
            ])
        
        elif model_type == "StatisticalDetector":
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Method"),
                        dcc.Dropdown(
                            id=f'{id_prefix}-param-method',
                            options=[
                                {'label': 'Z-Score', 'value': 'zscore'},
                                {'label': 'IQR', 'value': 'iqr'},
                                {'label': 'MAD', 'value': 'mad'},
                            ],
                            value='zscore',
                        ),
                    ], width=6),
                    
                    dbc.Col([
                        html.Label("Threshold"),
                        dcc.Input(
                            id=f'{id_prefix}-param-threshold',
                            type='number',
                            value=3.0,
                            min=1.0,
                            max=10.0,
                            step=0.1,
                            className="form-control",
                        ),
                    ], width=6),
                ], className="mb-3"),
            ])
        
        elif model_type == "AutoencoderDetector":
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Encoding Dimension"),
                        dcc.Input(
                            id=f'{id_prefix}-param-encoding-dim',
                            type='number',
                            value=8,
                            min=2,
                            max=64,
                            step=1,
                            className="form-control",
                        ),
                    ], width=6),
                    
                    dbc.Col([
                        html.Label("Hidden Dimensions"),
                        dcc.Input(
                            id=f'{id_prefix}-param-hidden-dims',
                            type='text',
                            value='32,16',
                            placeholder="Comma-separated values",
                            className="form-control",
                        ),
                    ], width=6),
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Activation"),
                        dcc.Dropdown(
                            id=f'{id_prefix}-param-activation',
                            options=[
                                {'label': 'ReLU', 'value': 'relu'},
                                {'label': 'Sigmoid', 'value': 'sigmoid'},
                                {'label': 'Tanh', 'value': 'tanh'},
                            ],
                            value='relu',
                        ),
                    ], width=6),
                    
                    dbc.Col([
                        html.Label("Contamination"),
                        dcc.Input(
                            id=f'{id_prefix}-param-contamination',
                            type='number',
                            value=0.1,
                            min=0.001,
                            max=0.5,
                            step=0.001,
                            className="form-control",
                        ),
                    ], width=6),
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Epochs"),
                        dcc.Input(
                            id=f'{id_prefix}-param-epochs',
                            type='number',
                            value=50,
                            min=10,
                            max=500,
                            step=10,
                            className="form-control",
                        ),
                    ], width=6),
                    
                    dbc.Col([
                        html.Label("Batch Size"),
                        dcc.Input(
                            id=f'{id_prefix}-param-batch-size',
                            type='number',
                            value=32,
                            min=8,
                            max=256,
                            step=8,
                            className="form-control",
                        ),
                    ], width=6),
                ], className="mb-3"),
            ])
    
    # Default component for unknown model type
    return html.Div([
        html.P(f"No specific parameters for model type: {model_type}"),
    ])

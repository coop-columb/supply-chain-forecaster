"""Model management page for the supply chain forecaster dashboard."""

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html


def create_model_management_layout():
    """
    Create the model management page layout.

    Returns:
        Model management page layout.
    """
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1("Model Management", className="mb-4"),
                            html.P(
                                "Manage your trained models. View model details, deploy models, "
                                "and delete models that are no longer needed.",
                                className="lead mb-4",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Tabs(
                [
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        html.H5("Trained Models")
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            dbc.Button(
                                                                "Refresh Model List",
                                                                id="refresh-trained-models-button",
                                                                color="primary",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                id="trained-models-list"
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(id="trained-model-details"),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Trained Models",
                        tab_id="trained-tab",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        html.H5("Deployed Models")
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            dbc.Button(
                                                                "Refresh Model List",
                                                                id="refresh-deployed-models-button",
                                                                color="primary",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                id="deployed-models-list"
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(id="deployed-model-details"),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Deployed Models",
                        tab_id="deployed-tab",
                    ),
                ],
                id="model-management-tabs",
                active_tab="trained-tab",
            ),
        ]
    )


def create_model_list_table(models_data, id_prefix):
    """
    Create a table of models.

    Args:
        models_data: Dictionary of model information.
        id_prefix: Prefix for the component ID.

    Returns:
        Model list table component.
    """
    if not models_data:
        return html.Div("No models found.")

    # Convert models data to list of dictionaries
    models_list = []
    for model_id, model_info in models_data.items():
        model_type = model_info.get("model_type", "Unknown")
        created_at = model_info.get("created_at", "Unknown")

        models_list.append(
            {
                "model_id": model_id,
                "model_type": model_type,
                "created_at": created_at,
            }
        )

    return dash_table.DataTable(
        id=f"{id_prefix}-table",
        columns=[
            {"name": "Model ID", "id": "model_id"},
            {"name": "Model Type", "id": "model_type"},
            {"name": "Created At", "id": "created_at"},
        ],
        data=models_list,
        style_table={"overflowX": "auto"},
        style_cell={
            "minWidth": "100px",
            "width": "150px",
            "maxWidth": "200px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
        row_selectable="single",
    )


def create_model_details(model_info, is_deployed=False, id_prefix="model"):
    """
    Create model details component.

    Args:
        model_info: Dictionary with model information.
        is_deployed: Whether the model is deployed.
        id_prefix: Prefix for component IDs.

    Returns:
        Model details component.
    """
    if not model_info:
        return html.Div()

    # Extract model details
    model_id = model_info.get("model_name", "Unknown")
    model_type = model_info.get("model_type", "Unknown")
    created_at = model_info.get("created_at", "Unknown")
    parameters = model_info.get("parameters", {})

    # Create parameters table data
    param_data = []
    for param_name, param_value in parameters.items():
        param_data.append(
            {
                "Parameter": param_name,
                "Value": str(param_value),
            }
        )

    # Create metrics table data if available
    metrics_data = []
    train_metrics = model_info.get("train_metrics", {})

    for metric_name, metric_value in train_metrics.items():
        metrics_data.append(
            {
                "Metric": metric_name,
                "Value": str(metric_value),
            }
        )

    return dbc.Card(
        [
            dbc.CardHeader(html.H5(f"Model Details: {model_id}")),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.P(f"Model Type: {model_type}"),
                                    html.P(f"Created At: {created_at}"),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            (
                                                dbc.Button(
                                                    "Deploy Model",
                                                    id=f"{id_prefix}-deploy-button",
                                                    color="success",
                                                    className="me-2",
                                                    disabled=is_deployed,
                                                )
                                                if not is_deployed
                                                else None
                                            ),
                                            dbc.Button(
                                                "Delete Model",
                                                id=f"{id_prefix}-delete-button",
                                                color="danger",
                                            ),
                                        ],
                                        className="d-flex justify-content-end",
                                    ),
                                ],
                                width=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.H6("Model Parameters"),
                    (
                        dash_table.DataTable(
                            data=param_data,
                            columns=[
                                {"name": "Parameter", "id": "Parameter"},
                                {"name": "Value", "id": "Value"},
                            ],
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "minWidth": "100px",
                                "width": "150px",
                                "maxWidth": "200px",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                            },
                        )
                        if param_data
                        else html.P("No parameters available.")
                    ),
                    html.H6("Performance Metrics", className="mt-3"),
                    (
                        dash_table.DataTable(
                            data=metrics_data,
                            columns=[
                                {"name": "Metric", "id": "Metric"},
                                {"name": "Value", "id": "Value"},
                            ],
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "minWidth": "100px",
                                "width": "150px",
                                "maxWidth": "200px",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                            },
                        )
                        if metrics_data
                        else html.P("No metrics available.")
                    ),
                    dcc.Store(id=f"{id_prefix}-selected-model", data=model_id),
                ]
            ),
        ]
    )

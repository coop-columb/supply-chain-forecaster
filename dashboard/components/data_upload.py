"""Data upload component for the supply chain forecaster dashboard."""

import base64
import io

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, dcc, html


def create_upload_component(id_prefix: str):
    """
    Create a file upload component.

    Args:
        id_prefix: Prefix for the component IDs.

    Returns:
        Upload component.
    """
    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Upload Data", className="card-title"),
                        html.P(
                            "Upload CSV file with historical data:",
                            className="card-text",
                        ),
                        dcc.Upload(
                            id=f"{id_prefix}-upload",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Select a File")]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px 0",
                            },
                            multiple=False,
                        ),
                        html.Div(id=f"{id_prefix}-upload-output"),
                        html.Div(id=f"{id_prefix}-data-preview"),
                        dcc.Store(id=f"{id_prefix}-data-store"),
                    ]
                )
            ),
        ]
    )


def parse_contents(contents, filename):
    """
    Parse uploaded file contents.

    Args:
        contents: File contents.
        filename: Name of the file.

    Returns:
        Parsed DataFrame and output component.
    """
    if contents is None:
        return None, html.Div()

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        if "csv" in filename:
            # Parse CSV file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Parse Excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, html.Div(f"Unsupported file type: {filename}")

        # Create data preview
        preview = html.Div(
            [
                html.Hr(),
                html.H6(f"File: {filename}"),
                html.P(f"Rows: {len(df)}, Columns: {len(df.columns)}"),
                dash_table.DataTable(
                    data=df.head(5).to_dict("records"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "minWidth": "100px",
                        "width": "150px",
                        "maxWidth": "200px",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                    },
                    tooltip_data=[
                        {
                            column: {"value": str(value), "type": "markdown"}
                            for column, value in row.items()
                        }
                        for row in df.head(5).to_dict("records")
                    ],
                    tooltip_duration=None,
                ),
                html.Hr(),
            ]
        )

        return df.to_json(date_format="iso", orient="split"), preview

    except Exception as e:
        return None, html.Div(
            [
                html.Hr(),
                html.H6(f"Error processing file: {filename}"),
                html.P(str(e)),
                html.Hr(),
            ]
        )

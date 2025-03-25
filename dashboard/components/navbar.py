"""Navbar component for the supply chain forecaster dashboard."""

import dash_bootstrap_components as dbc
from dash import html


def create_navbar():
    """
    Create the navigation bar.

    Returns:
        Navbar component.
    """
    return dbc.Navbar(
        dbc.Container(
            [
                # Logo and title
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.I(
                                    className="fas fa-chart-line",
                                    style={"font-size": "2rem"},
                                )
                            ),
                            dbc.Col(
                                dbc.NavbarBrand(
                                    "Supply Chain Forecaster", className="ms-2"
                                )
                            ),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                # Navigation links
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Home", href="/")),
                            dbc.NavItem(
                                dbc.NavLink("Data Exploration", href="/exploration")
                            ),
                            dbc.NavItem(
                                dbc.NavLink("Forecasting", href="/forecasting")
                            ),
                            dbc.NavItem(
                                dbc.NavLink("Anomaly Detection", href="/anomaly")
                            ),
                            dbc.NavItem(
                                dbc.NavLink("Model Management", href="/models")
                            ),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ]
        ),
        color="primary",
        dark=True,
        className="mb-4",
    )

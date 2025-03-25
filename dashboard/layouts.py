"""
Dashboard layout components for the Supply Chain Forecaster.

This module provides reusable layout components for the dashboard pages.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.components.navbar import create_navbar


def create_page_layout(title, children=None, fluid=True):
    """
    Create a standard page layout with a navbar and container.
    
    Args:
        title (str): The title of the page.
        children (list, optional): The child components to render in the container.
        fluid (bool, optional): Whether to use a fluid container. Defaults to True.
    
    Returns:
        dash.html.Div: The page layout component.
    """
    if children is None:
        children = []
    
    return html.Div([
        create_navbar(),
        dbc.Container(
            [
                html.H1(title, className="mt-4 mb-4"),
                html.Hr(),
                *children
            ],
            fluid=fluid,
            className="py-3"
        )
    ])


def create_card(title, children, card_id=None, className=None):
    """
    Create a styled card component.
    
    Args:
        title (str): The title of the card.
        children (list): The child components to render in the card body.
        card_id (str, optional): The ID of the card. Defaults to None.
        className (str, optional): Additional CSS classes for the card. Defaults to None.
    
    Returns:
        dash_bootstrap_components.Card: The card component.
    """
    return dbc.Card(
        [
            dbc.CardHeader(html.H4(title)),
            dbc.CardBody(children)
        ],
        id=card_id,
        className=f"mb-4 shadow-sm {className or ''}"
    )


def create_form_group(label, component, help_text=None):
    """
    Create a standard form group with a label and help text.
    
    Args:
        label (str): The label for the form component.
        component: The form component to render.
        help_text (str, optional): Optional help text to display. Defaults to None.
    
    Returns:
        dash_bootstrap_components.FormGroup: The form group component.
    """
    elements = [dbc.Label(label)]
    elements.append(component)
    
    if help_text:
        elements.append(dbc.FormText(help_text))
    
    return dbc.FormGroup(elements, className="mb-3")


def create_tabs(tab_items, tab_id):
    """
    Create a standard set of tabs.
    
    Args:
        tab_items (list): List of dicts with 'label' and 'content' keys.
        tab_id (str): The ID for the tabs component.
    
    Returns:
        list: A list containing the tabs and the tab content div.
    """
    tabs = dbc.Tabs(
        [
            dbc.Tab(label=item['label'], tab_id=f"{tab_id}-{i}") 
            for i, item in enumerate(tab_items)
        ],
        id=tab_id,
        active_tab=f"{tab_id}-0",
        className="mb-3"
    )
    
    tab_content = html.Div(id=f"{tab_id}-content")
    
    return [tabs, tab_content]


def create_alert(message, color="primary", dismissable=True, is_open=True):
    """
    Create a standard alert component.
    
    Args:
        message (str): The message to display in the alert.
        color (str, optional): The color of the alert. Defaults to "primary".
        dismissable (bool, optional): Whether the alert can be dismissed. Defaults to True.
        is_open (bool, optional): Whether the alert is initially open. Defaults to True.
    
    Returns:
        dash_bootstrap_components.Alert: The alert component.
    """
    return dbc.Alert(
        message,
        color=color,
        dismissable=dismissable,
        is_open=is_open,
        className="mb-3"
    )


def create_loading_container(children, id_prefix, type="circle"):
    """
    Create a container with a loading indicator.
    
    Args:
        children: The child components to render in the container.
        id_prefix (str): Prefix for the component ID.
        type (str, optional): The type of loading indicator. Defaults to "circle".
    
    Returns:
        dash.html.Div: The container with loading indicator.
    """
    return html.Div(
        [
            dcc.Loading(
                children=html.Div(children, id=f"{id_prefix}-content"),
                id=f"{id_prefix}-loading",
                type=type
            )
        ]
    )


def create_results_container(id_prefix):
    """
    Create a standard container for displaying model results.
    
    Args:
        id_prefix (str): Prefix for the component IDs.
    
    Returns:
        dash.html.Div: The results container.
    """
    return html.Div(
        [
            # Initial state: hidden
            html.Div(id=f"{id_prefix}-results", style={"display": "none"}, children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Results", className="mb-3"),
                        html.Div(id=f"{id_prefix}-metrics"),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id=f"{id_prefix}-chart"),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id=f"{id_prefix}-table"),
                    ])
                ])
            ]),
            # Error container
            html.Div(id=f"{id_prefix}-error", style={"display": "none"})
        ]
    )
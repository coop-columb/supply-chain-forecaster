# Dashboard and Documentation Screenshots

This directory contains screenshots for the Supply Chain Forecaster documentation.

## Directory Structure

```
screenshots/
├── dashboard/              # Dashboard interface screenshots
│   ├── home/               # Home page screenshots
│   ├── data_exploration/   # Data exploration page screenshots
│   ├── forecasting/        # Forecasting page screenshots
│   ├── anomaly_detection/  # Anomaly detection page screenshots
│   └── model_management/   # Model management page screenshots
├── api/                    # API-related screenshots (Swagger UI, etc.)
├── common_workflows/       # Screenshots illustrating common workflows
└── quickstart/             # Screenshots for the quickstart guide
```

## Naming Convention

Please follow these naming conventions for screenshot files:

1. Use descriptive names in lowercase with hyphens between words
2. Include a sequence number if multiple screenshots are related
3. Use .png format for all screenshots

Examples:
- `home-dashboard-overview.png`
- `forecasting-model-training-01-upload.png`
- `forecasting-model-training-02-configure.png`
- `forecasting-model-training-03-results.png`

## Screenshot Guidelines

When taking screenshots for documentation:

1. Use a consistent window size (1280x800 recommended)
2. Ensure the application is in a clean state with sample/demo data
3. Highlight relevant UI elements if needed
4. Crop screenshots to focus on relevant content
5. Optimize images for web (compress without significant quality loss)

## Adding New Screenshots

To add new screenshots to the documentation:

1. Take the screenshot following the guidelines above
2. Save it in the appropriate subdirectory
3. Reference it in the markdown documentation using relative paths:

```markdown
![Dashboard Home Page](../screenshots/dashboard/home/home-dashboard-overview.png)
```

## Required Screenshots

The following screenshots should be created for the documentation:

### Dashboard
- Home page overview
- Data upload interface
- Data visualization examples
- Model training interface
- Forecast results visualization
- Anomaly detection results
- Model management interface

### API
- Swagger UI overview
- Example API responses

### Common Workflows
- End-to-end forecasting workflow
- Anomaly detection workflow
- Model evaluation workflow
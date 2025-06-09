# sentinel_revamp
#  sentinel_project_root/README.md

# Sentinel Public Health Co-Pilot

**Version:** 6.0.0 (Platinum Standard - Public Health Mission)  
**Status:** Active Development & Demonstration

---

## 🚀 Mission Statement

The Sentinel Public Health Co-Pilot is a platinum-grade, edge-first health intelligence ecosystem. Its mission is to empower health workers, program managers, and epidemiologists in low- and middle-income countries (LMICs) with advanced diagnostics, predictive analytics, and actionable data visualization. By focusing on high-impact diseases like Tuberculosis, HIV, Malaria, and NTDs, Sentinel aims to enable affordable, scalable screening and surveillance to advance health equity for over one billion people by 2035.

---

## ✨ Key Features

The Sentinel platform provides a suite of specialized dashboards, each tailored to a specific role in the public health ecosystem:

*   **System Overview (`app.py`):** The central landing page providing an introduction to the Sentinel system and dynamically linking to all available dashboards.
*   **🧑‍⚕️ Field Operations Dashboard:** A command center for supervisors to monitor the activity and performance of frontline health workers, identify high-risk patients via a predictive AI model, and track emerging local trends.
*   **🏥 Clinic Operations Dashboard:** A console for clinic managers to oversee laboratory performance with statistical KPIs (turnaround time, rejection rates), and manage the supply chain with predictive, `Prophet`-based consumption forecasting.
*   **🗺️ District Command Center:** A strategic dashboard for District Health Officers (DHOs) to monitor programmatic KPIs against public health targets, perform geospatial analysis of disease burden, and plan targeted interventions for zones that need the most support.
*   **📊 Population Research Console:** A powerful toolkit for epidemiologists to conduct deep-dive cohort analysis, explore demographic and socioeconomic risk factors, and investigate disease comorbidities with interactive visualizations.

---

## 🛠️ Technology Stack

*   **Backend & Analytics:** Python 3.9+
*   **Web Framework:** Streamlit
*   **Data Manipulation:** Pandas
*   **Predictive Modeling:** Scikit-learn
*   **Time-Series Forecasting:** Prophet (by Meta)
*   **Statistical Analysis:** SciPy, NumPy
*   **Configuration:** Pydantic
*   **Visualization:** Plotly

---

## ⚙️ Getting Started

Follow these steps to set up and run the Sentinel application on your local machine.

### 1. Prerequisites

*   Python (version 3.9 or newer recommended).
*   A C++ compiler and Python development headers (required by some dependencies).
    *   **On Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install -y build-essential python3-dev`
    *   **On macOS (with Homebrew):** `brew install libomp`
    *   **On RHEL/CentOS:** `sudo yum groupinstall -y 'Development Tools'`

### 2. Environment Setup

This project uses a bash script to create an isolated Python virtual environment and install all dependencies.

```bash
# Make the setup script executable
chmod +x scripts/setup.sh

# Run the setup script
./scripts/setup.sh
[UI Layer: Streamlit Pages] -> [Visualization Layer] -> [Analytics Layer] -> [Data Layer]
-----------------------------------------------------------------------------------------
          |                           |                        |                   |
    pages/*.py               visualization/         analytics/           data_processing/
(Dashboards)                    |                        |                      |
     |                     plots.py                 prediction.py        loaders.py
     |                     ui_elements.py           forecasting.py       enrichment.py
     |                     themes.py                aggregation.py       pipeline.py
     |                                                                    helpers.py
     |
     +---- [Configuration] ----------------------------> config/settings.py
     +---- [Data & Assets] ----------------------------> data_sources/*.csv
                                                         assets/*.png, *.css
                                                         ml_models/*.joblib
config: Centralized Pydantic models manage all application settings.
data_processing: Handles loading, cleaning, and enriching all raw data.
analytics: Contains all predictive models, forecasting engines, and statistical aggregation logic.
visualization: A dedicated package for creating themed plots and reusable UI components.
pages: Each file is a self-contained Streamlit dashboard tailored to a specific user persona.
🧠 Note on the Machine Learning Model
The project includes a placeholder pre-trained risk model (ml_models/sentinel_risk_model_v1.joblib) to ensure the application is fully runnable out-of-the-box. In a real-world scenario, this would be replaced with a model trained on actual clinical data. The analytics/prediction.py module is designed to load any scikit-learn compatible model specified in config/settings.py.

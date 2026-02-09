# Brent Oil Price Analysis Project

## Change Point Analysis and Statistical Modeling of Time Series Data

### 📊 Project Overview

This project analyzes how major political and economic events affect Brent crude oil prices using Bayesian change point detection and advanced statistical modeling. The analysis helps investors, policymakers, and energy companies understand market dynamics, manage risks, and make data-driven decisions.

**Organization**: Birhan Energies  
**Project Duration**: February 4-10, 2026  
**Team**: Kerod (Lead), Filimon, Mahbubah

---

## 🎯 Objectives

1. **Identify Key Events**: Detect significant geopolitical and economic events impacting Brent oil prices over the past decade
2. **Quantify Impacts**: Measure the magnitude of price changes associated with specific events using statistical methods
3. **Provide Actionable Insights**: Deliver clear, data-driven recommendations for investment strategies, policy development, and operational planning
4. **Build Interactive Dashboard**: Create a user-friendly tool for stakeholders to explore price dynamics and event correlations

---

## 📁 Project Structure
```
brent-oil-analysis/
│
├── data/ # Data directory
│ ├── raw/ # Raw data files
│ │ └── brent_prices.csv # Historical Brent prices (1987-2022)
│ ├── processed/ # Cleaned and processed data
│ │ ├── cleaned_prices.csv # Processed price data
│ │ └── events_dataset.csv # Compiled event database
│ └── external/ # External data sources
│ └── macroeconomic_indicators.csv
│
├── notebooks/ # Jupyter notebooks for analysis
│ ├── 01_data_exploration.ipynb # EDA and data visualization
│ ├── 02_change_point_analysis.ipynb # Bayesian change point modeling
│ └── 03_advanced_modeling.ipynb # Advanced statistical models
│
├── src/ # Source code modules
│ ├── models/ # Statistical models
│ │ ├── bayesian_change_point.py # Bayesian change point models
│ │ └── advanced_models.py # VAR, Markov switching, GMM models
│ ├── utils/ # Utility functions
│ │ ├── data_loader.py # Data loading utilities
│ │ ├── data_preparation.py # Data preparation functions
│ │ └── visualization.py # Plotting and visualization
│ └── dashboard/ # Dashboard application
│ ├── backend/ # Flask API backend
│ └── frontend/ # React frontend
│
├── reports/ # Analysis reports and documentation
│ ├── task1_analysis_workflow.pdf # Task 1 deliverables
│ ├── task2_results_interpretation.pdf # Task 2 analysis
│ └── final_report.pdf # Comprehensive final report
│
├── dashboard/ # Dashboard deployment
│ ├── app.py # Main Flask application
│ ├── requirements.txt # Python dependencies
│ └── README.md # Dashboard setup instructions
│
├── results/ # Analysis outputs
│ ├── change_point_impacts.csv # Quantified change point impacts
│ ├── event_correlations.csv # Event-price correlations
│ └── event_impact_analysis.csv # Event study results
│
├── requirements.txt # Project dependencies
├── .gitignore # Git ignore file
└── README.md # This file
```


---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 14+ (for dashboard frontend)
- Git

### Installation

1. **Clone the repository**
```
git clone https://github.com/your-username/brent-oil-analysis.git
cd brent-oil-analysis
```

2. **Set up Python environment**
# Create virtual environment
```python -m venv venv```

# Activate virtual environment
# On Windows:
```venv\Scripts\activate```
# On macOS/Linux:
```source venv/bin/activate```

# Install dependencies
```pip install -r requirements.txt```
3. **Prepare data**
# Run data preparation
```python src/utils/data_preparation.py```

# The main dataset should be placed at:
```# data/raw/brent_prices.csv```
## Running the Analysis
# Option 1: Jupyter Notebooks (Recommended for Exploration)

1. **Start Jupyter**
```jupyter notebook```
2. **Run notebooks in order:**

       ``` notebooks/01_data_exploration.ipynb - Data exploration and visualization```

       ``` notebooks/02_change_point_analysis.ipynb - Bayesian change point detection```

       ``` notebooks/03_advanced_modeling.ipynb - Advanced statistical models```

# Option 2: Command Line
# Run the complete analysis pipeline
```python -m src.models.bayesian_change_point```
## Running the Dashboard
# Backend (Flask API)
```
cd dashboard/backend
pip install -r requirements.txt
python app.py
# Server runs at http://localhost:5000
```
#  Frontend (React)
```
cd dashboard/frontend
npm install
npm start
# Dashboard opens at http://localhost:3000
```
 Dataset Description
Primary Dataset: Brent Oil Prices

    Source: Historical daily Brent crude oil prices

    Period: May 20, 1987 to September 30, 2022

    Frequency: Daily

    Format: CSV with columns:

        Date: Date in DD-Mon-YY format (e.g., 20-May-87)

        Price: Brent crude price in USD per barrel

Event Dataset

Manually compiled database of 20+ major geopolitical and economic events:

    Categories: Conflicts, OPEC decisions, economic crises, sanctions, natural disasters

    Fields: Event name, date, category, description, expected impact

    Sources: Historical news, economic reports, policy announcements

🔬 Methodology
1. Bayesian Change Point Detection

    Model: Bayesian multiple change point model using PyMC

    Parameters: Changing means and variances across regimes

    Inference: Markov Chain Monte Carlo (MCMC) sampling

    Output: Posterior distributions of change points and regime parameters

2. Advanced Statistical Models

    Vector Autoregression (VAR): Multi-variable time series analysis

    Markov Switching Models: Regime detection with probabilistic transitions

    Gaussian Mixture Models: Clustering-based regime identification

    GARCH Models: Volatility modeling and forecasting

3. Event Study Analysis

    Window Analysis: Pre- and post-event price changes

    Statistical Testing: Significance of event impacts

    Correlation Analysis: Temporal alignment of events with change points

📈 Key Findings
Detected Change Points

    March 2020: COVID-19 pandemic onset

        Price drop: 48.2% ($61.25 → $31.72)

        Associated with: Global lockdowns, demand collapse

    June 2014: OPEC production decision

        Price drop: 34.6% ($108.92 → $71.25)

        Associated with: Shale boom, OPEC maintains production

    February 2022: Russia-Ukraine conflict

        Price increase: 32.7% ($78.92 → $104.78)

        Associated with: Sanctions, supply disruptions

Event Impact Analysis

    Highest Positive Impact: Russia-Ukraine War (2022)

    Highest Negative Impact: COVID-19 Pandemic (2020)

    Most Frequent Trigger: OPEC policy decisions

    Longest-lasting Effects: Geopolitical conflicts

🎮 Dashboard Features
Interactive Visualizations

    Price Timeline: Historical prices with event markers

    Regime Analysis: Visual identification of market regimes

    Event Impact Explorer: Detailed analysis of specific events

    Volatility Charts: Rolling volatility and risk metrics

Analysis Tools

    Date Range Selector: Custom time period analysis

    Event Filter: Filter by event type and impact

    Forecast Module: Short-term price predictions

    Export Functionality: Download charts and data

User-Specific Views

    Investor View: Risk metrics, opportunity identification

    Policy View: Market stability indicators, policy impacts

    Corporate View: Cost forecasting, supply chain insights

🛠️ Technical Implementation
Core Technologies

    Python 3.8+: Primary analysis language

    PyMC 5.0+: Bayesian modeling and inference

    Statsmodels: Traditional time series analysis

    Flask: Backend API development

    React: Frontend dashboard

    D3.js/Recharts: Data visualization
    Model Performance

    MCMC Convergence: R-hat values < 1.01 for all parameters

    Computational Efficiency: 30 minutes for full analysis (8-core CPU)

    Accuracy: 85% correlation between detected change points and actual events

📋 Deliverables
Task 1: Foundation and Planning

    Data analysis workflow document

    Structured event dataset (20+ events)

    Assumptions and limitations documentation

    Communication plan

Task 2: Change Point Modeling

    Bayesian change point detection implementation

    Quantified impact analysis

    Event correlation results

    Comprehensive Jupyter notebook

Task 3: Interactive Dashboard

    Flask backend with REST API

    React frontend with interactive visualizations

    Complete deployment setup

    User documentation

Additional Outputs

    Executive summary report

    Technical documentation

    Code repository with MIT license

    Docker container for easy deployment

🎯 Stakeholder Recommendations
For Investors

    Risk Management: Use change point detection for stop-loss triggers

    Opportunity Identification: Focus investments around detected regimes

    Portfolio Diversification: Allocate based on market volatility regimes

For Policymakers

    Market Monitoring: Implement early warning systems for price shocks

    Strategic Reserves: Time interventions based on regime analysis

    International Coordination: Focus on periods of high volatility

For Energy Companies

    Contract Timing: Align contracts with favorable price regimes

    Cost Management: Hedge during high volatility periods

    Supply Chain Planning: Adjust inventory based on price forecasts

⚠️ Limitations and Assumptions
Statistical Limitations

    Correlation ≠ Causation: Temporal alignment doesn't prove causation

    Multiple Events: Challenges isolating individual event impacts

    External Factors: Unobserved variables may influence results

    Model Simplicity: Basic models may miss complex dynamics

Data Limitations

    Frequency: Daily data may miss intraday volatility

    Completeness: Some historical events may be missing

    Quality: Potential data errors in historical records

Recommendations for Improvement

    Higher Frequency Data: Use intraday data for finer analysis

    Alternative Data: Incorporate news sentiment, satellite imagery

    Causal Methods: Implement difference-in-differences, synthetic controls

    Ensemble Models: Combine multiple modeling approaches

🔮 Future Work
Short-term Enhancements

    Real-time Updates: Live data integration

    Automated Event Detection: NLP for news analysis

    Mobile Application: Stakeholder access on-the-go

Medium-term Improvements

    Machine Learning Integration: LSTM forecasts, feature importance

    Alternative Data Sources: Shipping data, inventory levels

    Global Expansion: Include WTI, Dubai, other benchmarks

Long-term Vision

    Predictive Analytics: AI-driven price forecasting

    Scenario Simulation: What-if analysis for policy decisions

    Blockchain Integration: Transparent market data tracking

👥 Team and Acknowledgments
Core Team

    Kerod: Project Lead, Bayesian Modeling

    Filimon: Data Engineering, Dashboard Development

    Mahbubah: Statistical Analysis, Visualization

Acknowledgments

    10 Academy for project framework and guidance

    Bloomberg and EIA for data sources

    Open source community for analytical tools

📚 References
Academic Papers

    Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection

    Kim, C. J., & Nelson, C. R. (1999). State-Space Models with Regime Switching

    Sims, C. A. (1980). Macroeconomics and Reality

Technical Documentation

    PyMC Documentation: https://www.pymc.io

    Statsmodels User Guide: https://www.statsmodels.org

    React Documentation: https://reactjs.org

Data Sources

    U.S. Energy Information Administration (EIA)

    Bloomberg Terminal

    OPEC Monthly Oil Market Reports


# zetheta-Project1B
Its about working on Data Analyst role.
@'
# Ztheta WorkBridge Project 1B
## Data Analyst - Convexity Sensitivity AI Agent

A bond portfolio risk analytics and AI-assisted sensitivity platform built with Python, PCA, Monte Carlo simulation, machine learning, key-rate duration, convexity analysis, and an interactive Streamlit Bond Risk Lab dashboard.

## Project Objective

Analyze how a fixed-income portfolio responds to changing interest-rate environments and identify the major sources of duration, convexity, yield-curve, and key-rate risk.

The project combines:

- Bond portfolio analytics
- Yield-curve history
- PCA-based yield-curve factor modeling
- Monte Carlo scenario generation
- Duration and convexity P&L estimation
- Key-rate risk attribution
- Random Forest classification
- Neural Network classification
- XGBoost classification
- VaR and Expected Shortfall
- AI-style risk assessment and recommendations
- Interactive Streamlit dashboard
- Automated regression tests

Diagram:  <img width="878" height="1029" alt="image" src="https://github.com/user-attachments/assets/39237bd4-4415-43ba-a85c-c31fd90e61ab" />


## Architecture

```text
data/raw/
    |
    +-- bond_portfolio_data.csv
    +-- yield_curve_history.csv
    +-- monte_carlo_scenarios.csv
    |
    v
src/bond_risk_lab/data/loaders.py
    |
    v
Yield Curve Analytics
    |
    +-- PCA factors
    +-- factor loadings
    +-- explained variance
    |
    v
Monte Carlo Curve Scenarios
    |
    +-- PCA factor simulation
    +-- curve shock reconstruction
    |
    v
Scenario Pricing
    |
    +-- Duration P&L
    +-- Convexity P&L
    +-- Total P&L
    |
    +----------------------+
    |                      |
    v                      v
Risk Attribution       Machine Learning
    |                      |
    +-- Key-rate risk      +-- Random Forest
    +-- Duration           +-- Neural Network
    +-- Convexity          +-- XGBoost
    |                      |
    +----------+-----------+
               |
               v
       AI Risk Assessment
               |
               v
      Streamlit Dashboard

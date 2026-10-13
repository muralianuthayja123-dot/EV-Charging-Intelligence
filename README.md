# ⚡ EV Charging Intelligence Dashboard

### Machine Learning-Based EV Charging Demand and Infrastructure Capacity Analysis

An interactive data analytics dashboard built using Python, Streamlit, and Machine Learning to analyze electric vehicle charging demand, forecast charging sessions, evaluate infrastructure stress, and explore future charging capacity scenarios.

🔗 **Live Dashboard:** [EV Charging Intelligence](https://ev-charging-intelligence-xmvmnvvzxyujxaqnxorjid.streamlit.app/)

---

## 📌 Project Overview

The increasing adoption of electric vehicles creates a need for effective charging infrastructure planning. Understanding charging patterns, identifying high-demand locations, and evaluating infrastructure capacity can help improve resource allocation and support better planning decisions.

This project combines Exploratory Data Analysis (EDA), machine learning, data visualization, and interactive dashboards to analyze EV charging activity using ACN-Data-based analysis.

## 🎯 Project Objectives

- Analyze hourly EV charging demand across charging stations.
- Identify peak demand periods and high-activity charging sites.
- Forecast charging sessions using a Gradient Boosting Regression model.
- Compare actual and predicted charging demand.
- Analyze infrastructure occupancy and potential stress.
- Explore charging capacity scenarios and overload exposure.
- Generate infrastructure recommendations based on the available analysis.

## 🛠️ Technologies Used

- **Python** — Data processing and application logic
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computation
- **Matplotlib** — Data visualization
- **Scikit-learn** — Machine learning and Gradient Boosting Regression
- **Streamlit** — Interactive dashboard development
- **ACN-Data** — Basis for the EV charging analysis

## 📊 Dashboard Modules

### 1. System Overview

Provides a summary of EV charging activity through key performance indicators, including total charging sessions, average hourly demand, peak hourly demand, and the busiest charging site.

It also visualizes charging demand trends and site-wise charging activity.
<img width="1703" height="852" alt="image" src="https://github.com/user-attachments/assets/0455a984-1646-40d7-a6d4-8f928847b4e6" />


### 2. EV Charging Demand Analysis

Analyzes charging patterns at individual sites.

**Features:**
- Site selection
- Hourly charging demand analysis
- Demand by day of the week
- Charging energy analysis
- Demand statistics

<img width="1550" height="822" alt="image" src="https://github.com/user-attachments/assets/ef01ec30-4efb-4e78-9afe-99bf18f3ad01" />


### 3. AI-Based Demand Forecast

Uses a Gradient Boosting Regression model to estimate charging-session demand using historical demand patterns and engineered time-based features.

**Features:**
- Charging-site selection
- Predicted versus actual demand
- Latest prediction difference
- Actual-versus-predicted visualization
- Recent prediction data
<img width="1543" height="838" alt="image" src="https://github.com/user-attachments/assets/bff3fb8b-43f9-45b8-809d-97070b08ed02" />


### 4. Infrastructure Stress Analysis

Evaluates charging-site occupancy to help identify infrastructure that may require further investigation.

**Features:**
- Average occupancy
- 95th-percentile occupancy
- Maximum occupancy
- Site-level stress classification
- Comparative occupancy visualization

<img width="1477" height="783" alt="image" src="https://github.com/user-attachments/assets/7f13c59a-1a8a-4ebb-b69a-43f15befc9c0" />


### 5. Charging Capacity Simulator

Explores the relationship between charging capacity and overload exposure.

**Features:**
- Site-specific capacity analysis
- Capacity versus overload exposure
- Marginal benefit of additional capacity
- Capacity scenario comparison

<img width="1531" height="782" alt="image" src="https://github.com/user-attachments/assets/5cdc5a87-5bf7-4111-a673-a6025ae634af" />


### 6. Infrastructure Recommendations

Displays site-level recommendations and future capacity requirements based on the available scenario analysis.

<img width="1495" height="811" alt="image" src="https://github.com/user-attachments/assets/ef03a9c9-e7c5-4804-97bb-f79aca23fbed" />


## 🧠 Machine Learning Methodology

The dashboard uses a **Gradient Boosting Regressor** to estimate EV charging-session demand.

### Feature Engineering

The model uses the following features:

- Hour of the day
- Day of the week
- Weekend indicator
- Previous-hour demand (lag 1, lag 2, and lag 3)
- Demand from 24 hours earlier
- Demand from 168 hours earlier
- Three-hour rolling average
- Twenty-four-hour rolling average

### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Gradient Boosting Regressor |
| Number of estimators | 100 |
| Learning rate | 0.05 |
| Maximum depth | 5 |
| Random state | 42 |

The training procedure uses a chronological 80% cutoff. Negative predictions are clipped to zero because charging-session demand cannot be negative.

## 📈 Model Evaluation

Previously recorded model evaluation results:

| Evaluation Metric | Value |
|---|---:|
| Mean Absolute Error (MAE) | 0.5173 |
| Root Mean Squared Error (RMSE) | 1.2265 |
| R² Score | 0.2302 |

**Interpretation:** The recorded R² score indicates limited explanatory performance on the evaluated data. These results should be independently verified before being treated as final model performance.

## 📁 Project Structure

```text
ev-charging-intelligence/
│
├── app.py
├── requirements.txt
├── caltech_hourly_demand.csv
├── final_site_analysis.csv
├── future_capacity_scenarios.csv
├── site_stress_analysis.csv
└── capacity_scenario_analysis.csv
```

## 🚀 How to Run the Project Locally

### Prerequisites

- Python
- pip
- Project source files and required datasets

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ev-charging-intelligence.git
cd ev-charging-intelligence
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the dashboard.

## ⚠️ Limitations

- The recorded model evaluation indicates that predictive performance requires improvement.
- Forecasts depend on the quality and availability of historical charging data.
- Lag and rolling features require sufficient historical observations.
- Infrastructure recommendations depend on the assumptions and scenarios represented in the input datasets.
- Further validation is required before using the results for real-world infrastructure decisions.

## 🔮 Future Enhancements

- Compare additional machine learning and time-series forecasting algorithms.
- Improve predictive performance through systematic time-series validation.
- Introduce interactive date-range filtering.
- Compare forecasting performance across charging sites.
- Improve future-demand forecasting and infrastructure planning scenarios.
- Enhance visualizations and dashboard usability.

## 🎓 Skills Demonstrated

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Regression modelling
- Model evaluation
- Data visualization
- Interactive dashboard development
- Infrastructure scenario analysis
- Python application deployment

## 🔗 Project Links

- **Live Dashboard:** [EV Charging Intelligence](https://ev-charging-intelligence-xmvmnvvzxyujxaqnxorjid.streamlit.app/)
- **GitHub Repository:** Available on this repository's GitHub page.

---

*Developed as a data analytics and machine learning project focused on EV charging demand and infrastructure capacity analysis.*

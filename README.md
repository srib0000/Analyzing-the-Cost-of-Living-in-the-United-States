Analyzing-the-Cost-of-Living-in-the-United-States

This project applies machine learning techniques to predict the total cost of living across U.S. regions based on multiple socio-economic factors such as housing, food, transportation, taxes, healthcare, and childcare. It includes a user-friendly web application built with Streamlit for real-time predictions.

Overview

Accurate prediction of cost of living is essential for:
- Individuals making relocation decisions
- Businesses determining compensation
- Policymakers addressing regional disparities

By leveraging a dataset of over 31,000 records, this project identifies key cost drivers and applies robust predictive models under standard, noisy, and non-linear data conditions.

Features

- **Interactive Web App**: Streamlit interface for real-time cost prediction
- **Data Preprocessing**: Includes missing value imputation, feature scaling, and encoding
- **Feature Selection**: Uses correlation analysis and Random Forest importance
- **Model Comparison**: Evaluates Linear Regression, Ridge Regression, Decision Trees, Random Forest, and Gradient Boosting
- **Robust Prediction Engine**: Gradient Boosting selected as the final model for its superior accuracy and robustness

Machine Learning Models

| Model              | R² Score | MSE               | Notes                                 |
|-------------------|----------|-------------------|----------------------------------------|
| Linear Regression  | 1.00     | 0.00              | Overfits clean data, poor on noise     |
| Ridge Regression   | 1.00     | 0.17              | Good for clean, linear data            |
| Decision Trees     | 0.95     | 23,802,981.50     | High variance, prone to overfitting    |
| Random Forest      | 1.00     | 971,309.96        | Strong but less robust than GBM        |
| Gradient Boosting  | 1.00     | ~2M (robust)      | Best accuracy, handles non-linearity   |

System Architecture

- **Backend**: Python, Scikit-learn, Pandas, NumPy
- **Frontend**: Streamlit for user interaction and visualization
- **Deployment**: Local/Cloud hosting of Streamlit app

Getting Started

1. Clone the repo:
   git clone https://github.com/yourusername/cost-of-living-predictor.git
   cd cost-of-living-predictor

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run app.py

Dataset:

Source: Kaggle - US Cost of Living Dataset
Attributes: Region, isMetro, housing_cost, food_cost, transportation_cost, childcare_cost, healthcare_cost, taxes, etc.

Results:

Top Cost Drivers: food_cost, housing_cost, childcare_cost
Best Model: Gradient Boosting for its robustness and accuracy across all scenarios
Usability: Real-time predictions with interactive visualization and input features

Future Work:

Integrate real-time data (e.g., inflation, housing trends)
Add geospatial visualization with maps
Incorporate explainable AI (XAI) for transparency
Explore deep learning models (e.g., RNN, Transformer)

Contributors:

Sasank Sribhashyam (sasank.sribhashyam-1@ou.edu)
Hima Deepika Mannam (hima.deepika.mannam-1@ou.edu)
Venkat Tarun Adda (adda0003@ou.edu)

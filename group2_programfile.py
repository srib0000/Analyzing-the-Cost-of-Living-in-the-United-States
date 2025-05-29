import streamlit as st
import numpy as np
import joblib

# Define the Gradient Boosting Regressor class
class GradientBoostingRegressorScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        predictions = np.full(y.shape, self.initial_prediction)
        for _ in range(self.n_estimators):
            residuals = y - predictions
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            predictions += self.learning_rate * tree.predict(X)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Load the trained model and scaler
model = joblib.load("cost_of_living_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "introduction"  # Default page is the introduction

# Navigation Logic
if st.session_state.page == "introduction":
    # Introduction Page
    st.title("Welcome to the Cost of Living Prediction Tool")
    st.write("""
    This tool helps you estimate the cost of living based on key expenses such as:
    - Food costs
    - Childcare costs
    - Healthcare costs
    - Taxes
    - Other necessary expenses
    
    Use this tool to plan your finances or compare costs across different regions. Click the button below to start.
    """)
    
    # Button to navigate to the Prediction Form Page
    if st.button("Go to Prediction Form"):
        st.session_state.page = "form"  # Update session state to form
        st.rerun()  # Refresh the app to show the new page

elif st.session_state.page == "form":
    # Prediction Form Page
    st.title("Cost of Living Prediction Tool")
    st.write("Fill in the form below to predict the estimated cost of living.")

    # Form for user inputs
    with st.form(key="unique_prediction_form"):
        st.header("Cost Details")
        food_cost = st.number_input("Food Cost (e.g., 3000)", min_value=0, max_value=30000, value=5000, step=500)
        other_necessities = st.number_input("Other Necessities Cost (e.g., 5000)", min_value=0, max_value=30000, value=7000, step=500)
        childcare = st.number_input("Childcare Cost (e.g., 6000)", min_value=0, max_value=30000, value=10000, step=500)
        taxes = st.number_input("Taxes (e.g., 7000)", min_value=0, max_value=30000, value=8000, step=500)
        healthcare = st.number_input("Healthcare Cost (e.g., 5000)", min_value=0, max_value=30000, value=6000, step=500)
        
        # Submit button
        submit_button = st.form_submit_button("Submit")

    # Process input and predict when the form is submitted
    if submit_button:
        input_data = np.array([[food_cost, other_necessities, childcare, taxes, healthcare]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Display results
        st.success(f"Estimated Total Cost of Living: ${prediction:.2f}")
        
        # Button to go back to the Introduction Page
        if st.button("Try other values.."):
            st.session_state.page = "introduction"  # Update session state to introduction
            st.rerun()  # Refresh the app to show the introduction page

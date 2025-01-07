import streamlit as st
from scripts.options import args_parser
from datasets.data_preprocess import data_preprocess
from src.xgboost_scratch import XGBoostModel
from scripts.squared_error_objective import SquaredErrorObjective

print("App started")

parser = args_parser()
_args, unknown = parser.parse_known_args()
(x_train, y_train), (x_test, y_test) = data_preprocess(_args)

# Title of the web app
st.title("XGBoost Model (from scratch) Prediction App")

# Description of the app
st.write("""
This is a app to predict using a xgboost machine learning model from scratch.
""")

# Input from user
num_boost_round = st.number_input("Do you want to change the number of boosting rounds? [100]", min_value=1, value=100)
learning_rate = st.number_input("Do you want to change the learning rate? [0.1]", min_value=0.001, value=0.1, step=0.001)
max_depth = st.number_input("Do you want to change the max depth? [10]", min_value=1, value=10)
subsample = st.number_input("Do you want to change the subsample? [0.7]", min_value=0.001, value=0.7, step=0.001)
reg_lambda = st.number_input("Do you want to change the reg lambda? [1.3]", min_value=0.001, value=1.3, step=0.001)
gamma = st.number_input("Do you want to change the gamma? [0.001]", min_value=0.001, value=0.001, step=0.001)
min_child_weight = st.number_input("Do you want to change the min child weight? [25]", min_value=1, value=25, step=1)
base_score = st.number_input("Do you want to change the base score? [0.0]", min_value=0.0, value=0.0, step=0.001)
tree_method = st.selectbox("Do you want to change the tree method? [exact]", ["exact", "approx"])

default_params = {
    'learning_rate': learning_rate,  # Updated with user input
    'max_depth': max_depth,  # Updated with user input
    'subsample': subsample,  # Updated with user input
    'reg_lambda': reg_lambda,  # Updated with user input
    'gamma': gamma,  # Updated with user input
    'min_child_weight': min_child_weight,  # Updated with user input
    'base_score': base_score,  # Updated with user input
    'tree_method': tree_method,  # Updated with user input
}
st.write("Default Parameters:", default_params)

# Predict
if st.button("Predict"):
    try:
        def xgboost(param: dict):
            # train the from-scratch XGBoost model
            model = XGBoostModel(param, x_train, y_train, random_seed=42)
            model.fit(SquaredErrorObjective(), num_boost_round, verboose=True)
            prediction = model.predict(x_test)
            st.write(f'Loss Score: {SquaredErrorObjective().loss(y_test, prediction)}')

        xgboost(default_params)

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

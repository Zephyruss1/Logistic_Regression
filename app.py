import streamlit as st
import pandas as pd
from scripts.options import args_parser
from datasets.data_preprocess import data_preprocess
from src.xgboost_scratch import XGBoostModel
from scripts.squared_error_objective import SquaredErrorObjective
from sklearn.metrics import r2_score
import plotly_express as px

# Set page configuration
st.set_page_config(
    page_title="Advanced XGBoost Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Advanced XGBoost Model (from Scratch) Prediction App")
st.write("""
Welcome to the advanced version of the XGBoost Prediction App! 
This app offers a user-friendly interface to experiment with custom XGBoost models and analyze predictions interactively.
""")

# Sidebar for hyperparameter input
st.sidebar.header("Model Hyperparameters")
num_boost_round = st.sidebar.number_input(
    "Number of Boosting Rounds", min_value=1, value=100, help="Number of iterations to train the model."
)
learning_rate = st.sidebar.number_input(
    "Learning Rate", min_value=0.001, value=0.1, step=0.001, help="Step size shrinkage used in updates."
)
max_depth = st.sidebar.number_input(
    "Max Depth", min_value=1, value=10, help="Maximum depth of each tree."
)
subsample = st.sidebar.number_input(
    "Subsample", min_value=0.001, value=0.7, step=0.001, help="Subsample ratio of the training instances."
)
reg_lambda = st.sidebar.number_input(
    "Regularization Lambda", min_value=0.001, value=1.3, step=0.001, help="L2 regularization term on weights."
)
gamma = st.sidebar.number_input(
    "Gamma", min_value=0.001, value=0.001, step=0.001, help="Minimum loss reduction required to make a split."
)
min_child_weight = st.sidebar.number_input(
    "Min Child Weight", min_value=1, value=25, step=1, help="Minimum sum of instance weight needed in a child."
)
base_score = st.sidebar.number_input(
    "Base Score", min_value=0.0, value=0.0, step=0.001, help="Initial prediction score of all instances."
)
tree_method = st.sidebar.selectbox(
    "Tree Method", ["exact", "approx"], help="Algorithm used for tree construction."
)

# Sidebar display of selected parameters
default_params = {
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    'subsample': subsample,
    'reg_lambda': reg_lambda,
    'gamma': gamma,
    'min_child_weight': min_child_weight,
    'base_score': base_score,
    'tree_method': tree_method,
}
st.sidebar.write("**Selected Parameters:**")
st.sidebar.json(default_params)

# Data Preprocessing
parser = args_parser()
_args, unknown = parser.parse_known_args()
(x_train, y_train), (x_test, y_test) = data_preprocess(_args)

# Prediction Section
st.subheader("Train and Predict")
st.write("Click the button below to train the model and view predictions.")

if st.button("Run Prediction"):
    with st.spinner("Training the model and making predictions..."):
        try:
            def xgboost(param: dict):
                model = XGBoostModel(param, x_train, y_train, random_seed=42)
                losses = model.fit(SquaredErrorObjective(), num_boost_round, verboose=True)
                prediction = model.predict(x_test)
                loss = SquaredErrorObjective().loss(y_test, prediction)
                return losses, prediction, loss

            losses, predictions, loss_score = xgboost(default_params)

            # Display results
            st.success(f"Training completed! Loss Score: {loss_score:.4f}")
            st.write(f"Number of Boosting Rounds: {num_boost_round}")

            r2 = r2_score(y_test, predictions)
            st.write(f"R² Score: {r2:.4f}")

            if losses:
                st.subheader("Loss Progression")

                # Create a DataFrame for Plotly
                loss_df = pd.DataFrame({"Boosting Round": range(len(losses)), "Loss": losses})

                # Generate an interactive line plot with Plotly
                fig = px.line(
                    loss_df,
                    x="Boosting Round",
                    y="Loss",
                    title="Loss Progression During Training",
                    labels={"Loss": "Loss", "Boosting Round": "Boosting Round"},
                    line_shape="linear"
                )

                # Show the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No losses were recorded during training.")

            # Show Predictions as DataFrame
            if predictions is not None:
                st.subheader("Predictions")
                results_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
                st.dataframe(results_df)

                # Download predictions as CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("Predictions were not generated. Please check the model and dataset.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

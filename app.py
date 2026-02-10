#importing all relevant modules to build the streamlit app
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Loading the trained model
model = joblib.load("projectModel.pkl")

# creating the base page
st.set_page_config(page_title="Bank Marketing Predictor", layout="centered") # this will set the tab title for the page and center all content
st.title("Bank Marketing Term Deposit Predictor") # title of the page
st.write("Fill in the inputs, then click **Predict**.") # this is just a instruction line

# Get feature names in the same order as training to ensure the data is fitted correctly into the model
if not hasattr(model, "feature_names_in_"):
    st.error(
        "Your saved model does NOT contain feature_names_in_. "
        "This happens if you trained using numpy arrays instead of a pandas DataFrame.\n\n"
        "Fix: retrain using a DataFrame (X_train must be a DataFrame with columns), then save again."
    )
    st.stop()

FEATURES = list(model.feature_names_in_)  # exact order from fit

# converting inputs from frontend to the datatypes the model has to ensure fitting is done properly
def make_blank_row():
    return pd.DataFrame([{col: 0 for col in FEATURES}])

def set_one_hot(df, prefix, chosen_value):
    """
    Sets df[f"{prefix}_{chosen_value}"]=1 if it exists.
    Clears other columns with same prefix_.
    """
    for c in df.columns:
        if c.startswith(prefix + "_"):
            df.at[0, c] = 0

    col_name = f"{prefix}_{chosen_value}"
    if col_name in df.columns:
        df.at[0, col_name] = 1

def pick_label_that_exists(prefix, options):
    """
    options = ["Unknown","unknown"] etc.
    returns the option whose column name exists in FEATURES.
    """
    for opt in options:
        if f"{prefix}_{opt}" in FEATURES:
            return opt
    # fallback: just return first option
    return options[0]



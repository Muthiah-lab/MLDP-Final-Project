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

# creates a new row where all the features are set to default zero(prefilled form)
def make_blank_row():
    return pd.DataFrame([{col: 0 for col in FEATURES}])

# converts all values into one hot encoded values
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
    # fallback: return first option
    return options[0]

# Determine what your model expects for Unknown casing
education_unknown_label = pick_label_that_exists("education", ["unknown", "Unknown"])
contact_unknown_label   = pick_label_that_exists("contact",   ["unknown", "Unknown"])
poutcome_unknown_label  = pick_label_that_exists("poutcome",  ["unknown", "Unknown"])

# frontend user interface
with st.expander("Numeric inputs", expanded=True):
    c1, c2 = st.columns(2)

    age = c1.number_input("age", min_value=0, max_value=120, value=58, step=1)
    balance = c2.number_input("balance", value=2143, step=1)

    day = c1.number_input("day", min_value=1, max_value=31, value=5, step=1)
    duration = c2.number_input("duration (seconds)", min_value=0, value=261, step=1)

    campaign = c1.number_input("campaign", min_value=0, value=1, step=1)
    pdays = c2.number_input("pdays", value=-1, step=1)

    previous = c1.number_input("previous", min_value=0, value=0, step=1)
    pdays_contacted = c2.number_input("pdays_contacted", min_value=0, value=1, step=1)

with st.expander("Categorical inputs", expanded=True):
    job = st.selectbox(
        "job",
        ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
         "retired", "self-employed", "services", "student", "technician", "unemployed"],
        index=4
    )

    marital = st.selectbox("marital", ["divorced", "married", "single"], index=1)

    education = st.selectbox(
        "education",
        [education_unknown_label, "primary", "secondary", "tertiary"],
        index=3
    )

    default = st.selectbox("default", ["no", "yes"], index=0)
    housing = st.selectbox("housing", ["no", "yes"], index=0)
    loan = st.selectbox("loan", ["no", "yes"], index=0)

    contact = st.selectbox(
        "contact",
        [contact_unknown_label, "cellular", "telephone"],
        index=1
    )

    month = st.selectbox(
        "month",
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        index=4
    )

    poutcome = st.selectbox(
        "poutcome",
        [poutcome_unknown_label, "failure", "other", "success"],
        index=0
    )

# Build row (in training order)
row = make_blank_row()

# numeric values
row.at[0, "age"] = int(age)
row.at[0, "balance"] = int(balance)
row.at[0, "day"] = int(day)
row.at[0, "duration"] = int(duration)
row.at[0, "campaign"] = int(campaign)
row.at[0, "pdays"] = int(pdays)
row.at[0, "previous"] = int(previous)
row.at[0, "pdays_contacted"] = int(pdays_contacted)

# one-hot encode the new values
set_one_hot(row, "job", job)
set_one_hot(row, "marital", marital)
set_one_hot(row, "education", education)
set_one_hot(row, "default", default)
set_one_hot(row, "housing", housing)
set_one_hot(row, "loan", loan)
set_one_hot(row, "contact", contact)
set_one_hot(row, "month", month)
set_one_hot(row, "poutcome", poutcome)


# Prediction using modal
if st.button("Predict"):
    try:
        pred = model.predict(row)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(row)
            if p.shape[1] >= 2:
                proba = float(p[0, 1])

        # display
        if str(pred) in ["1", "yes", "Yes", "YES", "True", "true"]:
            st.success("✅ Prediction: Client WILL subscribe to term deposit")
        else:
            st.warning("❌ Prediction: Client will NOT subscribe to term deposit")

        if proba is not None:
            st.info(f"Probability of subscribing (class 1): **{proba:.4f}**")

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
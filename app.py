#importing all relevant modules to build the streamlit app
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import base64

# Loading the trained model
model = joblib.load("projectModel.pkl")

# creating the base page
st.set_page_config(page_title="Bank Marketing Predictor", layout="centered") # this will set the tab title for the page and center all content
st.title("Bank Marketing Term Deposit Predictor") # title of the page
st.write("Fill in the inputs, then click **Predict**.") # this is just a instruction line
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function
set_bg("image.png")

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

# expander means a div container
with st.expander("Numeric inputs", expanded=True):
    c1, c2 = st.columns(2)
    age = c1.number_input("Client's Age", min_value=0, max_value=120, value=58, step=1)
    balance = c2.number_input("Client's Current Balance", value=2143, step=1)
    day = c1.number_input("Which day the client was contacted(the day of the month)?", min_value=1, max_value=31, value=5, step=1)
    duration = c2.number_input("How long did the call last(in seconds)?", min_value=0, value=261, step=1)
    campaign = c1.number_input("How many times the client was contacted?", min_value=0, value=1, step=1)
    pdays = c2.number_input("Days since last contact (-1 means never contacted)", value=-1, step=1)
    previous = c1.number_input("How many times the customer was contacted previously?", min_value=0, value=0, step=1)
    pdays_contacted = c2.number_input("Has client been contacted before? (0 = No, 1 = Yes)?", min_value=0, value=1, step=1)

# expander means a div container
with st.expander("Categorical inputs", expanded=True):
    job = st.selectbox("Job",["Admin.", "Blue-Collar", "Entrepreneur", "Housemaid", "Management","Retired", "Self-employed", "Services", "Student", "Technician", "Unemployed"],index=4)
    marital = st.selectbox("Marital Status", ["Unknown","Divorced", "Married", "Single"], index=1)
    education = st.selectbox("Education",[education_unknown_label,"Primary", "Secondary", "Tertiary"],index=3)
    default = st.selectbox("Does the customer have credit?", ["No", "Yes"], index=0)
    housing = st.selectbox("Housing status?", ["No", "Yes"], index=0)
    loan = st.selectbox("Does the customer have a loan", ["No", "Yes"], index=0)
    contact = st.selectbox("How was the customer approached/contacted",["In-Person", "Cellular", "Telephone"],index=1)
    month = st.selectbox("Which month was the customer contacted?",["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],index=4)
    poutcome = st.selectbox("During the pervious approach, What was the customer's response?",[ "Yes", "No", "Maybe"],index=0)

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
            st.markdown(
                """
                    <div style="background-color:#15803d; padding:15px;border-radius:10px;color:white;font-weight:bold;font-size:18px;">
                        ✅ Prediction: Client WILL subscribe to term deposit
                    </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                    <div style="background-color:#dc2626; padding:15px;border-radius:10px;color:white;font-weight:bold;font-size:18px;">
                        ❌ Prediction: Client will NOT subscribe to term deposit
                    </div>
                """,
                unsafe_allow_html=True
            )
        if proba is not None:
            st.markdown(
                f"""
                    <div style="background-color:#1e40af;padding:12px;border-radius:10px;color:white;font-size:16px;=font-weight:500;">
                        Probability of subscribing (class 1): <b>{proba:.4f}</b>
                    </div>
                """, unsafe_allow_html=True
            )            
    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
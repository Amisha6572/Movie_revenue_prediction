import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model + columns
model = joblib.load("best_movie_model.pkl")
training_columns = joblib.load("training_columns.pkl")

st.title("🎬 Movie Revenue Prediction App")

st.write("Enter complete movie details:")

# Inputs
movie_name = st.text_input("Movie Name")
genre = st.text_input("Genre")
lead_star = st.text_input("Lead Star")
director = st.text_input("Director")
music_director = st.text_input("Music Director")

new_actor = st.text_input("New Actor")
new_director = st.text_input("New Director")
new_music_director = st.text_input("New Music Director")

budget = st.number_input("Budget(INR)", min_value=0.0)
screens = st.number_input("Number of Screens", min_value=1)

franchise = st.selectbox("Whether Franchise", ["Yes", "No"])
remake = st.selectbox("Whether Remake", ["Yes", "No"])
release_period = st.selectbox("Release Period", ["Holiday", "Non-Holiday"])

if st.button("Predict Revenue"):

    input_df = pd.DataFrame([{
        "Movie Name": movie_name,
        "Genre": genre,
        "Lead Star": lead_star,
        "Director": director,
        "Music Director": music_director,
        "New Actor": new_actor,
        "New Director": new_director,
        "New Music Director": new_music_director,
        "Budget(INR)": budget,
        "Number of Screens": screens,
        "Whether Franchise": franchise,
        "Whether Remake": remake,
        "Release Period": release_period
    }])

    # Dummy encoding
    input_df = pd.get_dummies(input_df)

    # Add missing cols
    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[training_columns]

    # Predict log revenue
    pred_log = model.predict(input_df)[0]

    revenue = np.expm1(pred_log)

    st.success(f"🎉 Predicted Revenue: ₹ {revenue:,.2f}")

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import pickle

# Define teams and cities
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']
cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

# Model path
new_model_path = 'simple_pipe.pkl'

# Check if the model file exists before trying to load it
if os.path.exists(new_model_path):
    try:
        # Load the model
        loaded_pipe = joblib.load(new_model_path)
        print("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading the model: {e}")
else:
    # Handle the case where the file does not exist
    st.error(f"Model file '{new_model_path}' not found. Please train the model first.")
    st.stop()


# Streamlit app
st.title('Cricket Score Predictor')

# Layout columns
col1, col2 = st.columns(2)

# Dropdown for selecting batting team
with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))

# Dropdown for selecting bowling team
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

# Dropdown for selecting city
city = st.selectbox('Select city', sorted(cities))

# Layout columns for additional input fields
col3, col4, col5 = st.columns(3)

# Input field for current score
with col3:
    current_score = st.number_input('Current Score')

# Input field for overs done
with col4:
    overs = st.number_input('Overs done (works for over > 5)')

# Input field for wickets out
with col5:
    wickets = st.number_input('Wickets out')

# Input field for runs scored in last 5 overs
last_five = st.number_input('Runs scored in last 5 overs')

# Predict button
if st.button('Predict Score'):
    balls_left = (120 - (overs * 6))
    wickets_left = (10 - wickets)
    crr = current_score / overs

    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'current_score': [current_score],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'last_five': [last_five]
    })

    # Make prediction
    result = pipe.predict(input_df)

    # Display predicted score
    st.header("Predicted Score: " + str(int(result[0])))

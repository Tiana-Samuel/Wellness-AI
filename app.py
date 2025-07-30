import streamlit as st
import pandas as pd
import joblib

# Load the saved model and label encoder
loaded_model = joblib.load('risk_level_model.pkl')
loaded_le = joblib.load('risk_level_label_encoder.pkl')

def identify_conditions(row):
    mental = set(row['Mental Symptoms'])
    physical = set(row['Physical Symptoms'])

    conditions = []

    if 'Overthinking' in mental or 'Anxiety' in mental or 'Panic Attack' in mental:
        conditions.append("Anxiety")
    if 'Hopelessness' in mental or 'Sadness' in mental or 'Suicidal thoughts' in mental:
        conditions.append("Depression")
    if 'Feeling overwhelmed' in mental or 'Mood Swings' in mental or 'Irritability' in mental:
        conditions.append("Stress")
    if len(physical) >= 3:
        conditions.append("Physical Illness")

    return conditions if conditions else ["Unknown"]

def predict_risk_level_with_conditions(new_data):
    """
    Predicts the risk level and identifies possible conditions for new data.

    Args:
        new_data (pd.DataFrame): A DataFrame containing the new data with the same columns
                                 as the original data (excluding the target).
                                 Must include 'Physical Symptoms' and 'Mental Symptoms' columns.

    Returns:
        tuple: A tuple containing the predicted risk level (str) and a list of possible conditions (list).
    """
    # Backup original for condition analysis
    original_data = new_data.copy()

    # Drop non-numeric columns used only for condition identification
    data_for_prediction = new_data.drop(['Physical Symptoms', 'Mental Symptoms', 'Food Log', 'Date'], axis=1, errors='ignore')


    # Fill missing dummy columns
    # This requires access to the original training data columns (X) or a predefined list
    # For this example, we'll assume X is available globally or passed in, or define column names.
    # A more robust solution in a real app would save the list of training columns.
    # Assuming X.columns are available from a previous cell or loaded from a file
    # For now, let's hardcode the expected columns based on the previous code's X
    expected_columns = ['Mood', 'Sleep Hours', 'Screen Time', 'Stress Level', 'Wellness Score', 'Back pain', 'Chest pain', 'Cold', 'Constipation', 'Cough', 'Diarrhea', 'Dizziness', 'Fatigue', 'Fever', 'Headache', 'Joint pain', 'Loss of appetite', 'Muscle pain', 'Nausea', 'Rapid heartbeat', 'Shortness of breath', 'Sore throat', 'Stomach upset', 'Anger', 'Anxiety', 'Crying spells', 'Difficulty concentrating', 'Feeling overwhelmed', 'Hopelessness', 'Insomnia', 'Irritability', 'Loneliness', 'Low self-esteem', 'Mood Swings', 'Numbness', 'Overthinking', 'Panic Attack', 'Restlessness', 'Sadness', 'Suicidal thoughts', 'Exercise_No', 'Exercise_Yes']


    missing_cols = set(expected_columns) - set(data_for_prediction.columns)
    for c in missing_cols:
        data_for_prediction[c] = 0
    data_for_prediction = data_for_prediction[expected_columns]


    # Predict
    prediction_encoded = loaded_model.predict(data_for_prediction)
    predicted_risk = loaded_le.inverse_transform(prediction_encoded)[0]

    # Identify possible conditions
    conditions = identify_conditions(original_data.iloc[0])  # Assuming 1-row input

    return predicted_risk, conditions

st.title('Mental Health Risk Level Predictor')
st.write('Enter your daily data to predict your mental health risk level and identify potential conditions.')

# Input fields for features


st.header('Daily Data Inputs')

mood = st.number_input('Mood (1-5)', min_value=1, max_value=5, value=3)
sleep_hours = st.number_input('Sleep Hours', min_value=0.0, value=7.0, step=0.1)
screen_time = st.number_input('Screen Time (hours)', min_value=0.0, value=5.0, step=0.1)
exercise = st.selectbox('Exercise', ['Yes', 'No'])
stress_level = st.number_input('Stress Level (1-5)', min_value=1, max_value=5, value=3)
wellness_score = st.number_input('Wellness Score (1-10)', min_value=1, max_value=10, value=5)

# Dropdowns and multiselects
physical_symptom_options = [
    "Headache", "Fatigue", "Fever", "Nausea", "Stomach upset", "Rapid heartbeat",
    "Cough", "Chest pain", "Cold", "Muscle pain", "Back pain", "Joint pain",
    "Loss of appetite", "Constipation", "Diarrhea", "Sore throat",
    "Shortness of breath", "Dizziness"
]

mental_symptom_options = [
    "Overthinking", "Panic Attack", "Hopelessness", "Loneliness", "Irritability",
    "Mood Swings", "Difficulty concentrating", "Anxiety", "Sadness", "Insomnia",
    "Crying spells", "Restlessness", "Feeling overwhelmed", "Low self-esteem",
    "Anger", "Suicidal thoughts", "Numbness"
]

food_log_options = ["Ate once", "Ate twice", "Ate thrice"]

selected_physical_symptoms = st.multiselect('Select Physical Symptoms', physical_symptom_options)
selected_mental_symptoms = st.multiselect('Select Mental Symptoms', mental_symptom_options)
food_log_input = st.selectbox('Food Log', food_log_options)

physical_symptoms = selected_physical_symptoms
mental_symptoms = selected_mental_symptoms


# Access the values entered by the user
user_data = {
    'Mood': mood,
    'Sleep Hours': sleep_hours,
    'Screen Time': screen_time,
    'Exercise': exercise,
    'Stress Level': stress_level,
    'Wellness Score': wellness_score,
    'Physical Symptoms': selected_physical_symptoms,
    'Mental Symptoms': selected_mental_symptoms,
    'Food Log': food_log_input,
    'Date': '2025-07-01' # Placeholder for Date, as it's not used in prediction but needed for identify_conditions structure
}

# Convert to DataFrame for prediction function
user_df = pd.DataFrame([user_data])

# Apply one-hot encoding to the 'Exercise' column
user_df = pd.get_dummies(user_df, columns=['Exercise'], drop_first=False)

# Call the prediction function
predicted_risk_with_conditions, identified_conditions = predict_risk_level_with_conditions(user_df)

# Display the results
st.header('Prediction Results')

st.write(f"**Predicted Risk Level:** {predicted_risk_with_conditions}")

if identified_conditions and identified_conditions != ["Unknown"]:
    st.write(f"**Identified Possible Conditions:** {', '.join(identified_conditions)}")
else:
    st.write("**Identified Possible Conditions:** No specific conditions identified based on your inputs.")

# To run this Streamlit app:
# 1. Save the code above as 'app.py'. This has been done automatically by the notebook cells.
# 2. Open the Colab terminal (usually View -> Terminal -> New terminal).
# 3. Install Streamlit if you haven't already: pip install streamlit
# 4. Run the app using the command: streamlit run app.py

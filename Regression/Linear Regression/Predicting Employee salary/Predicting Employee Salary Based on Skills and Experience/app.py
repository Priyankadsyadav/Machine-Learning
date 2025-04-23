import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load the trained model, scaler, and selected features
model = joblib.load('salary_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Streamlit App: Create user input form
st.title("Salary Prediction")

# Create input fields for user to input their data
st.sidebar.title("Input Parameters")
experience_years = st.sidebar.slider("Years of Experience", min_value=0, max_value=15, value=5)
certifications = st.sidebar.number_input("Certifications", min_value=0, max_value=5, value=2)
skill_match = st.sidebar.slider("Skill Match (0.5 to 1.0)", min_value=0.5, max_value=1.0, value=0.8)
education_level = st.sidebar.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
city = st.sidebar.selectbox("Location", ["New York", "Berlin", "Mumbai", "San Francisco"])

# Label encoding for education level (assuming 0 for Bachelors, 1 for Masters, 2 for PhD)
education_level_mapping = {"Bachelors": 0, "Masters": 1, "PhD": 2}
encoded_education_level = education_level_mapping[education_level]

# Prepare the input data (one-hot encoding for categorical features)
input_data = {
    'Experience_Years': [experience_years],
    'Certifications': [certifications],
    'Skill_Match_Score': [skill_match],
    'Education_Level': [encoded_education_level],  # Directly use the label encoded value
    'Location_Berlin': [1 if city == "Berlin" else 0],
    'Location_Mumbai': [1 if city == "Mumbai" else 0],
    'Location_New York': [1 if city == "New York" else 0],
    'Location_San Francisco': [1 if city == "San Francisco" else 0]
}

# Convert the input data into a DataFrame
input_df = pd.DataFrame(input_data)

# Ensure the input data has the selected features
input_df = input_df[selected_features]

# Scale the input data
scaled_input = scaler.transform(input_df)

# Make the prediction
prediction = model.predict(scaled_input)

# Display the predicted salary
st.write(f"Predicted Salary: ${prediction[0]:,.2f}")

# Visualization: Salary Distribution (to show general salary trends)
def plot_salary_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Salary'], kde=True, bins=30)
    plt.title('Salary Distribution')
    st.pyplot(plt)

# Example dataset (Replace with your actual dataset or load it from a file)
df = pd.DataFrame({
    'Salary': [50000, 60000, 70000, 80000, 75000, 85000, 90000],
    'Experience_Years': [1, 2, 3, 5, 6, 8, 10],
    'Certifications': [1, 2, 3, 4, 2, 3, 4],
    'Skill_Match_Score': [0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9],
    'Education_Level_Masters': [0, 1, 0, 1, 0, 1, 0],
    'Education_Level_PhD': [0, 0, 0, 0, 1, 0, 0],
    'Location_Berlin': [1, 0, 0, 0, 0, 0, 0],
    'Location_Mumbai': [0, 1, 0, 0, 0, 0, 0],
    'Location_New_York': [0, 0, 1, 0, 0, 0, 0],
    'Location_San_Francisco': [0, 0, 0, 1, 0, 0, 1]
})

plot_salary_distribution(df)

# Add a download button for the prediction result
result_df = pd.DataFrame({
    'Experience_Years': [experience_years],
    'Certifications': [certifications],
    'Skill_Match_Score': [skill_match],
    'Predicted_Salary': [prediction[0]]
})

# Convert the result dataframe to CSV and then to bytes
csv_data = result_df.to_csv(index=False)
csv_bytes = csv_data.encode('utf-8')

# Add the download button with the correct data format
st.download_button(
    label="Download Prediction Result",
    data=csv_bytes,
    file_name="salary_prediction.csv",
    mime="text/csv"
)

# Add testimonials section for better engagement
st.subheader("How this can help you")
st.write("""
    This tool helps professionals estimate their expected salary based on factors like experience, education, and location.
    Use this tool for career planning, job interviews, and salary negotiations.
""")

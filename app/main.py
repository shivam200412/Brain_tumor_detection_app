import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

def get_clean_data():
  data_path = os.path.join("data", "processed_brain_tumor_data.csv")
  data = pd.read_csv(data_path)

  data['Tumor_Type'] = data[['Tumor_Type_Benign', 'Tumor_Type_Malignant']].idxmax(axis=1)
  data['Tumor_Type'] = data['Tumor_Type'].map({'Tumor_Type_Benign': 0, 'Tumor_Type_Malignant': 1})
  data.drop(['Tumor_Type_Benign', 'Tumor_Type_Malignant'], axis=1, inplace=True)

  return data



def add_sidebar():
  st.sidebar.header("Patient Information")

  data = get_clean_data()

  age = st.sidebar.slider("Enter your age:", min_value=0, max_value=120, step=1)
  Tumor_Size = st.sidebar.slider( 'Tumor Size',min_value=float(0),max_value=float(10),value=float(5))
  Gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
  Location = st.sidebar.selectbox('Location', ['Frontal', 'Occipital', 'Parietal', 'Temporal'])
  Histology = st.sidebar.selectbox('Histology', ['Astrocytoma', 'Glioblastoma', 'Medulloblastoma', 'Meningioma'])
  Symptom_1 = st.sidebar.selectbox('Symptom_1', ['Headache', 'Nausea', 'Seizures', 'Vision Issues'])
  Symptom_2 = st.sidebar.selectbox('Symptom_2', ['Headache', 'Nausea', 'Seizures', 'Vision Issues'])
  Symptom_3 = st.sidebar.selectbox('Symptom_3', ['Headache', 'Nausea', 'Seizures', 'Vision Issues'])
  Family_History = st.sidebar.selectbox('Family_History', ['Yes', 'No'])

  input_data = {
    "Age": age,
    "Tumor_Size": Tumor_Size,
    "Gender": Gender,
    "Location": Location,
    "Histology": Histology,
    "Symptom_1": Symptom_1,
    "Symptom_2": Symptom_2,
    "Symptom_3": Symptom_3,
    "Family_History": Family_History
  }

  return input_data




def create_input_visualization(input_data):
    # Create a card-like visualization for each input
    cols = st.columns(3)  # 3 columns layout

    # Format the data for display
    display_data = input_data

    # First column
    with cols[0]:
      st.metric("Age", f"{display_data['Age']} years")
      st.metric("Tumor Size", f"{display_data['Tumor_Size']:.1f} cm")
      st.metric("Gender", display_data['Gender'])

    # Second column
    with cols[1]:
      st.metric("Location", display_data['Location'])
      st.metric("Histology", display_data['Histology'])
      st.metric("Family History", display_data['Family_History'])

    # Third column
    with cols[2]:
      st.metric("Primary Symptom", display_data['Symptom_1'])
      st.metric("Secondary Symptom", display_data['Symptom_2'])
      st.metric("Tertiary Symptom", display_data['Symptom_3'])




def get_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb"))
  encoder = pickle.load(open("model/encoder.pkl", "rb"))

  df = pd.DataFrame([input_data])

  # Define categorical columns to encode
  categorical_cols = ['Gender', 'Location', 'Histology', 'Symptom_1',
                      'Symptom_2', 'Symptom_3', 'Family_History']

  df[categorical_cols] = df[categorical_cols].astype(str)

  one_hot_encoded = encoder.transform(df[categorical_cols])
  one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_cols))
  df_encoded = pd.concat([df, one_hot_df], axis=1)
  df = df_encoded.drop(categorical_cols, axis=1)

  prediction = model.predict(df)

  probability = model.predict_proba(df)

  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")

  return prediction[0], probability[0]




# Function to create a gauge chart
def create_gauge_chart(probability, prediction):
  if prediction == 1:  # Assuming 1 is malignant
    value = probability[1]
    title = "Malignant Probability"
    color = "red"
  else:
    value = probability[0]
    title = "Benign Probability"
    color = "green"

  fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value * 100,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': title},
    gauge={
      'axis': {'range': [0, 100]},
      'bar': {'color': color},
      'steps': [
        {'range': [0, 50], 'color': "lightgray"},
        {'range': [50, 100], 'color': "gray"}
      ],
      'threshold': {
        'line': {'color': "black", 'width': 4},
        'thickness': 0.75,
        'value': 50
      }
    }
  ))
  return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    if model:
      prediction, probability = get_predictions(input_data)
      if prediction is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
          if prediction == 1:  # Assuming 1 is malignant
            st.markdown(f"<h2 style='color: red;'>Malignant</h2>", unsafe_allow_html=True)
            st.markdown(f"<p>Probability: {probability[1] * 100:.1f}%</p>", unsafe_allow_html=True)
          else:
            st.markdown(f"<h2 style='color: green;'>Benign</h2>", unsafe_allow_html=True)
            st.markdown(f"<p>Probability: {probability[0] * 100:.1f}%</p>", unsafe_allow_html=True)
        with col2:
          fig = create_gauge_chart(probability, prediction)
          st.plotly_chart(fig, use_container_width=True)
        st.write(
          "**Note:** This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice. Please consult with a healthcare provider for accurate diagnosis and treatment.")
    else:
      st.error("Model could not be loaded.")




def main():
  st.set_page_config(
    page_title="Brain Tumor Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )

  input_data = add_sidebar()


  with st.container():
    st.title("Brain Tumor Predictor")
    st.write(
      "This app uses machine learning to predict brain tumor malignancy based on patient data.")

    # Input visualization section
    st.subheader("Patient Input Data")
    create_input_visualization(input_data)


    if st.button("Predict"):
      st.subheader("Prediction")
      add_predictions(input_data)


if __name__ == '__main__':
  main()


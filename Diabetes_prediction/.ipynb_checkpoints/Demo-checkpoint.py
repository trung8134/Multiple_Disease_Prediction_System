import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# loading the saved model
# Path in python: "\" -> "/"
load_model_diabetes = pickle.load(open('C:/Users/GIANG LINH/Data_Science/Getting_Knowing_Data/StreamLit_Project/StreamLit_Example/Diabetes_prediction/diabetes_model_deploy.sav', 'rb'))


def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshape = input_data_as_array.reshape(1, -1)

    # scale input_data
    scale_input_data = scaler.fit_transform(input_data_reshape)

    prediction = load_model_diabetes.predict(scale_input_data)
    
    if prediction[0]==0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
    

def main():
    # title
    st.title("Diabetes Prediction Web App")
    
    # getting the input data from the user
    # Pregnancies, Glucose, BloodPressure, SkinThickNess, Insulin, BMI, DiabetesPedigreeFunction, Age
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickNess = st.text_input("SkinThickNess Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the Person")
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickNess, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    
    st.success(diagnosis)
    


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




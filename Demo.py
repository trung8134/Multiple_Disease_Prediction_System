import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu

scaler = MinMaxScaler()

oad_model_parkinsons = pickle.load(open("D:/GitHub/Multiple_Disease_Prediction_System/Parkinsons_Prediction/Parkinsons_model_deploy.sav", "rb"))
load_model_diabetes = pickle.load(open('D:/GitHub/Multiple_Disease_Prediction_System/Diabetes_prediction/diabetes_model_deploy.sav', 'rb'))
load_model_Heart = pickle.load(open("D:/GitHub\Multiple_Disease_Prediction_System/Heart_Prediction/heart_model_deploy.sav", "rb"))

# diabetes model 
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
    
# parkinsons model
def parkinsons_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshape = input_data_as_array.reshape(1, -1)

    # scale input_data
    scale_input_data = scaler.fit_transform(input_data_reshape)
    
    prediction = load_model_parkinsons.predict(scale_input_data)
    
    if prediction[0] == 0:
        return "The person does not have Parkinsons Disease"
    else:
        return "The person has Parkinsons"
    
# heart model
def Heart_Prediction(input_data):
    
    input_data_as_array = np.asarray(input_data)

    # reshape input_data
    input_data_reshape = input_data_as_array.reshape(1, -1)

    # predicction
    predictions = load_model_Heart.predict(input_data_reshape)
    
    if predictions[0] == 0:
        return "The person does have not a Heart Disease"
    else:
        return "The person has Heart Disease"
    
# streamlit
def main():
    with st.sidebar:
        selected = option_menu("Multiple Disease Prediction System", 

                               ["Diabetes Prediction", 
                                "Heart Disease Prediction",
                                "Parkinsons Prediction"],
                               default_index = 1)

    # Diabetes Prediction page
    if selected == "Diabetes Prediction":
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


    # Heart Disease Prediction page
    if selected == "Heart Disease Prediction":
        st.title("Heart Diasease Prediction Web App")
    
        #age,sex,cp,
        #trestbps,chol,fbs,
        #restecg,thalach,exang,
        #oldpeak,slope,ca,
        #thal

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input("Age")
            trestbps = st.text_input("Trestbps")
            restecg = st.text_input("Restecg")
            oldpeak = st.text_input("Oldpeak")
            thal = st.text_input("Thal")

        with col2:
            sex = st.text_input("Sex")
            chol = st.text_input("chol")
            thalach = st.text_input("Thalach")
            slope = st.text_input("slope")

        with col3:
            cp = st.text_input("CP")
            fbs = st.text_input("FBS")
            exang = st.text_input("Exang")
            ca = st.text_input("CA")

        # code for prediction
        diagnosis = ''

        # creating a button for Prediction
        if st.button("Heart Test Result"):
            diagnosis = Heart_Prediction([age,sex,cp, trestbps,chol,fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

        st.success(diagnosis)

    # Parkinsons Prediction
    if selected == "Parkinsons Prediction":
        # title
        st.title("Parkinsons Prediction Web App")

        # getting the input data from the user
        # MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs),
        # MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB),
        # Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA, NHR,
        # HNR, RPDE, DFA, spread1, spread2,
        # D2, PPE

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            Fo = st.text_input("MDVP:Fo(Hz)")
            RAP = st.text_input("MDVP:RAP")
            Shimmer_APQ3 = st.text_input("Shimmer:APQ3")
            HNR = st.text_input("HNR")
            D2 = st.text_input("D2")

        with col2:
            Fhi = st.text_input("MDVP:Fhi(Hz)")
            PPQ = st.text_input("MDVP:PPQ")
            Shimmer_APQ5 = st.text_input(" Shimmer:APQ5)")
            RPDE = st.text_input(" RPDE")
            PPE = st.text_input("PPE")

        with col3:
            Flo = st.text_input("MDVP:Flo(Hz)")
            Jitter_DDP = st.text_input("Jitter:DDP")
            APQ = st.text_input("MDVP:APQ")
            DFA = st.text_input("DFA")

        with col4:
            Jitter_phantram = st.text_input("MDVP:Jitter(%)")
            MDVP_Shimmer = st.text_input("MDVP:Shimmer")
            Shimmer_DDA = st.text_input("Shimmer:DDA")
            spread1 = st.text_input("spread1")
        with col5:
            Jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
            Shimmer_dB = st.text_input(" MDVP:Shimmer(dB)")
            NHR = st.text_input("NHR")
            spread2 = st.text_input("spread2")


        # code for prediction
        diagnosis = ''

        # creating a button for Prediction
        if st.button("Parkinsons Test Result"):
            diagnosis = parkinsons_prediction([Fo, Fhi, Flo, Jitter_phantram, Jitter_Abs, RAP, PPQ, Jitter_DDP, MDVP_Shimmer, Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])


        st.success(diagnosis)
        
if __name__ == "__main__":
    main()
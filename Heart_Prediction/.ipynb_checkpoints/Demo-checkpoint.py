import numpy as np
import pickle
import streamlit as st

load_model_Heart = pickle.load(open("C:/Users/GIANG LINH/Data_Science/Getting_Knowing_Data/StreamLit_Project/StreamLit_Example/Heart_Prediction/heart_model_deploy.sav", "rb"))

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
    
    
    
def main():
    
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
    
    
        





if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# loading the saved model

# Path in python: "\" -> "/"
load_model_parkinsons = pickle.load(open("C:/Users/GIANG LINH/Data_Science/Getting_Knowing_Data/StreamLit_Toturial/StreamLit_Project/Parkinsons_Prediction/Parkinsons_model_deploy.sav", "rb"))

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
    
    
    
    
    
def main():
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
                    


if __name__ == '__main__':
    main()
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
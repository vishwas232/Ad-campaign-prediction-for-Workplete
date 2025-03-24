
####base libraries
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")




import joblib
import pickle
ohe =joblib.load('ohe.pkl')
log = joblib.load("log.pkl")

import streamlit as st
# Streamlit UI
st.title("AD campaign success")
st.write("Enter ad details to predict success.")

Platform = st.selectbox("Platform", ['instagram','linkedin','youtube','facebook','google'])
Content_Type = st.selectbox("Content_Type", ['video','image','text', 'story' ,'carousel'])
Target_Gender = st.selectbox("Target_Gender", ['female' ,'all' ,'male'])
Target_Age = st.selectbox("Target_Age", ['35-44' ,'45-54' ,'25-34', '18-24' ,'55+'])
Region = st.selectbox("Region", ['us' ,'uk', 'germany' ,'india' ,'canada'])
Budget= st.number_input("Budget(110-49950)", min_value=1, step=1)
Duration = st.number_input("Duration (0-60)", min_value=0, step=1)
Clicks = st.number_input("Clicks (100-50000)", min_value=0, step=1)
Conversions = st.number_input("Conversions (13-5000)", min_value=0.0, step=0.1)
CTR = st.number_input("CTR (0.20-1200)", min_value=0.0, step=0.1)
CPC = st.number_input("CPC (0-10)", min_value=0.0, step=0.1)
Conversion_Rate = st.number_input("Conversion_Rate (0-90)", min_value=0.0, step=0.1)


if st.button("Predict Skills"):
    input_data = pd.DataFrame([[Platform,Content_Type,Target_Gender,Target_Age,Region,Budget,Duration,Clicks,Conversions,CTR,CPC,Conversion_Rate]],
                              columns=['Platform', 'Content_Type', 'Target_Gender', 'Target_Age', 'Region',
                                        'Budget', 'Duration', 'Clicks', 'Conversions', 'CTR', 'CPC',
                                        'Conversion_Rate'])
    row=input_data.copy()
    # One-Hot Encoding
    row_ohe = ohe.transform(row[['Region','Target_Age','Target_Gender','Content_Type','Platform']]).toarray()
    row_ohe = pd.DataFrame(row_ohe, columns=ohe.get_feature_names_out())
    row = pd.concat([row.drop(['Region','Target_Age','Target_Gender','Content_Type','Platform'], axis=1), row_ohe], axis=1)
    
    print("Processed Input for Prediction:")
    st.dataframe(row)

    
    st.write("\n Predicted Skills:")
    prob0 = round(float(log.predict_proba(row)[0][0]),2)
    prob1 = round(float(log.predict_proba(row)[0][1]),2)
    st.write("Ad success Probabilities: no - {}, yes - {}".format(prob0, prob1))

    out = log.predict(row)[0]
    st.write(out)
    cats = {1:'yes',0:'no'}
    
    st.write("Ad Click Prediction:", cats[out])
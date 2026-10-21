import streamlit as st 
import pandas as pd 

from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).parent / "Fraud_detection_pipeline.pkl"
model = joblib.load(MODEL_PATH)

st.title("Fraud detection predection App")

st.markdown("please enter the trancation detailes and use the predict button")
st.divider()
transaction_type = st.selectbox("Transaction type", ["PAYMENT","TRANSFER","CAHS_OUT","DEPOSIT"])
amount = st.number_input("Amount", min_value = 0.0 , value = 1000.0)
oldbalanceOrg = st.number_input("old balance (sender)", min_value =0.0, value =10000.0)
newbalanceOrig =st.number_input("New balance (sender)" ,min_value = 0.0 , value =9000.0)
oldbalanceDest =st.number_input("old balance (reciever)" ,min_value = 0.0 , value =10000.0)
newbalanceDest =st.number_input("New balance (reciever)" ,min_value = 0.0 , value = 10000.0)


if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type" : transaction_type,
        "amount" : amount,
        "oldbalanceOrg" : oldbalanceOrg,
        "newbalanceOrig" :newbalanceOrig,
        "oldbalanceDest" : oldbalanceDest,
        "newbalanceDest" : newbalanceDest,
    }])

    prediction = model.predict(input_data)[0]
    st.subheader(f"Prediction : '{int(prediction)}'")
    
    if prediction == 1 :
        st.error("This transaction can be fraud")
    else:
        st.success("This transaction looks it is not fraud")
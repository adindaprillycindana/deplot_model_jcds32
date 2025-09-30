import streamlit as st
import pandas as pd
import numpy as np
import pickle # untuk save model

## Memberikan judul
st.title('Survive Predictor')
st.write('This website can be used to predict survival rate of Titanic Passanger')

st.sidebar.header("Please input customer's features")

# create user input
def create_user_input():
    # numerical: 'pclass',, 'age', 'sibsp', 'parch', 'fare'
    pclass=st.sidebar.slider('pclass',min_value=1,max_value=3,value=1)
    age=st.sidebar.slider('age',min_value=1,max_value=80,value=20)
    sibsp=st.sidebar.slider('sibsp',min_value=0,max_value=8,value=1)
    parch=st.sidebar.slider('parch',min_value=0,max_value=6,value=1)
    fare=st.sidebar.number_input('fare',min_value=0,max_value=512,value=30)

    #categorical: 'sex','embarked'
    sex=st.sidebar.radio('sex',['male','female'])
    embarked=st.sidebar.radio('embarked',['S','C','Q'])

    #create dictionary from data input harus sam adengan pickle 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'
    user_data={
        'pclass'=pclass,
        'sex'=sex,
        'age'=age,
        'sibsp'=sibsp,
        'parch'=parch,
        'fare'=fare,
        'embarked'=embarked
    }

    data_customer = create_user_input()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Customer Features')
        st.write(data_customer.transpose())
    # load model
    with open('best_model.sav','rb') as f:
        model_loaded = pickle.load(f)
    
    #predict to customer data
    target = model_loaded.predict(data_customer) # target 1 atau 0
    probability = model_loaded.predict_proba(data_customer)[0]

    # menmapilkan hasil prediksi
    # kanan
    with col2:
        st.subheader('Prediction Result')
        if target ==1:
            st.write('This passenger will SURVIVE')
        else:
            st.write('This passenger will NOT SURVIVE')
        st.write(f"Probability of Surviving: {probability[1]:.2f}")

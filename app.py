

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

import tensorflow_probability as tfp
tfd = tfp.distributions

from pickle import dump
from pickle import load



scaler = MinMaxScaler()


# load trained model
lgbm_base = pickle.load(open('lgbm_base.pkl', 'rb'))
lgbm_opt = pickle.load(open('lgbm_optimized.pkl', 'rb'))


tf.random.set_seed(42)

np.random.seed(42)


st.markdown("<body style ='color:#E2E0D9;'></body>", unsafe_allow_html=True)



st.markdown("<h4 style='text-align: center; color: #1B9E91;'>House Price Prediction in Ames,Iowa</h4>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: #1B9E91;'>A multi-step process is used to estimate the range of house prices based on your selection. </h5>", unsafe_allow_html=True)


name_list = [
 'OverallQual',
 'YearBuilt',
 'TotalBsmtSF',
 'GrLivArea',
 'MasVnrArea',
 'BsmtFinType1',
 'Neighborhood',
 'GarageType',
 'SaleCondition',
 'BsmtExposure']

name_list_train = [
 'OverallQual',
 'YearBuilt',
 'TotalBsmtSF',
 'GrLivArea',
 'MasVnrArea',
 'BsmtFinType1',
 'Neighborhood',
 'GarageType',
 'SaleCondition',
 'BsmtExposure']

data = pd.read_csv('train.csv')


data = data[name_list_train].values

scaler.fit(data)

description_list = [
 'What is the Overall material and finish quality?',
 'In which year was the Original construction date?',
 'What is the Total square feet of basement area?',
 'What is the Above grade (ground) living area in square feet?',
 'What is the Masonry veneer area in square feet??',
 'What is the Quality of basement finished area?',
 'Where is the physical locations within Ames city limits?',
 'Where is the location of the Garage?',
 'What is the condition of the sale?',
 'Does the house have walkout or garden-level basement walls?'
 ]

min_list = [1.0,
 1950.0,
 0.0,
 0.0,
 334.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0
]

max_list = [
 10.0,
 2010.0,
 2336.0,
 6110.0,
 4692.0,
 10.0,
 10.0,
 3.0,
 10.0,
 1.0,
]

count = 0

with st.sidebar:

    for i in range(len(name_list)):

            

        variable_name = name_list[i]
        globals()[variable_name] = st.slider(description_list[i] ,min_value=int(min_list[i]), max_value =int(max_list[i]),step=1)
      
    st.write("[Kaggle Link to Data Set](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)")

    


data_df = {

 'OverallQual': [OverallQual],
 'YearBuilt': [YearBuilt],
 'TotalBsmtSF': [TotalBsmtSF],
 'GrLivArea':[GrLivArea],
 'MasVnrArea': [MasVnrArea]
 'BsmtFinType1': [BsmtFinType1]
 'Neighborhood': [Neighborhood]
 'GarageType': [GarageType]
 'SaleCondition': [SaleCondition]
 'BsmtExposure': [BsmtExposure]
}

#negloglik = lambda y, p_y: -p_y.log_prob(y) # note this

data_df = pd.DataFrame.from_dict(data_df)

data_df_normal = scaler.transform(data_df)

y_pred_base = lgbm_base.predict(data_df)
y_pred_optimized = lgbm_opt.predict(data_df)



col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    center_button = st.button('Calculate range of house price')



if center_button:

    import time

    #my_bar = st.progress(0)

    with st.spinner('Calculating....'):
        time.sleep(2)



    st.markdown("<h5 style='text-align: center; color: #1B9E91;'>The price range of your house is between:</h5>", unsafe_allow_html=True)


    col1, col2 = st.columns([3, 3])

    lower_number = "{:,.2f}".format(int(yhat.mean().numpy()-1.95*yhat.stddev().numpy()))
    higher_number = "{:,.2f}".format(int(yhat.mean().numpy()+1.95*yhat.stddev().numpy()))

    col1, col2, col3 = st.columns(3)

    

    with col1:
        st.write("")

    with col2:
        st.subheader("USD "+ str(lower_number))
        st.subheader("       AND ")

        st.subheader(" USD "+str(higher_number))


    with col3:
        st.write("")

    

    

    import base64

    file_ = open("kramer_gif.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<center><img src="data:image/gif;base64,{data_url}" alt="cat gif"></center>',
        unsafe_allow_html=True,
    )
    


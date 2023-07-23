
# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle

np.random.seed(42)

st.markdown("<body style ='color:#E2E0D9;'></body>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: #1B9E91;'>House Price Prediction in Ames, Iowa</h4>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: #1B9E91;'>Optuna optimized LGBM model to estimate the range of house prices based on your selection. </h5>", unsafe_allow_html=True)

st.write("If you want to know the numbers that you picked for some of the features such as Overall Quality, Sale Conditions etc., please check the following link")
st.write("[link to the categorical encoding](https://github.com/aye-thuzar/CS634Project/edit/main/docs.md)")



#setting up the sliders and getting the input the sliders

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

description_list = [
 'What is the Overall material and finish quality?',
 'In which year was the Original construction date?',
 'What is the Total square feet of basement area?',
 'What is the Above grade (ground) living area in square feet?',
 'What is the Masonry veneer area in square feet?',
 'What is the Quality of the basement finished area?',
 'Where are the physical locations within Ames city limits?',
 'Where is the location of the Garage?',
 'What is the condition of the sale?',
 'What is the basement exposure: walkout or garden-level basement walls?'
 ]

min_list = [
 1.0,
 1950.0,
 0.0,
 0.0,
 334.0,
 1.0,
 1.0,
 1.0,
 1.0,
 0.0
]

max_list = [
 10.0,
 2010.0,
 2336.0,
 6110.0,
 4692.0,
 7.0,
 25.0,
 7.0,
 6.0,
 5.0,
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
 'MasVnrArea': [MasVnrArea],
 'BsmtFinType1': [BsmtFinType1],
 'Neighborhood': [Neighborhood],
 'GarageType': [GarageType],
 'SaleCondition': [SaleCondition],
 'BsmtExposure': [BsmtExposure]
}

data_df = pd.DataFrame.from_dict(data_df)

st.write("Please adjust the feature values using the slides on the left: ")
st.write(data_df.head())


#normalizing the data

diff = np.array(max_list)-np.array(min_list)
data_df = (data_df.values - np.array(min_list)) / diff

st.write("Normalized input data")
st.write(data_df)

# load trained model
lgbm_base = pickle.load(open('lgbm_base.pkl', 'rb'))
lgbm_opt = pickle.load(open('lgbm_optimized.pkl', 'rb'))
xgb = pickle.load(open('xgb_model.pkl', 'rb'))

st.write(lgbm_base)

y_pred_xgb = xgb.predict(data_df)
#y_pred_optimized = lgbm_opt.predict(data_df)

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

    lower_number = "{:,.2f}".format(int(y_pred_xgb.mean().numpy()-1.95*yhat.stddev().numpy()))
    higher_number = "{:,.2f}".format(int(y_pred_xgb.mean().numpy()+1.95*yhat.stddev().numpy()))

    col1, col2, col3 = st.columns(3)

    

    with col1:
        st.write("")

    with col2:
        st.subheader("USD "+ str(lower_number))
        st.subheader("       AND ")

        st.subheader(" USD "+str(higher_number))


    with col3:
        st.write("")

    

    

  


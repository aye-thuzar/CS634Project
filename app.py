
# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap


np.random.seed(42)

st.markdown("<body style ='color:#E2E0D9;'></body>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: #1B9E91;'>House Price Prediction in Ames, Iowa</h4>", unsafe_allow_html=True)

st.write("If you want to know the numbers that you picked for some of the features such as Overall Quality, Sale Conditions etc., please check the following link")
st.write("[link to the categorical encoding](https://github.com/aye-thuzar/CS634Project/edit/main/docs.md)")



#setting up the sliders and getting the input from the sliders

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
 800.0,
 500.0,
 334.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0
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

#st.write("Normalized input data")
#st.write(data_df)

# load trained models
lgbm_base = pickle.load(open('lgbm_base.pkl', 'rb'))
lgbm_opt = pickle.load(open('lgbm_optimized.pkl', 'rb'))
xgb_base = pickle.load(open('xgb_base.pkl', 'rb'))
xgb_opt = pickle.load(open('xgb_optimized.pkl', 'rb'))



y_pred = xgb_base.predict(data_df)
y_pred_optimized = xgb_opt.predict(data_df)

explainer_base = shap.TreeExplainer(xgb_base)
shap_interaction_base = explainer_base.shap_interaction_values(X_train)
# Get SHAP values
shap_values_base = explainer_base(data_df)

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

    base_model_prediction = "{:,.2f}".format(y_pred[0])
    optimized_model_prediction = "{:,.2f}".format(y_pred_optimized[0])

    st.write("")

    result1 = "Base model's prediciton: $" + str(base_model_prediction)
    html_str = f"""<style>p.a {{font: bold {28}px Courier;color:#1D5D9B;}}</style><p class="a">{result1}</p>"""   
    st.markdown(html_str, unsafe_allow_html=True)

    st.subheader("SHAP Summary Plot")
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    shap.plots.beeswarm(shap_explainer_base, max_display=10)
    st.markdown("</div>", unsafe_allow_html=True)
    st.pyplot()

    result2 = "Optimized model's prediciton: $" + str(optimized_model_prediction)
    html_str2 = f"""<style>p.a {{font: bold {28}px Courier;color:#1D5D9B;}}</style><p class="a">{result2}</p>"""
    st.markdown(html_str2, unsafe_allow_html=True)


    

    

  


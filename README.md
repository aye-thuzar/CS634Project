---
title: HousePricePredictionApp
emoji: 🏠
colorFrom: pink
colorTo: yellow
sdk: streamlit
sdk_version: 1.21.0
app_file: app.py
pinned: false
---

# CS634Project

Milestone-3 notebook: https://github.com/aye-thuzar/CS634Project/blob/main/CS634Project_Milestone3_Final_AyeThuzar.ipynb

Hugging Face App: https://huggingface.co/spaces/ayethuzar/HousePricePredictionApp

Landing Page for the App:

App Demonstration Video: 

***********

Results

***********

XGBoost Model's RMSE: 28986  (Milestone-2)

Optuna optimized XGBoost's RMSE: 28047

Baseline LGBM's RMSE: 34110

Optuna optimized LGBM's RMSE: 28329

***********

## Documentation 

***********

Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

**Data Processing and Feature Selection:**

For the feature selection, I started by dropping columns with a low correlation (< 0.4) with SalePrice. I then dropped columns with low variances (< 1). After that, I checked the correlation matrix between columns to drop selected columns that have a correlation greater than 0.5 but with consideration for domain knowledge. After that, I checked for NAs in the numerical columns. Then, based on the result, I used domain knowledge to fill the NAs with appropriate values. In this case, I used 0 to fill the NAs as it was the most relevant value. As for the categorical NAs, they were replaced with ‘None’. Once, all the NAs were taken care of, I used LabelEncoder to encode the categorical values. I, then, checked for a correlation between columns and dropped them based on domain knowledge.

Here are the 10 features I selected:

 'OverallQual': Overall material and finish quality
 
 'YearBuilt': Original construction date
 
 'TotalBsmtSF': Total square feet of basement area
 
 'GrLivArea': Above grade (ground) living area square feet
 
 'MasVnrArea': Masonry veneer area in square feet
 
 'BsmtFinType1': Quality of basement finished area
 
 'Neighborhood': Physical locations within Ames city limits
 
 'GarageType': Garage location
 
 'SaleCondition': Condition of sale
 
 'BsmtExposure': Walkout or garden-level basement walls

All the attributes are encoded and normalized before splitting into train and test with 80% train and 20% test.

**Milestone 2:**

For milestone 2, I used an XGBoost Model with objective="reg:squarederror" and max_depth=3. The RMSE score is 28986.

**Milestone 3:**

For milestone 3, I used a light gradient boosting machine (LGBM) with default parameters for baseline and hyperparameter-tuned with Optuna for the optimized model. The results are stated at the beginning of my readme file. I also hyperparameter-tuned my milestone-2 XGBoost model.

I tested the pickled models in this notebook: https://github.com/aye-thuzar/CS634Project/blob/main/CS634Project_Milestone3_AyeThuzar_Testing.ipynb

For the sliders of the categorical features in the app, the numbers and the corresponding meanings are described here: https://github.com/aye-thuzar/CS634Project/edit/main/docs.md

**Milestone 4:**

Please see Milestone4Documentation.md: https://github.com/aye-thuzar/CS634Project/blob/main/Milestone4Documentation.md

Here is the landing page for my app: 

**References:**

https://towardsdatascience.com/analysing-interactions-with-shap-8c4a2bc11c2a

https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454

https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c

https://github.com/adhok/streamlit_ames_housing_price_prediction_app/tree/main

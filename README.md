---
title: LightGBM House Sale Price Prediction
emoji: 🏠
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.21.0
app_file: app.py
pinned: false
---

# CS634Project

Milestone-3 notebook: [[https://colab.research.google.com/drive/17-7A0RkGcwqcJw0IcSvkniDmhbn5SuXe]](https://github.com/aye-thuzar/CS634Project/blob/milestone-3/CS634Project_Milestone3_AyeThuzar.ipynb)(https://colab.research.google.com/drive/1BeoZ4Dxhgd6OcUwPhk6rKCeFnDFMUCmt#scrollTo=TZ4Ci-YXOSl6)

Hugging Face App: 

App Demonstration Video: 

***********

Results

***********

XGBoost Model's RMSE: 28986  (Milestone-2)

Baseline LGBM's RMSE: 26233

Optuna optimized LGBM's RMSE: 13799.282803291926

***********

Hyperparameter Tuning with Optuna

************

Total number of trials:  120

Best RMSE score on validation data: 12338.665498601415

**Best params:**

boosting_type :	 goss

reg_alpha :	 3.9731274536451826

reg_lambda :	 0.8825276525195174

colsample_bytree :	 1.0

subsample :	 1.0

learning_rate :	 0.05

max_depth :	 6

num_leaves :	 48

min_child_samples :	 1

***********

## Documentation for Milestone 4

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

For milestone 3, I used light gradient boosting machine (LGBM) with default parameters for baseline and hyperparameter-tuned with Optuna for the optimized model. The results are stated at the beginning of my readme file.

**References:**

https://towardsdatascience.com/analysing-interactions-with-shap-8c4a2bc11c2a

https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454

https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c

https://github.com/adhok/streamlit_ames_housing_price_prediction_app/tree/main

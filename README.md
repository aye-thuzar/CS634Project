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

Landing Page for the App: https://sites.google.com/view/cs634-realestatehousepricepred/home

App Demonstration Video: https://www.youtube.com/watch?v=jYB1xpeikYQ&t=13s

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

Here is the landing page for my app: https://sites.google.com/view/cs634-realestatehousepricepred/home

**Notebook:** 

This Python notebook predicts house prices using feature selection and XGBoost, a popular gradient boosting algorithm. The dataset used for this prediction is taken from a CSV file containing information about various features of houses, such as lot size, number of rooms, neighborhood, and other related attributes. The main objective of this notebook is to create a machine learning model that can accurately predict house prices based on the given features.

***Data Preprocessing and Feature Selection:***

The first step in the notebook involves data preprocessing and feature selection. Missing values are handled, and the data is cleaned and prepared for modeling. Feature selection techniques are applied to identify the most relevant features for predicting house prices. By selecting the most informative features, the model's accuracy is expected to improve significantly.

***Building the XGBoost Regression Model:***

Next, the focus shifts to building an XGBoost regression model to predict house prices. The data preprocessing and feature selection steps discussed earlier have been incorporated into the XGBoost model training process. This ensures that the model is trained on the most important features, enhancing its predictive capabilities.

***Model Evaluation and SHAP Analysis:***

The performance of the XGBoost model is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). Additionally, SHAP (SHapley Additive exPlanations) values are computed to understand the feature importance and contributions to the model's predictions. SHAP values provide valuable insights into the factors driving the model's decisions.

***Hyperparameter Tuning with Optuna:***

To further improve the XGBoost model's performance, hyperparameter tuning is performed using Optuna, an optimization framework for hyperparameter tuning. Hyperparameter tuning is a crucial step in finding the best set of hyperparameters for the XGBoost model, resulting in better predictions.

***Comparison with Light Gradient Boosting Machine (LGBM) Model:***

The notebook also introduces the Light Gradient Boosting Machine (LGBM) as a baseline model for house price prediction. The performance of both the XGBoost and LGBM models is evaluated using MAE, MSE, and RMSE scores. Additionally, SHAP analysis is performed for the LGBM model to gain insights into its feature importance and decision-making process.

***Insights from SHAP Analysis:***

The SHAP analysis highlights features that significantly impact house prices in both the XGBoost and LGBM models. Key features such as Overall Quality, Above Ground Living Area, and Total Basement Square Foot are identified as significant contributors to the model's predictions. These insights can help potential buyers and sellers make informed decisions in the real estate market.

***Hyperparameter Tuning for LGBM:***

Similar to the XGBoost model, Optuna is used to perform hyperparameter tuning for the LGBM model. The objective function is defined, and Optuna efficiently searches the hyperparameter space to find the best combination of hyperparameters that minimize the RMSE.

***Comparison and Conclusion:***

The notebook concludes with a comparison of the performance of the optimized XGBoost and LGBM models. Both models show promising results, but the LGBM model outperforms the optimized XGBoost model in this scenario. The SHAP analysis provides valuable insights into the decision-making process of both models, enhancing their trustworthiness and interpretability.

***Final Deployment:***

The pickled models, both the optimized XGBoost and LGBM models, are now ready to be deployed in a Streamlit web application for real-world use. These models can be used to make accurate predictions on new data, assisting stakeholders in making informed decisions based on the house price predictions.

***Summary: ***

This Python notebook demonstrates an approach to predicting house prices using feature selection techniques and the XGBoost algorithm. The optimized models achieved better performance compared to the baseline models, showcasing the effectiveness of hyperparameter tuning in improving model accuracy. The SHAP analysis enhances the interpretability and transparency of the models, providing valuable insights into the factors that affect house prices. By comparing and analyzing the performance of different models, data scientists and researchers can make informed choices to build accurate and reliable models for house price prediction tasks.

**References:**

https://towardsdatascience.com/analysing-interactions-with-shap-8c4a2bc11c2a

https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454

https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c

https://github.com/adhok/streamlit_ames_housing_price_prediction_app/tree/main

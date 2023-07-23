# Milestone4 Documentation

**Objective: Create a streamlit app with a baseline and an optimized model with Optuna to predict house prices in Ames, Iowa.**

I use the following dataset to train the models:

Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data


*************

## Deliverables

Github: https://github.com/aye-thuzar/CS634Project

Landing Page for the App: https://sites.google.com/view/cs634-realestatehousepricepred/home

Streamlit App: https://huggingface.co/spaces/ayethuzar/HousePricePredictionApp

Video demonstration of the Streamlit App: https://www.youtube.com/watch?v=jYB1xpeikYQ&t=13s

*************

## Data Processing and Feature Selection

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

## XGBoost Model (baseline)
```py
xgb_model = xgb.XGBRegressor(objective="reg:squarederror",max_depth=3)
xgb_model.fit(X_train, y_train)
```

## SHAP for XGBoost baseline

## Tuning XGBoostWIthOptuna

## Optimized XGBoost

## SHAP for Optimized XGBoost 

## XGBoost Model (baseline)

## SHAP for XGBoost baseline

## Tuning XGBoostWIthOptuna

## Optimized XGBoost

## SHAP for Optimized XGBoost 

## Pickled the models for streamlit app

Finally, the trained models are saved using pickle to be used with Streamlit for deployment. Pickling allows the models to be easily loaded and utilized in real-world applications, making the model predictions readily available for end-users.

*************

References:

https://towardsdatascience.com/analysing-interactions-with-shap-8c4a2bc11c2a

https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454

https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://practicaldatascience.co.uk/machine-learning/how-to-tune-an-xgbregressor-model-with-optuna

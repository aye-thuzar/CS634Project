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
The XGBoost model is initialized using the xgb.XGBRegressor class from the xgboost library. Hyperparameters such as the objective function, maximum depth of the trees, and the number of boosting rounds are set. The model is then trained on the training data (X) and the target variable (y) using the fit() method.

```py
xgb_model = xgb.XGBRegressor(objective="reg:squarederror",max_depth=3)
xgb_model.fit(X_train, y_train)
```

The model performance is tested for MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and average of y_test.

```py
xgbt_pred = xgb_model.predict(X_test)
print("MAE test score:", int(mean_absolute_error(y_test, xgbt_pred)))
print("MSE test score:", int(mean_squared_error(y_test, xgbt_pred)))
print("RMSE test score:", int(sqrt(mean_squared_error(y_test, xgbt_pred))))
y_test.mean()
```

The results were as follows:
- MAE test score: 18490
- MSE test score: 840217398
- RMSE test score: 28986
- y_test.mean: 181370

## SHAP for XGBoost baseline

<p align="center">
<img src="/img/XGBoost_SHAP_summary.png">
</p>

This summary plot visualises all of the SHAP values. On the y-axis, the values are grouped by feature and higher feature values are redder. This plot highlights important relationships: for example, for the Overall Quality and Above grade (ground) living area square feet, as the feature value increases the SHAP values increase. But for the Basement Exposure, which refers to walkout or garden level walls, has the opposite relationship. From these Beeswarm plots, we can also see where the high density SHAP values are because the points are vertically stacked.

<p align="center">
<img src="/img/XGBoost_SHAP_summary_interaction.png">
</p>

This summary plot gives additional insight through visualizing the relationship between features and their SHAP interaction values. As we can see, certain features tend to have a more significiant impact on the prediction, and the distributions of the plots tell us which interactions are more significant than others. For example, Overall Quality, Above Ground Living Area, Total Basement Square Foot, and Neighborhood.

## XGBoostWithOptuna

```py
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'random_state': trial.suggest_int('random_state', 1, 1000)
    }
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)
```

Optuna is used to perform hyperparameter tuning for the XGBoost model. The objective function is defined, which takes a set of hyperparameters as input and returns the MSE as the evaluation metric to minimize. Optuna then searches the hyperparameter space to find the best combination of hyperparameters that result in the lowest MSE.

## Optimized XGBoost

```py
xgb_optimized = xgb.XGBRegressor(**study.best_params)
xgb_optimized.fit(X_train, y_train)
y_pred = xgb_optimized.predict(X_test)
```

After hyperparameter tuning, the best set of hyperparameters found by Optuna is used to create an optimized XGBoost model (xgb_optimized). This model is expected to perform better than the initial XGBoost model due to the fine-tuned hyperparameters.

## SHAP for Tuned Optimized XGBoost

<p align="center">
<img src="/img/XGBoostOptimized_SHAP_summary.png">
</p>

The optimized summary plot gives a very similar results to the baseline XGBoost. However, the feature value order changed with GrLivArea now taking the top spot. YearBuilt and BsmtExposure also climbed one spot each. The density of each plot also seems a bit more spreaded out with distinct disconnect between density areas.

<p align="center">
<img src="/img/XGBoostOptimized_SHAP_summary_interaction.png">
</p>

This optimized interaction plot also has different feature value order as above. It also has less outliers and range is expanded. However we also see the density of SHAP values being grouped at certain CHAP values.

## LGBM

```py
reg_lgbm_baseline = lgbm.LGBMRegressor()  # default - 'regression'
reg_lgbm_baseline.fit(X_train, y_train)
lgbm_predict = reg_lgbm_baseline.predict(X_test)
```

LGBM is known for its fast processing and performance, making it an excellent candidate for comparison with XGBoost. The baseline LGBM model (reg_lgbm_baseline) is trained on the training data (X_train and y_train), and its performance is evaluated using MAE, MSE, and RMSE scores. I did XGBoost for milestone-2 and switch to LGBMRegressor for milestone-3 and the baseline model is already better than the XGBoost, with RMSE = 26233.

## SHAP for LGBM

<p align="center">
<img src="/img/LGBM_SHAP_summary.png">
</p>

The LGBM baseline plot's feature order is the same as XGBoost baseline. However, the outliers in the plot are no longer present. The SHAP value range is now more compact. There is also a change in the density shapes of the plots which can be accounted for the more compact SHAP range.

<p align="center">
<img src="/img/LGBM_SHAP_summary_interaction.png">
</p>

The LGBM baseline interaction plot reverts to the baseline feature order while the density is expanded. It also features more distinct areas of density of SHAP values.

## LGBM with Optuna

```py
def objective(trial, data=X,target=y):

    params = {
                'metric': 'rmse',
                'random_state': 22,
                'n_estimators': 20000,
                'boosting_type': trial.suggest_categorical("boosting_type", ["gbdt", "goss"]),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.85, 1.0]),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]),
                'max_depth': trial.suggest_int('max_depth', 2, 12, step=1),
                'num_leaves' : trial.suggest_int('num_leaves', 13, 148, step=5),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 96, step=5),
            }
    reg = lgbm.LGBMRegressor(**params)
    reg.fit(X_train ,y_train,
            eval_set=[(X_test, y_test)],
            #categorical_feature=cat_indices,
            callbacks=[log_evaluation(period=1000),
                       early_stopping(stopping_rounds=50)
                      ],
           )

    y_pred = reg.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return rmse
```

Similar to XGBoost, Optuna is utilized to perform hyperparameter tuning for the LGBM model. Optuna searches for the best combination of hyperparameters that minimize the RMSE on the validation data. The tuned LGBM model is expected to improve the baseline performance.

## SHAP for LGBM tuned with Optuna

<p align="center">
<img src="/img/LGBMTuned_SHAP_summary.png">
</p>

The SHAP summary plot for tuned LGBM introduces Neighborhood into the top 10 features, while dropping BsmtExposure. It also reverse the positions of MasVnrArea and SaleCondition. The range of SHAP values increased and outliers are present again. The density shapes are different from the XGBoost models.

<p align="center">
<img src="/img/LGBMTuned_SHAP_summary_interaction.png">
</p>

Tuned LGBM SHAP summary interaction also reintroduces outliers while maintaining an expanded range. The density of SHAP values also differ from the XGBoost models.

## Pickled the models for streamlit app

Finally, the trained models are saved using pickle to be used with Streamlit for deployment. Pickling allows the models to be easily loaded and utilized in real-world applications, making the model predictions readily available for end-users.

```py
# Save LGBM baseline model
pickle.dump(reg_lgbm_baseline, open('lgbm_base.pkl', 'wb'))

# Save LightGBM model optimized with Optuna
pickle.dump(lgbmreg_optimized, open('lgbm_optimized.pkl', 'wb'))

# Save XGBoost baseline model
pickle.dump(xgb_model, open('xgb_base.pkl', 'wb'))

# Save XGBoost model optimized with Optuna
pickle.dump(xgb_optimized, open('xgb_optimized.pkl', 'wb'))
```

*************

References:

https://towardsdatascience.com/analysing-interactions-with-shap-8c4a2bc11c2a

https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454

https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://practicaldatascience.co.uk/machine-learning/how-to-tune-an-xgbregressor-model-with-optuna

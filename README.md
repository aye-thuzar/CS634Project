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

## Notebook:

Real estate pricing is a complex and crucial task that heavily relies on various factors influencing property values. In recent years, machine learning models have emerged as powerful tools for predicting house prices, allowing stakeholders to make informed decisions in the real estate market. This Python notebook presents a comprehensive approach to predicting house prices using feature selection techniques, XGBoost, and Light Gradient Boosting Machine (LGBM) models. By employing cutting-edge machine learning algorithms and interpretability techniques, this notebook aims to build accurate, reliable, and transparent models for house price prediction.

### Libraries Used:

Before delving into the implementation, the notebook begins by importing essential Python libraries, each serving a specific purpose throughout the analysis and model development:

1. **shap:** A library for explaining machine learning models. SHAP values provide insights into how features contribute to model predictions, enhancing interpretability.
 
2. **sklearn:** The popular machine learning library with tools for classification, regression, clustering, and more. It provides utilities for data preprocessing, model evaluation, and train-test splitting.
  
3. **optuna:** An optimization framework for hyperparameter tuning. Optuna efficiently searches the hyperparameter space to find the best set of hyperparameters for the models.
  
4. **math, numpy, and pandas:** Basic numerical and data manipulation libraries used for mathematical operations and data handling.
  
5. **matplotlib and seaborn:** Libraries for data visualization, creating insightful plots and charts for better understanding of the data.
   
6. **graphviz:** A library for visualizing decision trees. It helps in understanding the individual trees in the ensemble models.
   
7. **xgboost and lightgbm:** Libraries for gradient boosting algorithms. XGBoost and LGBM are known for their excellent performance in regression tasks like house price prediction.

8. **pickle:** A library for saving and loading Python objects. This is utilized for storing trained models for later use.

### Data Processing and Feature Selection:

The notebook proceeds with comprehensive data processing and feature selection steps to prepare the data for training the machine learning models. The following steps are performed:

1. **Importing Data:** The dataset, containing information about houses and their attributes, is imported using the pd.read_csv() function. The data is then divided into the training set (dataset) and test set (testset) for model evaluation.

2. **Exploratory Data Analysis:** An initial exploration of the training dataset is conducted using the info() function, providing insights into the dataset's structure and missing values.

3. **Setting the Target Variable:** The target variable, SalePrice, is separated from the training set and stored in a separate numpy array y. This is the variable we want the models to predict.

4. **Feature Selection based on Correlation:** Columns with low correlation (< 0.4) with the target variable are dropped from both the training and test sets. Correlation values are calculated using the corr() function.

5. **Feature Selection based on Variance:** Columns with low variance (< 1) are dropped from both the training and test sets. Variance values are calculated using the var() function.

6. **Feature Selection based on High Correlation:** Columns with high correlation (> 0.5) with other features are dropped from both the training and test sets. These columns are identified using the correlation matrix and the corr() function.

7. **Handling Missing Data:** Missing values in numerical columns (numerical) are filled with the value 0 based on domain knowledge. Missing values in categorical columns (categorical) are filled with the string 'None'.

8. **Label Encoding:** Categorical data is encoded using the LabelEncoder from sklearn.preprocessing. This step converts categorical data into numerical format, making it suitable for model training.

9. **Final Feature Selection using Decision Trees (Random Forest):** The notebook employs a Random Forest Regressor to identify the top 10 features that contribute most to predicting the target variable. The least important features are dropped from both the training and test sets.

10. **Normalizing Data:** Finally, the data is normalized using Min-Max scaling to bring all feature values within the range of 0 to 1. This ensures that features with different scales do not dominate the model training process.

### Model Training and Evaluation:

With the data fully processed and features selected, the training data (X) is prepared and used to train the XGBoost and LGBM models. The following steps are performed:

1. **XGBoost Model Training:** The XGBoost model is initialized using the xgb.XGBRegressor class from the xgboost library. Hyperparameters such as the objective function, maximum depth of the trees, and the number of boosting rounds are set. The model is then trained on the training data (X) and the target variable (y) using the fit() method.

2. **Feature Importances:** The feature importances are computed using the trained XGBoost model. The top 10 features that contribute most to the prediction are visualized using a bar plot. This provides valuable insights into which features play a significant role in determining house prices.

3. **Prediction on Test Data:** The trained XGBoost model is used to predict house prices for the test set (testset). The predictions are saved for later comparison and evaluation.

4. **Data Splitting for Testing:** Before training the XGBoost model, the training data (X) is split into training and testing sets using the train_test_split() function from sklearn.model_selection. The testing set will be used to evaluate the model's performance.

5. **Model Evaluation:** The performance of the XGBoost model is evaluated using the testing data (X_test). The Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) are computed and printed to evaluate the model's accuracy in predicting house prices.

### SHAP (SHapley Additive exPlanations) Analysis for XGBoost:

To gain a deeper understanding of the XGBoost model's predictions, SHAP values are computed. SHAP values provide insights into how each feature contributes to the model's predictions for individual data points. The following SHAP plots are generated to visualize the feature contributions and interactions:

1. **Waterfall Plot:** A waterfall plot is created for the first observation in the training data. It shows how each feature contributes to the difference between the predicted price and the expected value. This plot helps identify the most influential features for a particular prediction.

2. **Mean SHAP Value Plot:** This plot displays the mean SHAP values across all observations instead of positive and negative offsets. It helps identify the most important features in the model's predictions.

3. **Summary Plot:** The summary plot visualizes all SHAP values for each feature. It groups the values by feature and represents higher feature values in redder shades. This plot highlights important relationships between features and their impact on the predictions.

4. **Summary Plot with Interaction Values:** This summary plot shows the relationship between features and their SHAP interaction values. It provides additional insights into significant feature interactions.

5. **Dependence Plot:** The dependence plot illustrates the relationship between two features, GrLivArea and OverallQual, and their SHAP interaction values. It shows how the predicted price changes as the features' values change. This plot helps understand how individual feature values affect the model's predictions.

### Hyperparameter Tuning with Optuna for XGBoost:

After understanding the initial performance of the XGBoost model, the notebook proceeds with hyperparameter tuning using Optuna. Hyperparameter tuning is a crucial step in improving the model's performance by finding the best set of hyperparameters for the XGBoost model.

1. **Creating the Optuna Study:** An Optuna study is created using optuna.create_study(), and the direction parameter is set to 'minimize' as the objective is to minimize the Mean Squared Error (MSE). The study aims to find the best hyperparameters by exploring the hyperparameter space for a defined number of trials (n_trials).

2. **Hyperparameter Tuning for XGBoost using Optuna:** Optuna is used to perform hyperparameter tuning for the XGBoost model. The objective function is defined, which takes a set of hyperparameters as input and returns the MSE as the evaluation metric to minimize. Optuna then searches the hyperparameter space to find the best combination of hyperparameters that result in the lowest MSE.

3. **Optimized XGBoost Model:** After hyperparameter tuning, the best set of hyperparameters found by Optuna is used to create an optimized XGBoost model (xgb_optimized). This model is expected to perform better than the initial XGBoost model due to the fine-tuned hyperparameters.

4. **XGBoost Model Evaluation:** The performance of the optimized XGBoost model is evaluated using the testing data (X_test). The MAE, MSE, and RMSE scores are calculated and printed to assess the model's improved accuracy and predictions.

#### SHAP (SHapley Additive exPlanations) Analysis for Optimized XGBoost Model:

With the optimized XGBoost model, SHAP analysis is performed once again to gain deeper insights into its predictions and feature importances. The same SHAP plots as before are generated, revealing how the optimized model's predictions differ from the initial model.

#### LGBM Baseline Model:

Moving on, the notebook introduces a baseline model using Light Gradient Boosting Machine (LGBM). LGBM is known for its fast processing and performance, making it an excellent candidate for comparison with XGBoost. The baseline LGBM model (reg_lgbm_baseline) is trained on the training data (X_train and y_train), and its performance is evaluated using MAE, MSE, and RMSE scores.

#### SHAP (SHapley Additive exPlanations) Analysis for LGBM Baseline Model:

With the baseline LGBM model trained, SHAP analysis is conducted to interpret its predictions and understand the feature importances. The SHAP plots showcase how LGBM's predictions differ from XGBoost and which features have the most significant impact on its predictions.

#### Hyperparameter Tuning with Optuna for LGBM:

Similar to XGBoost, Optuna is utilized to perform hyperparameter tuning for the LGBM model. Optuna searches for the best combination of hyperparameters that minimize the RMSE on the validation data. The tuned LGBM model is expected to improve the baseline performance.

#### SHAP (SHapley Additive exPlanations) Analysis for Optimized LGBM Model:

After hyperparameter tuning, SHAP analysis is performed on the optimized LGBM model to gain insights into its predictions. The SHAP plots reveal how the optimized LGBM model's predictions differ from the baseline and how features' importances change.

#### Model Comparison:

The performance of the optimized XGBoost and LGBM models is compared to identify the best-performing model for house price prediction. Metrics like RMSE, MAE, and MSE are compared to evaluate the models' accuracy and reliability.

#### Model Pickling:

Finally, the trained models are saved using pickle to be used with Streamlit for deployment. Pickling allows the models to be easily loaded and utilized in real-world applications, making the model predictions readily available for end-users.

### Conclusion:

In conclusion, this Python notebook presents an approach to predicting house prices using feature selection techniques, XGBoost, and LGBM models. The notebook covers data preprocessing, feature selection, model training, hyperparameter tuning, model evaluation, and SHAP analysis. The SHAP plots provide valuable insights into the models' decision-making processes and highlight the most significant features impacting the predictions. By employing sophisticated machine learning algorithms and interpretability techniques, the notebook delivers accurate and transparent models for house price prediction. The optimized models achieved improved performance compared to the baseline, showcasing the effectiveness of hyperparameter tuning. The models' predictions can empower stakeholders to make informed decisions in the real estate market, benefiting buyers, sellers, and real estate professionals alike. With the models trained, hyperparameters optimized, and SHAP analysis conducted, the pickled models are now ready to be deployed in real-world applications, providing valuable predictions for house prices and enhancing decision-making processes in the real estate industry.

**References:**

https://towardsdatascience.com/analysing-interactions-with-shap-8c4a2bc11c2a

https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454

https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://www.kaggle.com/code/rnepal2/lightgbm-optuna-housing-prices-regression/notebook

https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c

https://github.com/adhok/streamlit_ames_housing_price_prediction_app/tree/main

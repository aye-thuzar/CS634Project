# CS634Project

Milestone-3 notebook: https://colab.research.google.com/drive/17-7A0RkGcwqcJw0IcSvkniDmhbn5SuXe

Results:

XGBoost Model's RMSE: 28986 

Baseline LGBM's RMSE: 26233

Optuna optimized LGBM's RMSE: 13799.282803291926

***********

Totalnumber of trials:  120

Best RMSE score on validation data: 12338.665498601415

------------------------------

Best params:

------------------------------

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

Reference:

https://github.com/adhok/streamlit_ames_housing_price_prediction_app/tree/main

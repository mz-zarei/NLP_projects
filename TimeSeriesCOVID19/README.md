# Comparing time series predictions of COVID-19 deaths

- Task 1: Preprocess the data using pandas to be ready for machine learning, and visualize the data using matplotlib
- Task 2: Create a SARIMAX model, optimize the model hyperparameters, use the model for forecasting future COVID-19 deaths and visualize the results
- Task 3: Create a prophet model and use the model for forecasting future COVID-19 deaths and visualize the results
- Task 4: Create a function that extracts features for training the XGBOOST and a feedforward neural network models
- Task 5: Split time series feature dataset into training and test datasets and perform data normalization
- Task 6: Train an XGBOOST model and a feedforward neural network model, and finally compare the predictions of all the models covered in the project

# SARIMAX model
SARIMAX(Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors) is an updated version of the ARIMA model. ARIMA includes an autoregressive integrated moving average, while SARIMAX includes seasonal effects and eXogenous factors with the autoregressive and moving average component in the model. Therefore, we can say SARIMAX is a seasonal equivalent model like SARIMA and Auto ARIMA.

# Prophet model
Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

# XGBOOST
It is an ensemble of decision trees algorithm where new trees fix errors of those trees that are already part of the model. Trees are added until no further improvements can be made to the model. XGBoost provides a highly efficient implementation of the stochastic gradient boosting algorithm and access to a suite of model hyperparameters designed to provide control over the model training process. XGBoost is designed for classification and regression on tabular datasets, although it can be used for time series forecasting.


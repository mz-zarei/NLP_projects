# Time Series Analysis Projects

- Project 1: Comparing time series forecasting of COVID-19 deaths
  - Task 1: Preprocess the data using pandas to be ready for machine learning, and visualize the data using matplotlib
  - Task 2: Create a SARIMAX model, optimize the model hyperparameters, use the model for forecasting future COVID-19 deaths and visualize the results
  - Task 3: Create a prophet model and use the model for forecasting future COVID-19 deaths and visualize the results
  - Task 4: Create a function that extracts features for training the XGBOOST and a feedforward neural network models
  - Task 5: Split time series feature dataset into training and test datasets and perform data normalization
  - Task 6: Train an XGBOOST model and a feedforward neural network model, and finally compare the predictions of all the models covered in the project

- Project 2: Comparing time series forecasting of pedestrian/cyclists counts
  - Setting up the data set and performing EDA
  - Identifying Stationarity
  - Transform Data to be Stationary
  - Triple Exponential Smoothing
  - Forecasting Using SARIMA model
  - Forcasting using Facebook's Prophet model
  - Forcasting using LSTM/GRU model
  - Forcasting using XGBoost

# Time Series Analysis Courses

- Time series data analysis with Pandas library
  - Understand time series applications for NumPy and Pandas
  - Summarize a dataframe with a datetime index
    - `setindext('date')` for accessing Datetime Components
    - `reindex` and `daterange` for Handling duplicate or missing indices
    - `resample` for upsampling (e.g. moving from Monthly to Annual) and downsampling (e.g. moving from Annual to Monthly)
  - Generate simple time series plots
    - `pct_change()` to get Variable Percent Change
    - `rolling(window_size)` to get rolling mean/STD
    - Time series plots from `statsmodels.graphics.tsaplots`:
      - `plot_acf`: Plot of the Autocorrelation Function
      - `plot_pacf`: Plot of the Partial Autocorrelation Function
      - `month_plot`: Seasonal Plot for Monthly Data
      - `quarter_plot`: Seasonal Plot for Quarterly Data
      
- Decomposition of time series data into three components (Trend, Seasonality, Residual)
  - `seasonal_decompose()`
  
- Assessing stationarity of time series data sets and transformations methods
  - What it means for time series to be stationary.
    - constant mean, constant variance, constant autocorrelation structure, no periodic component
  - Common ways to identify stationarity.
    - run_sequence_plot()
    - calculate statistics for each splitted chunk of data (`np.split()`)
    - Histogram
    - Augmented Dickey-Fuller Test `adfuller`
  - Useful nonstationary-to-stationary transformations.
    - Decomposition models for removing trend/seasonality
    - Log-transformation
    - Removing autocorrelation with differencing by approporiate lag
   
  
- Simple and exponentially weighted moving average smoothing of time series

- Developing Autoregressive-Moving Average (ARMA) models
  - A practical understanding of Autoregressive (AR) models.
  - A practical understanding of Moving Average (MA) models.
  - A practical understanding of the Autocorrelation Function (ACF).
  - A practical understanding of the Partial Autocorrelation Function (PACF).
  - Insight into choosing the order q of MA models.
  - Insight into choosing the order p of AR models.
  
- Developing Autoregressive Intergrated Moving Average (ARIMA) using SARIMAX and FaceBook Prophet
  - A practical understanding of ARIMA models
  - Insight into checking fit of model
  - Create forecasts with ARIMA models 
  - A practical understanding of `fbprophet`
  - How to check fit of `fbprophet` model
  - Means of adjusting and improving 'fbprophet' model parameters

- Train/Test RNN, LSTM and GRU models for time series forecasting
  - Build and train RNN, LSTM, GRU for time series forecasting, using keras
  - Fine-tune RNN/LSTM parameters
  

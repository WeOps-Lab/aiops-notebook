import nni
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error


# Load data and split into train and test sets
forecast_data = pd.read_csv('D:\\LogReduce\\aiops-notebook\\时序预测\\test_data\\monthly-car-sales.csv', index_col='Month', parse_dates=True)
split_index = int(0.8 * len(forecast_data))
train_data = forecast_data.iloc[:split_index]
test_data = forecast_data.iloc[split_index:]

# Define Holt-Winters model with hyperparameters from NNI
params = nni.get_next_parameter()
alpha = params['alpha']
beta = params['beta']
gamma = params['gamma']

model = ExponentialSmoothing(train_data, trend='mul', seasonal='mul', seasonal_periods=12)
fitted_model = model.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
y_pred = fitted_model.forecast(len(test_data))

# Calculate RMSE
rmse = mean_squared_error(test_data, y_pred, squared=False)

# Report RMSE to NNI
nni.report_final_result(rmse)

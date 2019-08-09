# time series forecasting for Monterey Airport Weather Almanacs
# Jeff Trevino, 2019
from datetime import datetime
from random import seed, random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, coint, arma_order_select_ic
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from fbprophet import Prophet

# pandas settings
pd.set_option('display.max_columns', 125) # csv contains 124 columns
pd.set_option('display.max_rows', 4000) # display more rows
pd.plotting.register_matplotlib_converters()

df = pd.read_csv('cleaned_df.csv', parse_dates=['datetime'], index_col=['datetime'])

x = df
bool_index = (x.index.hour >= 10) & (x.index.hour <= 16) # consider the clearest and driest part of each day
x = x[bool_index]

obsc = x['averageObscuration'].resample(rule='D').mean().dropna()
obsc = obsc - obsc.min() # avoid negative predictions by subtracting the minimum value
hum = x['HourlyRelativeHumidity'].resample(rule='D').mean().dropna()
temp = x['DailyMaximumDryBulbTemperature'].resample(rule='D').last().dropna()

# we need the 2019 data to assess predictions later
obsc_all = obsc[obsc.index.year == 2019]
hum_all = hum[hum.index.year == 2019]
temp_all = temp[temp.index.year == 2019]

sns.set()

bool_index = (obsc.index.year >= 2014) & (obsc.index.year <= 2018)
obsc = obsc[bool_index]
bool_index = (temp.index.year >= 2010) & (temp.index.year <= 2018)
temp = temp[bool_index]
bool_index = (hum.index.year >= 2010) & (hum.index.year <= 2018)
hum = hum[bool_index]

plt.figure(figsize=(12, 6))
plt.plot(obsc, color='b', alpha=0.2)
obsc.rolling(14).mean().plot()
obsc.rolling(14).var().plot(alpha=0.5)
plt.xlabel('date')
plt.ylabel('obscuration')
plt.legend(('daily obscuration', 'rolling 2-week mean', 'rolling 2-week variance'))
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(temp, color='b', alpha=0.2)
temp.rolling(14).mean().plot()
temp.rolling(14).var().plot(alpha=0.5)
plt.xlabel('date')
plt.ylabel('degrees (F)')
plt.legend(('daily max temp (F)', 'rolling 2-week mean', 'rolling 2-week variance'))

plt.figure(figsize=(12, 6))
plt.plot(hum, color='b', alpha=0.2)
hum.rolling(14).mean().plot()
hum.rolling(14).var().plot(alpha=0.5)
plt.xlabel('date')
plt.ylabel('humidity (%)')
plt.legend(('daily max temp (F)', 'rolling 2-week mean', 'rolling 2-week variance'))
plt.show()

# Dickey-Fuller tests for temperature, humidity, obscuration
results = adfuller(obsc)
print("p-value is:", results[1])
results = adfuller(temp)
print("p-value is:", results[1])
# conduct test
results = adfuller(hum)
print("p-value is:", results[1])

# sanity check: does a random walk have a time dependent structure?

# Generate random residuals
np.random.seed(0)
errors = np.random.normal(0, 1, 1000)

# Create AR(1) (random walk) samples for models with and without unit roots
x_unit_root = [0]
x_no_unit_root = [0]
for i in range(len(errors)):
    x_unit_root.append(x_unit_root[-1] + errors[i])
    x_no_unit_root.append(0.9*x_no_unit_root[-1] + errors[i]) # (0.9 isn't 1, so no unit root)

# Calculate Augmented Dickey-Fuller p-values
adfuller(x_unit_root)[1], adfuller(x_no_unit_root)[1] # good: a random walk is non-stationary

# autocorrelation and partial autocorrelation plots
plot_acf(obsc, lags=20)
plt.show()
plot_pacf(obsc, lags=20)
plt.show()

plot_acf(temp, lags=75)
plt.show()
plot_pacf(temp, lags=100)
plt.show()

plot_acf(hum, lags=25)
plt.show()
plot_pacf(hum, lags=25)
plt.show()

for series in [temp, obsc, hum]:
    result = arma_order_select_ic(series)['bic_min_order']
    print(str(series.name), ": ", result)


# ARMA Models
# Fit an ARMA model to the first simulated data
model = ARMA(temp, order=(3,1)) # fit to ARMA model
fitted = model.fit()

# Print out summary information on the fit
print(fitted.summary())

# Print out the estimate for the constant and for phi
print("The estimate of phi (and the constant) are:")
print(fitted.params)

# forecast the past...
cast = fitted.predict(start='01-01-2010', end='12-31-2018')
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(temp, label='truth')
plt.plot(cast, label='ARMA (3,1)')
plt.xlabel('date')
plt.ylabel('temperature (F)')
plt.title('ARMA Tempearture Prediction')
plt.legend()
plt.show()

forecast = fitted.forecast(31)[0]
# forecast Jan 2019 and measure error against observed
# get January of 2019 values
x = df
jan_nineteen = x[x.index.year == 2019]
jan_nineteen = jan_nineteen[(jan_nineteen.index.hour >= 10) & (jan_nineteen.index.hour <= 16)]
jan_nineteen = jan_nineteen['DailyMaximumDryBulbTemperature'].resample(rule='D').last().dropna()
jan_nineteen = jan_nineteen[jan_nineteen.index.month == 1]

# calculate error
rmse = np.sqrt(np.mean(np.square(forecast - jan_nineteen.values)))

# give predicted values a datetime index
index = pd.date_range(start='01-01-2019', end='01-31-2019')
jan_predicted = pd.DataFrame(forecast)
jan_predicted = jan_predicted.set_index(index)

# overlay predicted values with measured values
plt.plot(jan_nineteen, alpha=.4, label='truth, Jan 19')
plt.plot(jan_predicted, label='ARMA (3,1) RMSE: {:0.2f}'.format(rmse))
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('date')
plt.ylabel('temperature (F)')
plt.title('ARMA temperature estimates for January of 2019 average 4.65 degrees of error')
plt.show()

# the same for obscuration
# Fit an ARMA model to the first simulated data
model = ARMA(obsc, order=(1,0)) # fit to ARMA model
fitted = model.fit()

# Print out summary information on the fit
print(fitted.summary())

# Print out the estimate for the constant and for phi
print("The estimate of phi (and the constant) are:")
print(fitted.params)

# forecast the past...
cast = fitted.predict(start='01-01-2014', end='12-31-2018')
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(obsc, label='truth')
plt.plot(cast, label= 'ARMA (1,0)')
plt.xlabel('date')
plt.ylabel('obscuration')
plt.title('ARMA Obscuration Prediction')
plt.legend()
plt.show()

forecast = fitted.forecast(31)[0]

# get January of 2019 values
x = df
jan_nineteen = x[x.index.year == 2019]
jan_nineteen = jan_nineteen[(jan_nineteen.index.hour >= 10) & (jan_nineteen.index.hour <= 16)]
jan_nineteen = jan_nineteen['averageObscuration'].resample(rule='D').last().dropna()
jan_nineteen = jan_nineteen[jan_nineteen.index.month == 1]

# calculate error
rmse = np.sqrt(np.mean(np.square(forecast - jan_nineteen.values)))

# give predicted values a datetime index
index = pd.date_range(start='01-01-2019', end='01-31-2019')
jan_predicted = pd.DataFrame(forecast)
jan_predicted = jan_predicted.set_index(index)

# overlay predicted values with measured values
plt.plot(jan_nineteen, alpha=.4, label='truth, Jan 19')
plt.plot(jan_predicted, label='ARMA (1,0) RMSE: {:0.2f}'.format(rmse))
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('date')
plt.ylabel('obscuration')
plt.title('ARMA obscuration estimates for January of 2019 average error of 3 (nearly 50%)')
plt.show()

# humidity
# Fit an ARMA model to the first simulated data
model = ARMA(hum, order=(2, 2)) # fit to ARMA model
fitted = model.fit()

# Print out summary information on the fit
print(fitted.summary())

# Print out the estimate for the constant and for phi
print("The estimate of phi (and the constant) are:")
print(fitted.params)

# forecast the past...
cast = fitted.predict(start='01-01-2010', end='12-31-2018')
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(hum, label='truth')
plt.plot(cast, label='ARMA (2,2)')
plt.xlabel('date')
plt.ylabel('humidity (%)')
plt.title('ARMA Humidity Prediction')
plt.legend()
plt.show()
plt.show()

forecast = fitted.forecast(31)[0]

# get January of 2019 values
x = df
jan_nineteen = x[x.index.year == 2019]
jan_nineteen = jan_nineteen[(jan_nineteen.index.hour >= 10) & (jan_nineteen.index.hour <= 16)]
jan_nineteen = jan_nineteen['HourlyRelativeHumidity'].resample(rule='D').mean().dropna()
jan_nineteen = jan_nineteen[jan_nineteen.index.month == 1]

# calculate error
rmse = np.sqrt(np.mean(np.square(forecast - jan_nineteen.values)))

# give predicted values a datetime index
index = pd.date_range(start='01-01-2019', end='01-31-2019')
jan_predicted = pd.DataFrame(forecast)
jan_predicted = jan_predicted.set_index(index)

# overlay predicted values with measured values
plt.plot(jan_nineteen, alpha=.4, label='truth, Jan 19')
plt.plot(jan_predicted, label='ARMA (2, 2) RMSE: {:0.2f}'.format(rmse))
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('date')
plt.ylabel('humidity (%)')
plt.title('ARMA humidity estimates for Jan 2019 average 15% error')
plt.show()

# naive season-trend decomposition
results = seasonal_decompose(obsc, model='additive', freq=365)
results.plot()
plt.show()

df_results = adfuller(results.trend.dropna())
print("trend df p-value is:", df_results[1]) # trend isn't stationary (it trends)

df_results = adfuller(results.seasonal.dropna())
print("seasonal df p-value is:", df_results[1]) # seasonality is

df_results = adfuller(results.resid.dropna())
print("residuals p-value is:", df_results[1]) # so are residuals

results = seasonal_decompose(temp, model='additive', freq=365)
results.plot()
plt.show()

df_results = adfuller(results.trend.dropna())
print("trend df p-value is:", df_results[1]) # trend isn't stationary (it trends)

df_results = adfuller(results.seasonal.dropna())
print("seasonal df p-value is:", df_results[1]) # seasonality is

df_results = adfuller(results.resid.dropna())
print("residuals df p-value is:", df_results[1]) # so are residuals

# Holt-Winters Seasonal Smoothing
# separate data into train and test sets
train = temp[:-365]
test = temp.iloc[-365:]
# initialize models
model1 = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365)
model2 = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365, damped=True)
model3 = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=365, damped=True)
# fit models to data
fit1 = model1.fit()
cast1 = fit1.forecast(365)
fit2 = model2.fit()
cast2 = fit2.forecast(365)
fit3 = model3.fit()
cast3 = fit3.forecast(365)
# calculate error
sse1 = np.sqrt(np.mean(np.square(test.values - cast1.values)))
sse2 = np.sqrt(np.mean(np.square(test.values - cast2.values)))
sse3 = np.sqrt(np.mean(np.square(test.values - cast3.values)))
# plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[-365:], train.values[-365:])
ax.plot(test.index, test.values, label='truth',color='b', alpha=.5);
ax.plot(test.index, cast1, color='r', label="add undamped (RMSE={:0.2f}, AIC={:0.2f})".format(sse1, fit1.aic));
ax.legend();
ax.set_xlabel('date')
ax.set_ylabel('degrees (F)')
ax.set_title("Holt-Winter's Seasonal Smoothing Temperature Forecast");
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[-365:], train.values[-365:])
ax.plot(test.index, test.values, label='truth',color='b', alpha=.5);
ax.plot(test.index, cast2, color='g', label="add damped (RMSE={:0.2f}, AIC={:0.2f})".format(sse2, fit2.aic));
ax.legend();
ax.set_xlabel('date')
ax.set_ylabel('degrees (F)')
ax.set_title("Holt-Winter's Seasonal Smoothing Temperature Forecast");
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[-365:], train.values[-365:])
ax.plot(test.index, test.values, label='truth',color='b', alpha=.5);
ax.plot(test.index, cast3, color='black', label="mult damped (RMSE={:0.2f}, AIC={:0.2f})".format(sse3, fit3.aic));
ax.legend();
ax.set_xlabel('date')
ax.set_ylabel('degrees (F)')
ax.set_title("Holt-Winter's Seasonal Smoothing Temperature Forecast");
plt.show()

# separate data into train and test sets
train = obsc.iloc[:-365]
test = obsc.iloc[-365:]
# initialize models
model1 = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365)
model2 = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365, damped=True)

# fit models to data
fit1 = model1.fit()
cast1 = fit1.forecast(365)
cast1 = cast1 - cast1.min()

# failing as all NaNs for unknown reason
# fit2 = model2.fit()
# cast2 = fit2.forecast(365)
# cast2 = cast2 - cast2.min()
# cast2

# calculate error
sse1 = np.sqrt(np.mean(np.square(test.values - cast1.values)))
# sse2 = np.sqrt(np.mean(np.square(test.values - cast2.values))) # fails as NaN

# plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[-365:], train.values[-365:])
ax.plot(test.index, test.values, label='truth',color='b', alpha=.5);
ax.plot(test.index, cast1, color='r', label="add undamped (RMSE={:0.2f}, AIC={:0.2f})".format(sse1, fit1.aic));
ax.legend();
ax.set_xlabel('date')
ax.set_ylabel('obscuration')
ax.set_title("Holt-Winter's Seasonal Smoothing Obscuration Forecast");
plt.show()

# failing
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(train.index[-365:], train.values[-365:])
# ax.plot(test.index, test.values, label='truth',color='b', alpha=.5);
# ax.plot(test.index, cast2, color='g', label="add damped (RMSE={:0.2f}, AIC={:0.2f})".format(sse2, fit2.aic));
# ax.legend();
# ax.set_xlabel('date')
# ax.set_ylabel('obscuration')
# ax.set_title("Holt-Winter's Seasonal Smoothing Obscuration Forecast");
# plt.show()

# separate data into train and test sets
train = hum[:-365]
test = hum.iloc[-365:]

# initialize models
model1 = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365)
model2 = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365, damped=True)
model3 = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=365, damped=True)

# fit models to data
fit1 = model1.fit()
cast1 = fit1.forecast(365)
fit2 = model2.fit()
cast2 = fit2.forecast(365)
fit3 = model3.fit()
# cast3 = fit3.forecast(365) failing as NaNs
# cast3

# calculate error
sse1 = np.sqrt(np.mean(np.square(test.values - cast1.values)))
sse2 = np.sqrt(np.mean(np.square(test.values - cast2.values)))
sse3 = np.sqrt(np.mean(np.square(test.values - cast3.values)))

# plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[-365:], train.values[-365:])
ax.plot(test.index, test.values, label='truth',color='b', alpha=.5);
ax.plot(test.index, cast1, color='r', label="add undamped (RMSE={:0.2f}, AIC={:0.2f})".format(sse1, fit1.aic));
ax.legend();
ax.set_xlabel('date')
ax.set_ylabel('degrees (F)')
ax.set_title("Holt-Winter's Seasonal Smoothing Humidity Forecast");
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[-365:], train.values[-365:])
ax.plot(test.index, test.values, label='truth',color='b', alpha=.5);
ax.plot(test.index, cast2, color='g', label="add damped (RMSE={:0.2f}, AIC={:0.2f})".format(sse2, fit2.aic));
ax.legend();
ax.set_xlabel('date')
ax.set_ylabel('degrees (F)')
ax.set_title("Holt-Winter's Seasonal Smoothing Humidity Forecast");
plt.show()

# failing as NaNs
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(train.index[-365:], train.values[-365:])
# ax.plot(test.index, test.values, label='truth',color='b', alpha=.5);
# ax.plot(test.index, cast3, color='black', label="mult damped (RMSE={:0.2f}, AIC={:0.2f})".format(sse3, fit3.aic));
# ax.legend();
# ax.set_xlabel('date')
# ax.set_ylabel('degrees (F)')
# ax.set_title("Holt-Winter's Seasonal Smoothing Humidity Forecast");
# plt.show()

# calculate index for use in all predictions
forecast_index = pd.date_range(start='01-01-2019', end='12-31-2019')

# set training data
train = temp

# initialize models
temp_model = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=365, damped=True)

# fit models to data
fit_temp = model3.fit()
# temp_cast = fit_temp.forecast(365) forecasting NaNs

# broken
# # plot
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(forecast_index, temp_cast, color='r', label="mult damped".format(sse1, fit1.aic));
# ax.legend();
# ax.set_xlabel('date')
# ax.set_ylabel('degrees (F)')
# ax.set_title("Holt-Winter's Seasonal Smoothing Temperature Forecast");
# plt.show()

# set training data
train = obsc
# initialize model
model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365, damped=True)
# fit models to data
obsc_fit = model.fit()
obsc_cast = obsc_fit.forecast(365)
obsc_cast = obsc_cast - obsc_cast.min()
# plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast_index, obsc_cast, color='r', label="add damped".format(sse1, fit1.aic));
ax.legend();
ax.set_xlabel('date')
ax.set_ylabel('obscuration')
ax.set_title("Holt-Winter's Seasonal Smoothing Obscuration Forecast");
plt.show()

# set the training data
train = hum

# initialize models
model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365, damped=True)

# fit models to data
fit = model.fit()
hum_cast = fit.forecast(365)

# plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast_index, hum_cast, color='r', label="add damped");
ax.legend();
ax.set_xlabel('date')
ax.set_ylabel('humidity (%)')
ax.set_title("Holt-Winter's Seasonal Smoothing Humidity Forecast");
plt.show()

# Facebook Prophet
x = df
bool_index = (x.index.hour >= 10) & (x.index.hour <= 16) # consider the clearest and driest part of each day
x = x[bool_index]

obsc = x['averageObscuration'].resample(rule='D').mean().dropna()
obsc = obsc - obsc.min() # avoid negative predictions by subtracting the minimum value
hum = x['HourlyRelativeHumidity'].resample(rule='D').mean().dropna()
temp = x['DailyMaximumDryBulbTemperature'].resample(rule='D').last().dropna()

# we need the 2019 data to assess predictions later
obsc_all = obsc[obsc.index.year == 2019]
hum_all = hum[hum.index.year == 2019]
temp_all = temp[temp.index.year == 2019]

# seems like obscuration measurement changed around 2014; discard 2010 through 2014
obsc = obsc[obsc.index.year >= 2014]

# make dataframes fbprophet likes
def make_prophet_dataframe_from_series(series):
    frame = pd.DataFrame(series).reset_index()
    frame.columns = ['ds', 'y']
    return frame

    # make prophet frames
fb_temp = make_prophet_dataframe_from_series(temp)
fb_temp.tail()

# fit the model
m = Prophet()
m.fit(fb_temp)

# make future column
future = m.make_future_dataframe(periods=365)
future.tail()

# predict
temp_forecast = m.predict(future)
temp_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# plot predictions
fig1 = m.plot(temp_forecast)

# make prophet frames
fb_obsc = make_prophet_dataframe_from_series(obsc)
fb_obsc.tail()

# fit the model
m = Prophet()
m.fit(fb_obsc)

# make future column
future = m.make_future_dataframe(periods=365)

# predict
obsc_forecast = m.predict(future)

# plot predictions
fig1 = m.plot(obsc_forecast)

# make prophet frames
fb_hum = make_prophet_dataframe_from_series(hum)
fb_hum.tail()

# fit the model
m = Prophet()
m.fit(fb_hum)

# make future column
future = m.make_future_dataframe(periods=365)

# predict
hum_forecast = m.predict(future)

# plot predictions
fig1 = m.plot(hum_forecast)

# plot model components
fig2 = m.plot_components(hum_forecast)

# compare prophet predictions to first three months of 2019
# get 2019 values
temp_nineteen = temp_all[temp_all.index.year == 2019]

# get estimates
temp_guess_df = pd.DataFrame(temp_forecast['yhat'][:90])
i = pd.date_range(start='01/01/19', end='03/31/19')
temp_guess_df = temp_guess_df.set_index(i)

# calculate error
temp_rmse = np.sqrt(np.mean(np.square(temp_guess_df.values - temp_nineteen.values)))
temp_rmse

# overlay predicted values with measured values
plt.plot(temp_nineteen, alpha=.4, label='truth, 2019')
plt.plot(temp_guess_df, label='prophet RMSE: {:0.2f}'.format(temp_rmse))
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('date')
plt.ylabel('temperature (F)')
plt.title('Prophet temperature estimates for Jan 2019 outperform Holt-Winters smoothing')
plt.show()

# get 2019 values
obsc_nineteen = obsc_all[obsc_all.index.year == 2019]
obsc_nineteen += np.min(obsc_nineteen)
len(obsc_nineteen)

# get estimates
guess_df = pd.DataFrame(obsc_forecast['yhat'][:90])
i = pd.date_range(start='01/01/19', end='03/31/19')
guess_df = guess_df.set_index(i)

# calculate error
obsc_rmse = np.sqrt(np.mean(np.square(guess_df.values - obsc_nineteen.values)))
obsc_rmse

# overlay predicted values with measured values
plt.plot(obsc_nineteen, alpha=.4, label='truth, 2019')
plt.plot(guess_df, label='prophet RMSE: {:0.2f}'.format(obsc_rmse))
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('date')
plt.ylabel('obscuration')
plt.title('Prophet obscuration estimates for Jan 2019 outperforms Holt-Winteres smoothing')
plt.show()

# get 2019 values
hum_nineteen = hum_all[temp_all.index.year == 2019]

# get estimates
guess_df = pd.DataFrame(hum_forecast['yhat'][:90])
i = pd.date_range(start='01/01/19', end='03/31/19')
guess_df = guess_df.set_index(i)

# calculate error
rmse = np.sqrt(np.mean(np.square(guess_df.values - hum_nineteen.values)))
rmse

# overlay predicted values with measured values
plt.plot(hum_nineteen, alpha=.4, label='truth, 2019')
plt.plot(guess_df, label='prophet RMSE: {:0.2f}'.format(rmse))
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('date')
plt.ylabel('humidity (%)')
plt.title('Prophet humidity estimates for Jan 2019 perform slightly under Holt-Winters smoothing')
plt.show()

x = obsc_forecast
x = x.set_index('ds')
x.index.name = 'date'
obsc_predictions = x['yhat']

x = temp_forecast
x = x.set_index('ds')
x.index.name = 'date'
temp_predictions = x['yhat']

x = hum_forecast
x = x.set_index('ds')
x.index.name = 'date'
hum_predictions = x['yhat']

nineteen_hat = pd.DataFrame({'temp': temp_predictions, 'hum': hum_predictions, 'obsc': obsc_predictions})
nineteen_hat = nineteen_hat[nineteen_hat.index.year >= 2014]
sns.heatmap(nineteen_hat.isnull(), cbar=False)
nineteen_hat.tail()

nineteen_hat.to_csv('predictions.csv') # export fbrophet predictions

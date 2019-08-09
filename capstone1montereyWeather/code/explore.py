# exploratory visualization code for Monterey Airport Weather Almanacs
# Jeff Trevino, 2019
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_df.csv', parse_dates=['datetime'], index_col=['datetime'])

sns.set()

hourly_obscuration = pd.DataFrame(df.groupby(df.index.hour).averageObscuration.mean())
hourly_obscuration = hourly_obscuration.reset_index()
hourly_obscuration.columns = ['hour of the day', 'mean obscuration']
plt.figure(figsize=(16, 6))
sns.barplot(y='hour of the day', x='mean obscuration', color='grey', orient='h', data=hourly_obscuration).set_title('The Sky Is More Likely to Be Clear Between 10 AM and 4 PM in Monterey')
plt.show()


f, axes = plt.subplots(24,1)
f.set_size_inches(32,128)
axes = axes.flatten()

def add_subplot(master_frame, index):
    hour_frame = hours.loc[index]
    hour_frame = hour_frame.reset_index()
    v = sns.violinplot(y='averageObscuration', x='month', data=hour_frame, ax=axes[index])
    v.set_xlabel("month", fontsize=30)
    v.set_ylabel("obscuration", fontsize=30)
    title_string = str(index) + " o'clock"
    v.set_title(title_string, fontsize=30)

hours = df
hours = pd.DataFrame(df.groupby([df.index.hour, df.index.day, df.index.month]).averageObscuration.mean())
hours.index = hours.index.set_names(['hour', 'day', 'month'])
for x in range(24):
    add_subplot(hours, x)



left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.9    # the top of the subplots of the figure
wspace =  .5     # the amount of width reserved for blank space between subplots
hspace =  1.1    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left    =  left,
    bottom  =  bottom,
    right   =  right,
    top     =  top,
    wspace  =  wspace,
    hspace  =  hspace
)
plt.savefig('figures/hourlyAverageObscurationOverYear.png')
plt.show()

plt.figure(figsize=(12, 6))
sns.set_style('darkgrid')

obscuration = pd.DataFrame(df['averageObscuration'].groupby([df.index.date]).mean().dropna()) # ten years of individual dates
obscuration.name = 'There are more cloudy days than sunny days in the last decade.'

plt.xlim(0, 9)
sns.distplot(obscuration,bins=8, kde=False, norm_hist=False)

plt.xlabel('obscuration rating')
plt.ylabel('days')
plt.show()

plt.figure(figsize=(12, 6))

obscuration = pd.DataFrame(df['averageObscuration'].groupby([df.index.date]).mean().dropna()) # ten years of individual dates
obscuration.name = 'There are more cloudy days than sunny days in the last decade.'
sns.distplot(obscuration,bins=8, kde=False, label='3,400 days throughout the decade')

obscuration_year_averaged_across_decade = pd.DataFrame(df['averageObscuration'].groupby([df.index.month, df.index.day]).mean())
obscuration_year_averaged_across_decade.name = 'Averaging Obscuration Across the Ten Years Compresses the Distribution'
sns.distplot(obscuration_year_averaged_across_decade,bins=8, kde=False, label='365 calendar days averaged across ten years')


# plt.ylim(0, 500)
plt.xlim(0, 9)
plt.legend()
plt.xlabel('obscuration')
plt.ylabel('days')
plt.show()
# plt.savefig('figures/dailyMeanForDatesAcrossDecade.png') # uncomment to write out figure

plt.figure(figsize=(12, 6))
obscuration_year_averaged_across_decade = pd.DataFrame(df['averageObscuration'].groupby([df.index.month, df.index.day]).mean())
obscuration_year_averaged_across_decade.name = 'Averaging Obscuration Across the Ten Years Compresses the Distribution'
sns.distplot(obscuration_year_averaged_across_decade,bins=8, color='orange', kde=False, label='365 calendar days averaged across ten years')


# plt.ylim(0, 500)
plt.xlim(0, 9)
plt.legend()
plt.xlabel('obscuration')
plt.ylabel('calendar days')
plt.show()
# plt.savefig('figures/dailyMeanForDatesAcrossDecade.png') # uncomment to write out figure

by_date = df[(df.index.hour >= 10) & (df.index.hour <= 16)] # get 10 AM to 4 PM
by_date = df.groupby([df.index.month, df.index.day]).averageObscuration.mean() # average by calendar day across decade
by_date = by_date.sort_values()
by_date = by_date[by_date <= 3.5] # 3.5 is a conservative cut-off for a clear day: there are at worst "scattered clouds"
print(str(len(by_date)) + " days have had a decade average obscuration rating of under 3.5.")

by_date = df[(df.index.hour >= 10) & (df.index.hour <= 16)] # get 10 AM to 4 PM
by_date = df.groupby([df.index.month, df.index.day]).averageObscuration.mean()
by_date = by_date.sort_values()
by_date = by_date[by_date <= 3.5] # 3.5 is a conservative cut-off for a clear day: there are at worst "scattered clouds"
by_date = pd.DataFrame(by_date)
by_date.index = by_date.index.rename(["month", "day"])
# by_date = by_date.unstack(level=0)
by_date
by_date.groupby('month').count().rename(columns={'averageObscuration':' number of clearish days'}).plot(kind='bar', title='A Third of January and November Are Clearish Between 10 AM and 4 PM')
plt.show()

by_date = df[(df.index.hour >= 10) & (df.index.hour <= 16)] # get 10 AM to 4 PM
by_date = df.groupby([df.index.month, df.index.day]).averageObscuration.mean()
by_date = by_date.sort_values()
by_date = by_date[by_date <= 3.5] # 3.5 is a conservative cut-off for a clear day: there are at worst "scattered clouds"
by_date = pd.DataFrame(by_date)
by_date.index = by_date.index.rename(["month", "day"])

# set up subplots
f, axes = plt.subplots(2,3, sharey='row')
f.set_size_inches(12,12)
axes = axes.flatten()

# set up title lookup
month_dict = {1: 'January', 2: 'February', 3: 'March', 10: 'October', 11: 'November', 12: 'December'}

# plot a month
def plot_month(frame, month_index, plot_index):
    """Plots the decade mean obscuration for the index month's clearish days"""
    frame = frame.reset_index()
    frame = frame[frame['month'] == month_index]
    frame = frame.set_index('day')
    frame = frame.sort_index()
    b = sns.barplot(data=frame, x=frame.index, color='skyblue', y='averageObscuration', ax=axes[plot_index])
    b.set_xlabel('day of the month')
    b.set_ylabel('decade mean obscuration')
    b.set_title(month_dict[month_index])

for plot, month in enumerate(set(by_date.index.get_level_values(0))):
    plot_month(by_date, month, plot)

plt.suptitle("36 Calendar Days Have A Decade Mean Obscuration of Less Than 3.5 in Monterey")
plt.savefig('figures/clearishDaysByMonth.png')
plt.show()

plt.figure(figsize=(12, 6))

x = df
max_temp = x.groupby(df.index.dayofyear)['DailyMaximumDryBulbTemperature'].mean().rolling(14).mean().plot(label='max')
min_temp = x.groupby(df.index.dayofyear)['DailyMinimumDryBulbTemperature'].mean().rolling(14).mean().plot(label='min')

x['mean_temp'] = (df['DailyMaximumDryBulbTemperature'] + df['DailyMinimumDryBulbTemperature'])/2
mean_temp = x.groupby(df.index.dayofyear)['mean_temp'].mean().rolling(14).mean().plot(label='mean')

# plot
plt.legend()
plt.title('Moving Two-Week Average Annual Temperature')
plt.xlabel('Day of Year')
plt.ylabel('Degrees (F)')
plt.show()

plt.figure(figsize=(12, 6))

x = df
bool_index = (x.index.year >= 2010) & (x.index.year <= 2018)
x = x[bool_index]
x['mean_temp'] = (x['DailyMaximumDryBulbTemperature'] + x['DailyMinimumDryBulbTemperature'])/2

mean_temp = x.groupby([x.index.year, x.index.dayofyear])['mean_temp'].mean().rolling(14).mean()

mean_temp = mean_temp.unstack(level=0)

# plot
for col in mean_temp.columns:
    plt.plot(mean_temp[col], label=str(col))
plt.legend()
plt.title('Moving Two-Week Average of Calendar Day Temperature Exhibits Annual Seasonality')
plt.xlabel('Day of Year')
plt.ylabel('Degrees (F)')
plt.show()

plt.figure(figsize=(12, 12))

# processing setup
x = df
x.reset_index()

# wrangle: year columns, day of year index, daily high temperature values
# mean does nothing here: all entries have same max value
x = x.groupby([x.index.month, x.index.date]).DailyMaximumDryBulbTemperature.last() # all hourly entries have same value
x.index = x.index.rename(['month', 'date'])
x = x.unstack(level=0)
x.head(100)


# # plot
b = sns.boxplot(data=x)
b.set_title('Summer Month Daily Max Temperatures (F) Have Narrow Interquartile Ranges and Few Outliers')
b.set_ylabel('Temperature (F)')
plt.show()

plt.figure(figsize=(12, 12))

# processing setup
x = df
x.reset_index()

# wrangle: year columns, day of year index, daily high temperature values
x = x.groupby([x.index.month, x.index.date]).DailyMinimumDryBulbTemperature.first()
x.index = x.index.rename(['month', 'date'])
x = x.unstack(level=0)
x.head(100)


# # plot
sns.boxplot(data=x).set_title('Summer Months Have a Narrower Range of Minimum Temperatures')
plt.show()

# processing setup
x = df
x.reset_index()
x.head()

# wrangle: create average obscuration and max temperature columns
x = pd.DataFrame(x.groupby([x.index.date, x['DailyMaximumDryBulbTemperature']]).averageObscuration.mean())
x.index = x.index.rename(['date', 'max temp (F)'])
x = x.reset_index()
x = x.set_index(['date'])
x.columns = ['max temp (F)', 'mean obscuration']
x.head()

# plot
hexplot = sns.jointplot(x='max temp (F)', y='mean obscuration', height=10, data=x, kind='hex')
plt.show()

# processing setup
x = df
x.reset_index()
x.head()

# wrangle: create average obscuration and min temperature columns
x = pd.DataFrame(x.groupby([x.index.date, x['DailyMinimumDryBulbTemperature']]).averageObscuration.mean())
x.index = x.index.rename(['date', 'max temp (F)'])
x = x.reset_index()
x = x.set_index(['date'])
x.columns = ['min temp (F)', 'mean obscuration']
x.head()

# plot
hexplot = sns.jointplot(x='min temp (F)', y='mean obscuration', height=10, data=x, kind='hex')
plt.show()

# processing setup
x = df
x.reset_index()
x.head()

# wrangle: rename wind speed and precipitation columns
x = x.rename(columns={'DailyPrecipitation': "Daily Rain (in)", 'DailyPeakWindSpeed': 'Daily Max Wind Speed (m/s)'})

# plot
sns.set_style('white')
j = sns.jointplot(x='Daily Rain (in)', y='Daily Max Wind Speed (m/s)', height=10, data=x, kind='reg')
plt.show()

x = df[(df.index.year >= 2010) & (df.index.year < 2019)] # choose 2010 through 2018, because 2009 and 2019 are missing some dates
x = x.groupby([x.index.year, x.index.date]).DailyPrecipitation.sum()
x.index = x.index.rename(['year', 'date'])
x = x[x == 0]
x = x.reset_index()
x = x.groupby('year').count().rename(columns={'DailyPrecipitation':'Days Without Rain'}).drop(columns=['date'])
x

# average rainfall
x = df
x = x.groupby([x.index.year, x.index.date])['DailyPrecipitation'].first()
x.index = x.index.rename(['year', 'date'])
x = x.groupby(['year']).sum().mean()
x

x = df
x = x.groupby([x.index.year, x.index.date])['DailyPrecipitation'].first()
x.index = x.index.rename(['Year', 'Date'])
x = pd.DataFrame(x.groupby(['Year']).sum())
x.columns = ['Annual Precipitation in Inches']
x = x.reset_index()

# plot
sns.set_style('darkgrid')
sns.barplot(x='Year', y='Annual Precipitation in Inches', color='skyblue', data=x)
plt.show()

# daily precipitation for 36 clearest days on calendar
plt.figure(figsize=(12, 6))

by_date = df[(df.index.hour >= 10) & (df.index.hour <= 16)] # get 10 AM to 4 PM (see definition of 'clearish' above)

# mean the average daily obscuration and keep the first value for daily precipitaton for each date throughout decade
by_date = df.groupby([df.index.date]).agg({'DailyPrecipitation': 'first', 'averageObscuration': 'mean'})

# further average both obscuration and daily rainfall by calendar day
by_date = df.groupby([df.index.month, df.index.day]).agg({'DailyPrecipitation': 'mean', 'averageObscuration': 'mean'})
by_date.index = by_date.index.rename(['month', 'day'])
by_date.columns = ['mean rain (in)', 'mean obscuration']
by_date = by_date.reset_index()

# filter out clearish days
by_date = by_date[by_date['mean obscuration'] <= 3.5] # 3.5 is a conservative cut-off for a clear day: there are at worst "scattered clouds"

# sort by ascending rainfall
by_date = by_date.sort_values(by='mean rain (in)')

# add a date column to serve as the index
by_date['date'] = by_date.apply(lambda x: date(year=1, month=int(x['month']), day=int(x['day'])), axis=1)
by_date


# plot
b = sns.barplot(x='date', y='mean rain (in)', color='skyblue', data=by_date)
b.set_title('The 36 Clearish Days Ordered by Decade Average Daily Rainfall')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 6))

by_date = df[(df.index.hour >= 10) & (df.index.hour <= 16)] # get 10 AM to 4 PM (see definition of 'clearish' above)

# mean the average daily obscuration and keep the first value for daily precipitaton for each date throughout decade
by_date = df.groupby([df.index.date]).agg({'DailyMaximumDryBulbTemperature': 'first', 'averageObscuration': 'mean'})

# further average both obscuration and daily rainfall by calendar day
by_date = df.groupby([df.index.month, df.index.day]).agg({'DailyMaximumDryBulbTemperature': 'mean', 'averageObscuration': 'mean'})
by_date.index = by_date.index.rename(['month', 'day'])
by_date.columns = ['mean daily max temp (F)', 'mean obscuration']
by_date = by_date.reset_index()

# filter out clearish days
by_date = by_date[by_date['mean obscuration'] <= 3.5] # 3.5 is a conservative cut-off for a clear day: there are at worst "scattered clouds"

# sort by ascending rainfall
by_date = by_date.sort_values(by='mean daily max temp (F)')

# add a date column to serve as the index
by_date['date'] = by_date.apply(lambda x: date(year=1, month=int(x['month']), day=int(x['day'])), axis=1)
by_date


# plot
b = sns.barplot(x='date', y='mean daily max temp (F)', color='skyblue', data=by_date)
b.set_title('The 36 Clearish Days Ordered by Decade Average Daily Max Tempearture')
plt.xticks(rotation=90)
plt.show()

sns.set()
plt.figure(figsize=(12, 6))

x = df
bool_index = (x.index.year >= 2014) & (x.index.year <= 2018) # only these years have data for all days of the year
x = x[bool_index]

mean_obsc = x.groupby([x.index.year, x.index.dayofyear])['averageObscuration'].mean().rolling(14).mean()

mean_obsc_decade = x.groupby([x.index.dayofyear])['averageObscuration'].mean().rolling(14).mean().plot(label='decade average', linestyle='--')

mean_obsc = mean_obsc.unstack(level=0)

# plot
for col in mean_obsc.columns:
    plt.plot(mean_obsc[col], label=str(col), alpha=.4)
plt.legend()
plt.title('Moving Two-Week Average of Sky Obscuration Exhibits Annual Seasonality')
plt.xlabel('Day of Year')
plt.ylabel('Obscuration')
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))
sns.set_style('white')

x = df
bool_index = (x.index.year >= 2014) & (x.index.year <= 2018) # only these years have data for all days of the year
x = x[bool_index]
x = x[['averageObscuration', 'DailyMaximumDryBulbTemperature']]

days = x.groupby([x.index.year, x.index.dayofyear]).mean().rolling(45).mean()
days.index = days.index.rename(['year', 'day'])
days = days.reset_index()
days = days.drop(['year', 'day'], axis=1)

# plot
[ax1.axvline(x, color='g', linestyle='--') for x in [y*365 for y in range(5)]] # show year starts

ax1.plot(days['averageObscuration'], label='mean decade obscuration')
ax1.set_xlabel('Day in 2014-2018 Time Period')
ax1.set_ylabel('Obscuration (45-Day Moving Average)')
ax1.legend(loc='upper left')
ax1.axhline(4, color='r')
ax2 = ax1.twinx() # share x axis, use two separate y axes on left and right sides
ax2.plot(days['DailyMaximumDryBulbTemperature'], label='mean decade daily max temperature (F)', alpha=0.2)
ax2.legend(loc='lower right')
ax2.set_ylabel('Temperature (45-Day Moving Average) (F)')
plt.title('Annual Temperature and Obscuration Seasonalities Roughly Align')
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))

x = df
bool_index = (x.index.year >= 2014) & (x.index.year <= 2018) # only these years have data for all days of the year
x = x[bool_index]
x = x.groupby(x.index.dayofyear).mean()['HourlyRelativeHumidity'].rolling(14).mean()

plt.plot(x, label='hourly humidity (2-week rolling average)')
[plt.axvline(x, linestyle='--', color='g') for x in [y*92 for y in range(1, 4)]]
plt.xlabel('Calendar Day of the Year')
plt.ylabel('decade average % humidity (2-week moving average)')
plt.title('2014-2018 Averaged Annual Humidity Fluctuates between about 72% and 84%')
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))
sns.set_style('whitegrid')

x = df
bool_index = (x.index.year >= 2014) & (x.index.year <= 2018) # only these years have data for all days of the year
x = x[bool_index]
x = x.groupby([x.index.hour]).mean()['HourlyRelativeHumidity']
x.head()

sns.lineplot(data=x)
plt.xticks(range(24))
plt.xlabel('Hour of the Day')
plt.ylabel('% Humidity (Decade Average)')
ax1.xaxis.grid(which="major")
plt.axvline(10, color='r', linestyle='--')
plt.axvline(16, color='r', linestyle='--')
plt.title('Lower Obscuration (10 AM - 4 PM) Correlates with Humidity Below 70%')
plt.show()

# obscuration risk ratio
df_risk = pd.DataFrame()
x = df
# group boolean obscuration by date
x = df[(df.index.hour >= 10) & (df.index.hour <= 16)] # get 10 AM to 4 PM
x = x.groupby(x.index.date).mean() # obscuration averaged by date
x['is_obscured'] = x.averageObscuration >= 4.0 # add yes or no obscuration column
# recover datetime index
x.index = pd.Series(x.index, dtype='datetime64[ns]')

# drop Feb 29
mask = (x.index.day == 29) & (x.index.month == 2)
x = x.loc[~mask]
x.shape

# group yes counts by calendar day
x = x.groupby([x.index.month, x.index.day]).sum()
x.shape

def calculate_relative_obscuration_risk(row):
    i = row.name
    month = i[0]
    day = i[1]
    x_month = x.loc[month]
    x_not = x_month[x_month.index != day] # exclude the day in question from the rest of the month
    month_mean_obscured = x_not.is_obscured.mean() # average obscured days for rest of month
    ratio = row.is_obscured/month_mean_obscured # compare with day in question's obscured days
    return ratio

x['ratio'] = x.apply(calculate_relative_obscuration_risk, axis=1)
for i, month in enumerate(['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'August', 'Sept', 'Oct', 'Nov', 'Dec']):
    plt.figure(figsize=(12, 5))
    month_index = i + 1
    month_frame = x[x.index.get_level_values(0) == month_index]
    sns.barplot(x=month_frame.index, y=month_frame.ratio, data=month_frame, color='skyblue')
    percentiles = np.array([0, 25, 50, 75, 100])
    percentiles_ratio = np.percentile(month_frame['ratio'], percentiles)
    [plt.axhline(x, linestyle='--', color='r') for x in percentiles_ratio]
    plt.xlabel('Date in ' + month)
    plt.xticks(rotation=60)
    plt.ylabel('Relative Obscuration Risk Ratio')
    plt.show()

# write out obscuration risk to csv
x = x.reset_index()
x['ratio'].to_csv('calendar_obscuration_risk.csv')

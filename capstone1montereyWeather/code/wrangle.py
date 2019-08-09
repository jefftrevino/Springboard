# data wrangling code for Monterey Airport Weather Almanacs
# Jeff Trevino, 2019

from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pandas options
pd.set_option('display.max_columns', 125) # csv contains 124 columns
pd.set_option('display.max_rows', 4000) # display more rows

data = pd.read_csv('montereyClimateData.csv')

df = data
columns = ['DATE',
           'HourlySkyConditions',
           'HourlyVisibility',
           'HourlyDryBulbTemperature',
           'HourlyWindSpeed',
           'DailyMaximumDryBulbTemperature',
           'DailyMinimumDryBulbTemperature',
           'DailyPeakWindSpeed',
           'DailyPrecipitation',
           'HourlyRelativeHumidity'
          ]
df = df.loc[:, columns]

def date_val_to_datetime(to_parse):
    to_format = to_parse.split('T')
    return datetime.strptime(to_format[0] + ' ' + to_format[1], '%Y-%m-%d %H:%M:%S')

df['datetime'] = df.loc[:,'DATE'].apply(date_val_to_datetime)

df = df.set_index(['datetime'])

cols = ['HourlyVisibility', # columns to convert
 'HourlyDryBulbTemperature',
 'HourlyWindSpeed',
 'DailyMaximumDryBulbTemperature',
 'DailyMinimumDryBulbTemperature',
 'DailyPeakWindSpeed',
 'DailyPrecipitation',
 'HourlyRelativeHumidity',
       ]

# convert columns by applying to_numeric with error coersion
df.loc[:, cols] = df.loc[:, cols].apply(pd.to_numeric, errors='coerce')

# check for desired result
for c in cols:
    assert df.loc[:, c].dtypes == np.float64
    assert len(df.loc[df[c].astype(str).str[-1].isin(('*', 's'))]) == 0
    # no values have the "suspect" suffix anymore


df.loc[:,['DailyMaximumDryBulbTemperature', 'DailyMinimumDryBulbTemperature', 'DailyPeakWindSpeed', 'DailyPrecipitation']] = df.loc[:,['DailyMaximumDryBulbTemperature', 'DailyMinimumDryBulbTemperature', 'DailyPeakWindSpeed', 'DailyPrecipitation', 'HourlyRelativeHumidity']].bfill()

# column value is a string of a list of codes, 'BKN:07 15 OVC:08 20'
# desired output is a list of tuples, [('BKN', 7, 15), ('OVC', 8, 20)]
# clear days lack a second integer, i.e., 'CLR:00', appending 0 in place of missing value
from collections import namedtuple

SkyCondition = namedtuple('SkyCondition', 'obscuration, vertical_distance') # these will be the dict's values

def list_of_lists_by_n(the_list, n):
    """Yields the next n elements of a list as a sublist"""
    for i in range(0, len(the_list), n):
        yield the_list[i:i + n]

def from_many_to_two(the_string):
    split_at_spaces = the_string.split(' ')
    return list(list_of_lists_by_n(split_at_spaces, 2))

def from_two_to_three(list_of_lists):
    """
    input: ['CAPS:02', '35']
    output: {'CAPS':, SkyCondition(obscuration=02, vertical_distance=35)}
    """
    output = []
    for two_element_list in list_of_lists:
        first_element = two_element_list[0]
        if 2 >= len(first_element):
            return {} # for single trailing ints
        first_element_split = first_element.split(":")
        if 2 > len(two_element_list):
            two_element_list.append(0) # catch CLR days missing following 00
        condition = SkyCondition(int(first_element_split[1]), int(two_element_list[1]))
        output.append({first_element_split[0]: condition})
    return output

def condition_string_to_namedtuple_dict(value):
    """
    Converts string containing several of the following to a list of dictionaries as follows:
    input: "CAPS:03 34"
    output: {'CAPS':, SkyCondition(obscuration=3, vertical_distance=34)}
    """
    if isinstance(value, float): # the only floats are np.nan, which is a float...with a str repr
        return [] # replace NaNs as an empty list
    the_string = value
    list_of_twos = from_many_to_two(the_string)
    return from_two_to_three(list_of_twos)

df['HourlySkyConditions'] = df['HourlySkyConditions'].apply(condition_string_to_namedtuple_dict)

def calculate_average_obscuration(sky_conditions_for_hour):
    """Calculates the mean obscuration for each hour in the dataset"""
    if not sky_conditions_for_hour:
        return np.nan
    else:
        obscurations = [[y.obscuration for x, y in d.items()] for d in sky_conditions_for_hour]
        obscuration_mean = sum([x[0] for x in obscurations]) / len(obscurations) # calculate mean obscuration
        return obscuration_mean


df['averageObscuration'] = df['HourlySkyConditions'].apply(calculate_average_obscuration)

# impute the column mean for all remaining nan values in all numeric columns
x = df
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_cols = x.select_dtypes(include=numerics)

df.fillna({x:np.mean(df[x]) for x in numeric_cols.columns}, inplace=True)

# write out cleaned dataframe to csv
df.to_csv('cleaned_df.csv')

import datetime
import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def load_data(date=None, dType=None, columns=None):
    client = MongoClient('localhost', 27017)
    fitbitDb = client['fitbit']
    # Create a new collection called "fitbitCollection"
    fitbitCollection = fitbitDb['fitbitCollection']

    if dType == 'sleepLevelsData ':
        myquery = {'type': dType, 'data.dateTime': {'$regex': '^{}'.format(date)}}
    else:
        myquery = {'type': dType}
    mydoc = fitbitCollection.find(myquery)
    # df = pd.DataFrame(list(mydoc))
    df = pd.DataFrame(columns=columns)
    for doc in mydoc:
        new = pd.DataFrame.from_dict([doc["data"]])
        df = pd.concat([df, new])
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    return df


st.title('Fitbit Sleep-Activity Insights')

# Bar plot for the number of steps aggregated by day

st.subheader(f'Average number of steps per day')

dType = 'activities-steps'
columns = ['dateTime', 'value']

steps_per_day_df = load_data(dType=dType, columns=columns)

steps_per_day_df['dateTime'] = steps_per_day_df['dateTime'].apply(pd.to_datetime, utc=True)
steps_per_day_df['day'] = steps_per_day_df['dateTime'].dt.day_name()
steps_per_day_df = steps_per_day_df.drop(columns=['dateTime'])

cat_dtype = pd.CategoricalDtype(
    categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)

steps_per_day_df['day'] = steps_per_day_df['day'].astype(cat_dtype)
steps_per_day_df['value'] = pd.to_numeric(steps_per_day_df['value'])
steps_per_day_df = steps_per_day_df.groupby('day').agg({'value': 'mean'}).reset_index(drop=False)
steps_per_day_df['value'] = steps_per_day_df['value'].astype(int).round()
steps_per_day_df = steps_per_day_df.rename(columns={'value': 'Steps', 'day': 'Day'})
steps_per_day_df = steps_per_day_df.sort_values(by='Day')
fig = px.bar(steps_per_day_df, x='Day', y='Steps')
st.plotly_chart(fig)

# Stacked bar chart for the sleep levels duration

st.subheader(f'Average duration of sleep levels per day')

columns = ['dateTime', 'level', 'seconds']
sleep_levels_data = load_data(dType='sleepLevelsData', columns=columns)
sleep_levels_data['dateTime'] = sleep_levels_data['dateTime'].dt.date
sleep_levels_data['dateTime'] = pd.to_datetime(sleep_levels_data['dateTime'])
sleep_data_grouped = sleep_levels_data.groupby(['dateTime', 'level'])['seconds'].sum()
sleep_data_pivot = sleep_data_grouped.unstack()
sleep_data_pivot = sleep_data_pivot[['deep', 'light', 'rem']]
sleep_data_pivot = sleep_data_pivot.dropna()

sleep_data_pivot['deep'] = sleep_data_pivot['deep'].astype(float)
sleep_data_pivot['rem'] = sleep_data_pivot['rem'].astype(float)
sleep_data_pivot['light'] = sleep_data_pivot['light'].astype(float)

sleep_data_pivot['deep'] = sleep_data_pivot['deep'] / 60
sleep_data_pivot['light'] = sleep_data_pivot['light'] / 60
sleep_data_pivot['rem'] = sleep_data_pivot['rem'] / 60

sleep_data_pivot = sleep_data_pivot.reset_index()
sleep_data_pivot['day'] = sleep_data_pivot['dateTime'].dt.day_name()
sleep_data_pivot = sleep_data_pivot.drop(columns=['dateTime'])
cat_dtype = pd.CategoricalDtype(
    categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
sleep_data_pivot['day'] = sleep_data_pivot['day'].astype(cat_dtype)
sleep_data_pivot = sleep_data_pivot.groupby('day').agg({'deep': 'mean', 'rem': 'mean', 'light': 'mean'}).reset_index(
    drop=False)
sleep_data_pivot = sleep_data_pivot.reset_index(drop=True)
sleep_data_pivot.set_index('day', inplace=True)
sleep_data_pivot = sleep_data_pivot.loc[cat_dtype.categories]

fig = go.Figure(data=[
    go.Bar(name='Deep', x=sleep_data_pivot.index, y=sleep_data_pivot['deep']),
    go.Bar(name='REM', x=sleep_data_pivot.index, y=sleep_data_pivot['rem']),
    go.Bar(name='Light', x=sleep_data_pivot.index, y=sleep_data_pivot['light'])
])

fig.update_layout(barmode='stack', yaxis_title='Minutes')
st.plotly_chart(fig)

# Area chart for the activity levels duration

st.subheader(f'Activity status over time')

columns = ['dateTime', 'value']
minutesFairlyActive_df = load_data(dType='activities-minutesFairlyActive', columns=columns)
minutesFairlyActive_df = minutesFairlyActive_df.rename(columns={'value': 'fairly active'})

minutesVeryActive_df = load_data(dType='activities-minutesVeryActive', columns=columns)
minutesVeryActive_df = minutesVeryActive_df.rename(columns={'value': 'very active'})

minutesLightlyActive_df = load_data(dType='activities-minutesLightlyActive', columns=columns)
minutesLightlyActive_df = minutesLightlyActive_df.rename(columns={'value': 'lightly active'})

activity_df = minutesLightlyActive_df.merge(minutesFairlyActive_df, on='dateTime', how='left')
activity_df = activity_df.merge(minutesVeryActive_df, on='dateTime', how='left')

activity_df['lightly active'] = activity_df['lightly active'].astype(int)
activity_df['fairly active'] = activity_df['fairly active'].astype(int)
activity_df['very active'] = activity_df['very active'].astype(int)

activity_df.set_index('dateTime', inplace=True)
activity_df.index.name = None

st.area_chart(data=activity_df)

# Timeseries comparison between the activity and sleep duration

st.subheader(f'Timeseries comparison between the activity and sleep duration')

columns = ['dateTime', 'value']
minutesVeryActive_df = load_data(dType='activities-minutesVeryActive', columns=columns)
minutesVeryActive_df = minutesVeryActive_df.rename(columns={'value': 'very active'})

minutesFairlyActive_df = load_data(dType='activities-minutesFairlyActive', columns=columns)
minutesFairlyActive_df = minutesFairlyActive_df.rename(columns={'value': 'fairly active'})

minutesLightlyActive_df = load_data(dType='activities-minutesLightlyActive', columns=columns)
minutesLightlyActive_df = minutesLightlyActive_df.rename(columns={'value': 'lightly active'})

columns = ['dateTime', 'value']
totalMinutesAsleep_df = load_data(dType='sleep-duration', columns=columns)

sleep_time_activity_corr_df = minutesVeryActive_df.merge(totalMinutesAsleep_df, on='dateTime', how='left')
sleep_time_activity_corr_df = sleep_time_activity_corr_df.merge(minutesFairlyActive_df, on='dateTime', how='left')
sleep_time_activity_corr_df = sleep_time_activity_corr_df.merge(minutesLightlyActive_df, on='dateTime', how='left')
sleep_time_activity_corr_df = sleep_time_activity_corr_df.rename(columns={'value': 'sleep duration'})
sleep_time_activity_corr_df = sleep_time_activity_corr_df.dropna()

sleep_time_activity_corr_df['fairly active'] = sleep_time_activity_corr_df['fairly active'].astype(int)
sleep_time_activity_corr_df['lightly active'] = sleep_time_activity_corr_df['lightly active'].astype(int)
sleep_time_activity_corr_df['very active'] = sleep_time_activity_corr_df['very active'].astype(int)
sleep_time_activity_corr_df['sleep duration'] = sleep_time_activity_corr_df['sleep duration'].astype(float)
sleep_time_activity_corr_df['sleep duration'] = sleep_time_activity_corr_df['sleep duration'] / 60000
sleep_time_activity_corr_df['activity'] = sleep_time_activity_corr_df['very active'] + sleep_time_activity_corr_df[
    'lightly active'] + sleep_time_activity_corr_df['fairly active']

smoothed_ts = sleep_time_activity_corr_df[['dateTime', 'activity', 'sleep duration']]
smoothed_ts.set_index('dateTime', inplace=True)
smoothed_ts.index.name = None

resample_values = ['No resample/smooth', '2d', '3d', '4d']
container = st.container()
row = container.columns([2, 8])
with row[0]:
    resample = st.selectbox('Resample/Smooth', resample_values)

if resample == 'No resample/smooth':
    smoothed_ts_resampled = smoothed_ts[["activity", "sleep duration"]]
else:
    smoothed_ts_resampled = smoothed_ts[["activity", "sleep duration"]].resample(resample).median()

with row[1]:
    st.line_chart(smoothed_ts_resampled)

# 3d bubble graph for Steps|Very Active|Sleep duration
# Currently not in use
# We leverage the bubble_df DataFrame in the next 3d bubble graph though

columns = ['dateTime', 'value']
sleep_duration_df = load_data(dType='sleep-duration', columns=columns)
sleep_duration_df = sleep_duration_df.rename(columns={'value': 'Sleep duration'})
sleep_duration_df['Sleep duration'] = sleep_duration_df['Sleep duration'] / 60000

dType = 'activities-steps'
columns = ['dateTime', 'value']
steps_per_day_df = load_data(dType=dType, columns=columns)
steps_per_day_df = steps_per_day_df.rename(columns={'value': 'Steps'})

bubble_df = steps_per_day_df.merge(minutesVeryActive_df, on='dateTime', how='left')
bubble_df = bubble_df.merge(sleep_duration_df, on='dateTime', how='left')
bubble_df = bubble_df.dropna()

bubble_df['Steps'] = bubble_df['Steps'].astype(int)
bubble_df['very active'] = bubble_df['very active'].astype(int)
bubble_df['Sleep duration'] = bubble_df['Sleep duration'].astype(int)

# 3d bubble graph for sleep levels and coloring based on the number of steps

st.subheader(f'Relationship between the sleep levels duration and number of steps')

columns = ['dateTime', 'minutes']

rem_summary_df = load_data(dType='sleepLevelsSummary-rem', columns=columns)
rem_summary_df = rem_summary_df.drop(columns=['count', 'thirtyDayAvgMinutes'])
rem_summary_df = rem_summary_df.rename(columns={'minutes': 'REM minutes'})

deep_summary_df = load_data(dType='sleepLevelsSummary-deep', columns=columns)
deep_summary_df = deep_summary_df.drop(columns=['count', 'thirtyDayAvgMinutes'])
deep_summary_df = deep_summary_df.rename(columns={'minutes': 'Deep sleep minutes'})

light_summary_df = load_data(dType='sleepLevelsSummary-light', columns=columns)
light_summary_df = light_summary_df.drop(columns=['count', 'thirtyDayAvgMinutes'])
light_summary_df = light_summary_df.rename(columns={'minutes': 'Light sleep minutes'})

level_summary_merged = rem_summary_df.merge(deep_summary_df, on='dateTime', how='left')
level_summary_merged = level_summary_merged.merge(light_summary_df, on='dateTime', how='left')

steps = bubble_df[['dateTime', 'Steps']]
level_summary_merged = level_summary_merged.merge(steps, on='dateTime', how='left')

level_summary_merged['Light sleep minutes'] = level_summary_merged['Light sleep minutes'].astype(int)
level_summary_merged['Deep sleep minutes'] = level_summary_merged['Deep sleep minutes'].astype(int)
level_summary_merged['REM minutes'] = level_summary_merged['REM minutes'].astype(int)
level_summary_merged['Steps'] = level_summary_merged['Steps'].astype(float)

level_summary_merged = level_summary_merged.drop_duplicates()

fig = px.scatter_3d(level_summary_merged, x='Deep sleep minutes', y='Light sleep minutes', z='REM minutes',
                    color='Steps', color_continuous_scale='redor')

st.plotly_chart(fig)

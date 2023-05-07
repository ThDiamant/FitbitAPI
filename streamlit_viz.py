import datetime as dt
import streamlit as st
import pandas as pd
import pymongo as mongo
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

import functions as fun

# Options to be able to see all columns when printing
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# Set streamlit page width
st.markdown(
    """
    <style>
    .main {
        max-width: 1600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

USER_UUID = "3cc4e2ee-8c2f-4c25-955b-fe7f6ffcbe44"
DB_NAME = "fitbit"
DATA_COLLECTION_NAME = "fitbitCollection"
DATE_FORMAT = "%d %B %Y"

# # Connect to MongoDB collection where the data are stored
# fitbitCollection = fun.connect_to_db()
# distinctTypes = fitbitCollection.distinct("type")

START_DATE, END_DATE = fun.get_min_max_dates()
# Number of days considered
NUM_DAYS = (dt.datetime.strptime(END_DATE, DATE_FORMAT) - dt.datetime.strptime(START_DATE, DATE_FORMAT)).days
# print(START_DATE, END_DATE, NUM_DAYS)

# ----------------------------------------------------------------------------------------------------------------------
# Streamlit Dashboard title
st.title('Fitbit Sleep-Activity Insights')

# ----------------------------------------------------------------------------------------------------------------------
# Sleep - Numeric indicators

# Define the slider widget for the numeric indicators
period = st.sidebar.slider(label='Number of days',
                           min_value=1,
                           max_value=NUM_DAYS,
                           value=NUM_DAYS,
                           step=1)

date_range = st.sidebar.slider(
    "When do you start?",
    value=(dt.datetime.strptime(START_DATE, DATE_FORMAT), dt.datetime.strptime(END_DATE, DATE_FORMAT)),
    format="DD/MM/YYYY")
# st.write("Date range:", date_range)

start_date = date_range[0]
end_date = date_range[1]

# ----- Avg sleep duration (total)
def get_avg_sleep_duration(period):
    dType = 'sleep-duration'
    sleep_duration_df = fun.get_df(dType=dType)
    # Filter based on the selected period
    sleep_duration_df['dateTime'] = pd.to_datetime(sleep_duration_df['dateTime'])
    sleep_duration_df = sleep_duration_df.set_index('dateTime')
    sleep_duration_df = sleep_duration_df.loc[(sleep_duration_df.index >= period[0]) & (sleep_duration_df.index <= period[1])]

    # Convert sleep duration from ms to hours
    sleep_duration_df['value'] = sleep_duration_df['value'] / 3600000
    tot_avg_sleep_duration = round(sleep_duration_df['value'].mean(), 1)
    return tot_avg_sleep_duration

tot_avg_sleep_duration = get_avg_sleep_duration(date_range)
print(tot_avg_sleep_duration)


# ----- Sleep start time (most common one)
most_common_hour, nNights = fun.get_most_common_sleep_start_time(period)
print(most_common_hour, nNights, period)

# ----- Avg sleep efficiency
avg_sleep_efficiency = fun.get_avg_sleep_eff(period)

# ----- Avg number of steps
avg_steps = fun.get_avg_steps(period)


# Show the metrics side by side
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label=':sleeping: Avg sleep duration',
              value=f'{tot_avg_sleep_duration} hours')
with col2:
    st.metric(label=f':new_moon_with_face: Most common sleep hour',
              value=f'{most_common_hour} a.m.')
with col3:
    st.metric(label=f':ok_hand: Avg sleep efficiency',
              value=f'{avg_sleep_efficiency} %')
with col4:
    st.metric(label=f':walking: Avg steps',
              value=f'{avg_steps}')

# ----------------------------------------------------------------------------------------------------------------------
# Bar chart - Avg number of minutes in each stage (total)
avg_min_stage_df = fun.avg_num_min_each_stage_ser(period)


# Plot
title = 'Avg time spent in each Sleep Stage'
x_axis_title = 'Minutes'
y_axis_title = 'Sleep Stage'
avg_min_stage_fig = fun.plot_pie_from_ser(avg_min_stage_df, title)

st.plotly_chart(avg_min_stage_fig)

# ----------------------------------------------------------------------------------------------------------------------
# Bar chart - Avg minutes in different activity zones
avg_min_activity_ser = fun.get_avg_min_activity_ser(period)

# Plot
title = 'Avg time spent in each Activity Type'
x_axis_title = 'Minutes'
y_axis_title = 'Activity Type'
avg_min_activity_fig = fun.plot_pie_from_ser(avg_min_activity_ser, title)

st.plotly_chart(avg_min_activity_fig)

# ----------------------------------------------------------------------------------------------------------------------
# Plot time series for sleep and activity level data

start_date = fun.to_date(START_DATE)
end_date = fun.to_date(END_DATE)

# Define date widget
date = st.date_input(
    label=":calendar: Date selection",
    value=start_date)

# Make sure the selected date is within the correct limits
if date < start_date:
    st.write(f":exclamation: Selected date cannot be before {start_date.strftime(DATE_FORMAT)}."
             f"Please select another date.:exclamation:")
elif date > end_date:
    st.write(f":exclamation: Selected date cannot be after {end_date.strftime(DATE_FORMAT)}. "
             f"Please select another date.:exclamation:")
else:
    # Plot sleep levels time series ------------------------------------------------------------------------------
    # Here the functions are kind of used not as they are suppossed to
    # The functions where developed to create a list of dataframes with all the data, so here we only use them for
    # one date.
    dates = [date]
    sleepStartEnd_list = fun.get_sleep_start_end(dates)
    sleepStartEnd = sleepStartEnd_list[0]
    sleepLevelTimeSeries_df = fun.get_sleep_level_timeseries_df(sleepStartEnd)
    # Plot the timeseries
    fig = fun.plot_sleep_level_time_series(sleepLevelTimeSeries_df)
    st.plotly_chart(fig.to_dict())

    # Plot activity level time series ------------------------------------------------------------------------------
    activityDate = date.strftime('%Y-%m-%d')
    fullActivityTimeseries = fun.get_activity_detail_timeseries(activityDate)
    activity_timeseries_df = fun.get_activity_timeseries_df(fullActivityTimeseries)
    fig = fun.plot_activity_level_timeseries(activity_timeseries_df)
    st.plotly_chart(fig.to_dict())


# ----------------------------------------------------------------------------------------------------------------------
# Bar plot for the number of steps aggregated by day
st.subheader(f'Average number of steps per day of week')
# Get data
dType = 'steps'
steps_per_day_df = fun.get_df(dType=dType, addDateTimeCols=True)
# steps_per_day_df['day'] = steps_per_day_df['dateTime'].dt.day_name()
# st.write(steps_per_day_df)
steps_per_day_df = steps_per_day_df.groupby('day_name').agg({'value': 'mean'}).reset_index(drop=False)
steps_per_day_df['value'] = steps_per_day_df['value'].astype(int).round()
steps_per_day_df = steps_per_day_df.rename(columns={'value': 'Steps', 'day_name': 'Day'})
# steps_per_day_df = steps_per_day_df.sort_values(by='Day')
fig = px.bar(steps_per_day_df, x='Day', y='Steps')
st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------------
# Stacked bar chart for the sleep levels duration
st.subheader(f'Average duration of sleep levels per day of week')
# Get data
dType = 'sleepLevelsData-data'
sleep_levels_data = fun.get_df(dType=dType)

sleep_levels_data['dateTime'] = sleep_levels_data['dateTime'].dt.date
sleep_data_grouped = sleep_levels_data.groupby(['dateTime', 'level'])['value'].sum()
sleep_data_pivot = sleep_data_grouped.unstack()
sleep_data_pivot = sleep_data_pivot[['deep', 'light', 'rem', 'wake']]
sleep_data_pivot = sleep_data_pivot.dropna()

sleep_data_pivot['deep'] = sleep_data_pivot['deep'].astype(float)
sleep_data_pivot['rem'] = sleep_data_pivot['rem'].astype(float)
sleep_data_pivot['light'] = sleep_data_pivot['light'].astype(float)
sleep_data_pivot['wake'] = sleep_data_pivot['wake'].astype(float)

sleep_data_pivot['deep'] = sleep_data_pivot['deep'] / 60
sleep_data_pivot['light'] = sleep_data_pivot['light'] / 60
sleep_data_pivot['rem'] = sleep_data_pivot['rem'] / 60
sleep_data_pivot['wake'] = sleep_data_pivot['wake'] / 60

sleep_data_pivot = sleep_data_pivot.reset_index()
sleep_data_pivot['dateTime'] = pd.to_datetime(sleep_data_pivot['dateTime'])
sleep_data_pivot['day'] = sleep_data_pivot['dateTime'].dt.day_name()

sleep_data_pivot = sleep_data_pivot.drop(columns=['dateTime'])
cat_dtype = pd.CategoricalDtype(
    categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
sleep_data_pivot['day'] = sleep_data_pivot['day'].astype(cat_dtype)
sleep_data_pivot = sleep_data_pivot.groupby('day').agg({'deep': 'mean', 'rem': 'mean', 'light': 'mean', 'wake': 'mean'}).reset_index(
    drop=False)
sleep_data_pivot = sleep_data_pivot.reset_index(drop=True)
sleep_data_pivot.set_index('day', inplace=True)
sleep_data_pivot = sleep_data_pivot.loc[cat_dtype.categories]

fig = go.Figure(data=[
    go.Bar(name='Deep', x=sleep_data_pivot.index, y=sleep_data_pivot['deep']),
    go.Bar(name='REM', x=sleep_data_pivot.index, y=sleep_data_pivot['rem']),
    go.Bar(name='Light', x=sleep_data_pivot.index, y=sleep_data_pivot['light']),
    go.Bar(name='Wake', x=sleep_data_pivot.index, y=sleep_data_pivot['wake'])
])

fig.update_layout(barmode='stack', yaxis_title='Minutes')
st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------------
# Area chart for the activity levels duration
st.subheader(f'Activity status over time')

# Get data
dType = 'minutesFairlyActive'
minutesFairlyActive_df = fun.get_df(dType=dType)
minutesFairlyActive_df = minutesFairlyActive_df.rename(columns={'value': 'fairly active'})

dType = 'minutesVeryActive'
minutesVeryActive_df = fun.get_df(dType=dType)
minutesVeryActive_df = minutesVeryActive_df.rename(columns={'value': 'very active'})

dType = 'minutesLightlyActive'
minutesLightlyActive_df = fun.get_df(dType=dType)
minutesLightlyActive_df = minutesLightlyActive_df.rename(columns={'value': 'lightly active'})

# Merge data
activity_df = minutesLightlyActive_df.merge(minutesFairlyActive_df, on='dateTime', how='left')
activity_df = activity_df.merge(minutesVeryActive_df, on='dateTime', how='left')

activity_df['lightly active'] = activity_df['lightly active'].astype(int)
activity_df['fairly active'] = activity_df['fairly active'].astype(int)
activity_df['very active'] = activity_df['very active'].astype(int)

activity_df.set_index('dateTime', inplace=True)
activity_df.index.name = None

# Plot result
st.area_chart(data=activity_df)


# 3d bubble graph for Steps|Very Active|Sleep duration
# Currently not in use
# We leverage the bubble_df DataFrame in the next 3d bubble graph though

dType = 'sleep-duration'
sleep_duration_df = fun.get_df(dType=dType)
sleep_duration_df = sleep_duration_df.rename(columns={'value': 'Sleep duration'})
sleep_duration_df['Sleep duration'] = sleep_duration_df['Sleep duration'] / 60000

dType = 'steps'
steps_per_day_df = fun.get_df(dType=dType)
steps_per_day_df = steps_per_day_df.rename(columns={'value': 'Steps'})

bubble_df = steps_per_day_df.merge(minutesVeryActive_df, on='dateTime', how='left')
bubble_df = bubble_df.merge(sleep_duration_df, on='dateTime', how='left')
bubble_df = bubble_df.dropna()

bubble_df['Steps'] = bubble_df['Steps'].astype(int)
bubble_df['very active'] = bubble_df['very active'].astype(int)
bubble_df['Sleep duration'] = bubble_df['Sleep duration'].astype(int)

# ----------------------------------------------------------------------------------------------------------------------
# 3d bubble graph for sleep levels and coloring based on the number of steps
st.subheader(f'Relationship between the sleep levels duration and number of steps')
dType = "sleepSummary-stages"
sleep_level_summary_df = fun.get_df(dType=dType)

sleep_level_summary_df = sleep_level_summary_df.drop("wake", axis=1)
rename_cols = {
    "deep": "Deep sleep minutes",
    "light": "Light sleep minutes",
    "rem": "REM minutes"
}
sleep_level_summary_df = sleep_level_summary_df.rename(columns=rename_cols)

steps = bubble_df[['dateTime', 'Steps']]
sleep_level_summary_df = sleep_level_summary_df.merge(steps, on='dateTime', how='left')

sleep_level_summary_df['Light sleep minutes'] = sleep_level_summary_df['Light sleep minutes'].astype(int)
sleep_level_summary_df['Deep sleep minutes'] = sleep_level_summary_df['Deep sleep minutes'].astype(int)
sleep_level_summary_df['REM minutes'] = sleep_level_summary_df['REM minutes'].astype(int)
sleep_level_summary_df['Steps'] = sleep_level_summary_df['Steps'].astype(float)

sleep_level_summary_df = sleep_level_summary_df.drop_duplicates().dropna()

fig = px.scatter_3d(sleep_level_summary_df, x='Deep sleep minutes', y='Light sleep minutes', z='REM minutes',
                    color='Steps', color_continuous_scale='redor')

st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------------
st.subheader(f'Correlation Matrix')
correlation_df = fun.heatmpaPlots()
corr_matrix = correlation_df.corr()

# create the heatmap using Plotly
fig = px.imshow(corr_matrix, 
                x=corr_matrix.columns, 
                y=corr_matrix.columns,
                color_continuous_scale='YlGnBu')

st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------------
st.subheader(f'TimeSeries Comparison')
options = st.multiselect(
    'Select Variables',
    ['sleepEfficiency', 'Steps', 'deep', 'light', 'rem', 'wake', 'minutesFairlyActive', 'minutesLightlyActive', 'minutesSedentary', 'minutesVeryActive'],
    ['deep', 'light', 'rem', 'wake'])

df = correlation_df.set_index('dateTime')
# create an instance of the scaler
scaler = MinMaxScaler()

# normalize all columns in the dataframe
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Add the DateTime Column as index
df_normalized['dateTime'] = correlation_df['dateTime']
df_normalized = df_normalized.set_index('dateTime')

resample_values = ['No resample/smooth', '2d', '3d', '4d']
container = st.container()
row = container.columns([2, 8])
with row[0]:
    resample = st.selectbox('Resample/Smooth', resample_values)

if resample == 'No resample/smooth':
    smoothed_ts_resampled = df_normalized[options]
else:
    smoothed_ts_resampled = df_normalized[options].resample(resample).median()

with row[1]:
    #st.line_chart(smoothed_ts_resampled)
    st.line_chart(smoothed_ts_resampled)

# ----------------------------------------------------------------------------------------------------------------------
# Stacked bar chart for the activity stages duration
st.subheader(f'Average duration of activity stages per day of week')
# Get data
minutesFairlyActive_df = fun.get_df(dType="minutesFairlyActive").rename(columns={'value': 'Fairly Active'})
minutesFairlyActive_df['Fairly Active'] = minutesFairlyActive_df['Fairly Active'].astype(int)
minutesLightlyActive_df = fun.get_df(dType="minutesLightlyActive").rename(columns={'value': 'Lightly Active'})
minutesLightlyActive_df['Lightly Active'] = minutesLightlyActive_df['Lightly Active'].astype(int)
minutesSedentary_df = fun.get_df(dType="minutesSedentary").rename(columns={'value': 'Sedentary'})
minutesSedentary_df['Sedentary'] = minutesSedentary_df['Sedentary'].astype(int)
minutesVeryActive_df = fun.get_df(dType="minutesVeryActive").rename(columns={'value': 'Very Active'})
minutesVeryActive_df['Very Active'] = minutesVeryActive_df['Very Active'].astype(int)
df_list = [minutesFairlyActive_df, minutesLightlyActive_df, minutesSedentary_df, minutesVeryActive_df]
activitySummary_stages_df = fun.merge_dataframes(df_list=df_list, common_col="dateTime", how="outer")

activitySummary_stages_df['dateTime'] = activitySummary_stages_df['dateTime'].dt.date
activity_data_pivot = activitySummary_stages_df.dropna()

activity_data_pivot['Fairly Active'] = activity_data_pivot['Fairly Active'].astype(float)
activity_data_pivot['Lightly Active'] = activity_data_pivot['Lightly Active'].astype(float)
activity_data_pivot['Sedentary'] = activity_data_pivot['Sedentary'].astype(float)
activity_data_pivot['Very Active'] = activity_data_pivot['Very Active'].astype(float)

activity_data_pivot['Fairly Active'] = activity_data_pivot['Fairly Active'] / 60
activity_data_pivot['Lightly Active'] = activity_data_pivot['Lightly Active'] / 60
activity_data_pivot['Sedentary'] = activity_data_pivot['Sedentary'] / 60
activity_data_pivot['Very Active'] = activity_data_pivot['Very Active'] / 60


activity_data_pivot = activity_data_pivot.reset_index()
activity_data_pivot['dateTime'] = pd.to_datetime(activity_data_pivot['dateTime'])
activity_data_pivot['day'] = activity_data_pivot['dateTime'].dt.day_name()

activity_data_pivot = activity_data_pivot.drop(columns=['dateTime'])
cat_dtype = pd.CategoricalDtype(
    categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
activity_data_pivot['day'] = activity_data_pivot['day'].astype(cat_dtype)
activity_data_pivot = activity_data_pivot.groupby('day').agg({'Fairly Active': 'mean', 'Lightly Active': 'mean', 'Sedentary': 'mean', 'Very Active': 'mean' }).reset_index(
    drop=False)
activity_data_pivot = activity_data_pivot.reset_index(drop=True)
activity_data_pivot.set_index('day', inplace=True)
activity_data_pivot = activity_data_pivot.loc[cat_dtype.categories]

fig = go.Figure(data=[
    go.Bar(name='Fairly Active', x=activity_data_pivot.index, y=activity_data_pivot['Fairly Active']),
    go.Bar(name='Lightly Active', x=activity_data_pivot.index, y=activity_data_pivot['Lightly Active']),
    go.Bar(name='Sedentary', x=activity_data_pivot.index, y=activity_data_pivot['Sedentary']),
    go.Bar(name='Very Active', x=activity_data_pivot.index, y=activity_data_pivot['Very Active'])
])

fig.update_layout(barmode='stack', yaxis_title='Minutes')
st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------------
st.subheader(f'AutoRegression for Steps and Sleep Efficiency')
col1, col2 = st.columns(2)

with col1:
   targetVars = ['sleepEfficiency', 'Steps']
   target = st.selectbox('Select Target Variable', targetVars)

with col2:
   steps = number = st.number_input('Select Forecast Horizon', value = 2)

result_ar, _ = fun.AutoReg_TS(correlation_df, target, 1, steps)

fig_ar = go.Figure()
for col in result_ar.columns:
    fig_ar.add_trace(go.Scatter(x=result_ar.index, y=result_ar[col], mode='lines', name=col))

# Set the chart title and axis labels
fig_ar.update_layout(title='Line Chart Example', xaxis_title='Date', yaxis_title='Value')

# Set the chart x-axis to date type and format
fig_ar.update_xaxes(type='date', tickformat='%b %d')

# Set the minimum and maximum y-axis range
fig_ar.update_yaxes(range=[result_ar.values.min(), result_ar.values.max()])

# Render the plotly figure in Streamlit
st.plotly_chart(fig_ar, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------

st.subheader(f'LSTM on Sleep Efficiency')
col1, col2 = st.columns(2)

df_lstm = correlation_df.copy()
df_lstm.dropna(inplace=True)

df_lstm['dateTime'] = pd.to_datetime(df_lstm['dateTime'])

# set dateTime column as index
df_lstm.set_index('dateTime', inplace=True)

df_lstm['target'] = df_lstm['sleepEfficiency']
df_lstm.drop(columns='sleepEfficiency', inplace=True)

with col1:
   lstm_nodes = number = st.number_input('Insert LSTM Nodes', value = 25)

with col2:
   epochs = number = st.number_input('Insert Epochs', value = 100)

result_lstm, mape_val = fun.LSTM_model(df_lstm, lstm_nodes=lstm_nodes, epochs=epochs)

st.write(f'MAPE: {mape_val:.3f}%')

fig_lstm = go.Figure()
for col in result_lstm.columns:
    fig_lstm.add_trace(go.Scatter(x=result_lstm.index, y=result_lstm[col], mode='lines', name=col))

# Set the chart title and axis labels
fig_lstm.update_layout(title='Line Chart Example', xaxis_title='Date', yaxis_title='Value')
fig_lstm.update_layout(hovermode="x unified")
# Set the chart x-axis to date type and format
fig_lstm.update_xaxes(type='date', tickformat='%b %d')

# Set the minimum and maximum y-axis range
fig_lstm.update_yaxes(range=[result_lstm.values.min(), result_lstm.values.max()])

# Render the plotly figure in Streamlit
st.plotly_chart(fig_lstm, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------

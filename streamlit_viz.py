import datetime as dt
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import functions as fun

# Options to be able to see all columns when printing dfs
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# Globals
from functions import DATE_FORMAT, SLEEP_LEVEL_COLORS, ACTIVITY_LEVEL_COLORS, STEPS_COLOR
START_DATE, END_DATE = fun.get_min_max_dates()
# Number of days considered
NUM_DAYS = (dt.datetime.strptime(END_DATE, DATE_FORMAT) - dt.datetime.strptime(START_DATE, DATE_FORMAT)).days

# ----------------------------------------------------------------------------------------------------------------------
# Streamlit Dashboard title
st.title('Fitbit Sleep-Activity Insights')

st.write("Welcome to our Streamlit dashboard on Sleep vs Activity! ")

# ----------------------------------------------------------------------------------------------------------------------
# Sleep & Activity - Numeric indicators
st.subheader("General Information about sleep and activity")

st.write("Here we present general information about sleep and activity. Feel free to choose different date ranges both "
         "with respect to the number of days as well as the start and end date we consider.")

# Define the slider widget for the numeric indicators
start_date = dt.datetime.strptime(START_DATE, DATE_FORMAT)
end_date = dt.datetime.strptime(END_DATE, DATE_FORMAT)
date_range = st.slider(
    ":calendar: Please select the date period you want to consider:",
    min_value=start_date,
    max_value=end_date,
    value=(start_date, end_date),
    format="DD MMM YYYY")

# Print the number of selected days
nDaysSelected = (date_range[1] - date_range[0]).days
st.markdown(f"<h1 style='text-align: center; color: black; font-size: 20px; "
            f"font-weight: bold;'>{nDaysSelected}/{NUM_DAYS} days selected</h1>",
            unsafe_allow_html=True)

# ----- Avg sleep duration (total)
tot_avg_sleep_duration = fun.get_avg_sleep_duration(date_range)

# ----- Sleep start time (most common one)
most_common_hour, nNights = fun.get_most_common_sleep_start_time(date_range)

# ----- Avg sleep efficiency
avg_sleep_efficiency = fun.get_avg_sleep_eff(date_range)

# ----- Avg number of steps
avg_steps = fun.get_avg_steps(date_range)

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
# ----- Pie chart - Avg number of minutes in each stage (total)
avg_min_stage_ser = fun.avg_num_min_each_stage_ser(date_range)
# Define plot
title = 'Avg time spent in each Sleep Stage'
avg_min_stage_fig = fun.plot_pie_from_ser(avg_min_stage_ser, title, colors=SLEEP_LEVEL_COLORS)

# ----- Pie chart - Avg minutes in different activity zones
avg_min_activity_ser = fun.get_avg_min_activity_ser(date_range)
# Define plot
title = 'Avg time spent in each Activity Level'
avg_min_activity_fig = fun.plot_pie_from_ser(avg_min_activity_ser, title, colors=ACTIVITY_LEVEL_COLORS)

# Plot the two pie charts side by side
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(avg_min_stage_fig, use_container_width=True) # user_container_width helps in plots not overlapping
with col2:
    st.plotly_chart(avg_min_activity_fig, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
# Comments on the comclusions we can draw from these plots

st.write(" We can see here that the person wearing the Fitbit seems to be getting adequate sleep in general, given that"
         "most experts reccommend 7-9 hours of sleep per night. Additionally this person seems to be a bit of a "
         "nightowl sleeping generally around midnight or even past midnight."
         ""
         "This person is also quite active achieving the general goal of 10.000 steps per day on average."
         ""
         "Another interesting conclusion is the fact that the distributions of the sleep and activity levels seem to be"
         "quite robust. We can see that by selecting different ranges of dates to consider, both with respect to the "
         "number of days, as well as the start and end dates. "
         ""
         "Regarding sleep, we see that Light Sleep accounts for 50% of the sleep stages, with the next larger sleep "
         "stage being REM sleep with roughly 20% of the sleep."
         ""
         "Similarly, we see that more that activity of various levels, accounts for more than 25% of the users state, "
         "with the rest corresponding to the wearer being sedentary. One first conclusion, could be that hitting the"
         "10k step mark, equals to being active for 1/4 of the day."
         "")


# ----------------------------------------------------------------------------------------------------------------------
# Plot daily time series for sleep and activity level data
st.subheader("Sleep and Activity level time series per date")

st.write("Here we can explore the sleep and activity data in more detail. For each selected date we can see in the form"
         "of a time series how the wearer's sleep and activity levels evolve over time.")

# Define date widget
start_date = fun.to_date(START_DATE)
end_date = fun.to_date(END_DATE)
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
    try:
        # ----- Sleep levels time series
        # Here the functions are kind of used not as they are suppossed to
        # The functions where developed to create a list of dataframes with all the data, so here we only use them for
        # one date.
        dates = [date]
        sleepStartEnd_list = fun.get_sleep_start_end(dates)
        sleepStartEnd = sleepStartEnd_list[0]
        sleepLevelTimeSeries_df = fun.get_sleep_level_timeseries_df(sleepStartEnd)
        # Plot the timeseries
        fig = fun.plot_sleep_level_time_series(sleepLevelTimeSeries_df, colors=SLEEP_LEVEL_COLORS)
        st.plotly_chart(fig.to_dict())
    except ValueError as e:
        st.write(str(e))
    except KeyError as e:
        st.write(f":exclamation: A value that should not be there appeared and as result we cannot plot the Sleep "
                 f"Stages timeseries for {dates[0]}. This is likely related to some error furing the import of the data."
                 f"Please select another date. :exclamation:")

    # ----- Activity levels time series
    activityDate = date.strftime('%Y-%m-%d')
    fullActivityTimeseries = fun.get_activity_detail_timeseries(activityDate)
    activity_timeseries_df = fun.get_activity_timeseries_df(fullActivityTimeseries)
    # Plot the timeseries
    fig = fun.plot_activity_level_timeseries(activity_timeseries_df)
    st.plotly_chart(fig.to_dict())



# ----------------------------------------------------------------------------------------------------------------------
st.subheader(f'Averages per day of week')
# -------------------- Stacked bar chart for the sleep levels duration
# Get data
sleep_data_pivot = fun.get_sleep_data_pivot()

fig = go.Figure(data=[
    go.Bar(name='Awake', x=sleep_data_pivot.index, y=sleep_data_pivot['wake'],
               marker_color=SLEEP_LEVEL_COLORS['Awake']),
    go.Bar(name='Light', x=sleep_data_pivot.index, y=sleep_data_pivot['light'],
               marker_color=SLEEP_LEVEL_COLORS['Light']),
    go.Bar(name='REM', x=sleep_data_pivot.index, y=sleep_data_pivot['rem'],
               marker_color=SLEEP_LEVEL_COLORS['REM']),
    go.Bar(name='Deep', x=sleep_data_pivot.index, y=sleep_data_pivot['deep'],
           marker_color=SLEEP_LEVEL_COLORS['Deep'])
])

fig.update_layout(barmode='stack', yaxis_title='Minutes', title="Avg duration of sleep levels")
st.plotly_chart(fig)

# -------------------- Bar plot for the number of steps aggregated by day of week
# Get data
dType = 'steps'
steps_per_day_df = fun.get_df(dType=dType, addDateTimeCols=True)

steps_per_day_df = steps_per_day_df.groupby('day_name').agg({'value': 'mean'}).reset_index(drop=False)
steps_per_day_df['value'] = steps_per_day_df['value'].astype(int).round()
steps_per_day_df = steps_per_day_df.rename(columns={'value': 'Steps', 'day_name': 'Day'})
fig = px.bar(steps_per_day_df, x='Day', y='Steps', title="Avg steps")
fig.update_traces(marker_color=STEPS_COLOR['Steps'])
st.plotly_chart(fig)

# -------------------- Stacked bar chart for the activity stages duration
# Get data
activity_data_pivot = fun.get_activity_data_pivot()

fig = go.Figure(data=[
    go.Bar(name='Sedentary', x=activity_data_pivot.index, y=activity_data_pivot['Sedentary'],
               marker_color=ACTIVITY_LEVEL_COLORS['Sedentary']),
    go.Bar(name='Lightly Active', x=activity_data_pivot.index, y=activity_data_pivot['Lightly Active'],
               marker_color=ACTIVITY_LEVEL_COLORS['Lightly Active']),
    go.Bar(name='Fairly Active', x=activity_data_pivot.index, y=activity_data_pivot['Fairly Active'],
           marker_color=ACTIVITY_LEVEL_COLORS['Fairly Active']),
    go.Bar(name='Very Active', x=activity_data_pivot.index, y=activity_data_pivot['Very Active'],
           marker_color=ACTIVITY_LEVEL_COLORS['Very Active'])
])

fig.update_layout(barmode='stack', yaxis_title='Minutes', title="Avg duration of activity levels")
st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------------
st.subheader(f'Activity status over time')
# Area chart for the activity levels duration

# Get data
activity_df = fun.get_activity_df().drop(columns='Sedentary', axis=1)
activity_df.set_index('dateTime', inplace=True)
activity_df.index.name = None

# Plot result
fig = px.area(activity_df, color_discrete_map=ACTIVITY_LEVEL_COLORS)
st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------------
st.subheader(f'Relationship between the sleep levels duration and number of steps')
# 3d bubble graph for sleep levels and coloring based on the number of steps
# Get data
sleep_level_summary_df = fun.get_sleep_level_summary_df()

# Plot data
fig = px.scatter_3d(sleep_level_summary_df, x='Deep sleep minutes', y='Light sleep minutes', z='REM minutes',
                    color='Steps', color_continuous_scale='redor')

st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------------
st.subheader(f'Correlation Matrix')
correlation_df = fun.heatmapPlots()
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
    ['Sleep Efficiency', 'Steps', 'Deep Sleep', 'Light Sleep', 'REM Sleep', 'Awake', 'Fairly Active',
     'Lightly Active', 'Sedentary', 'Very Active'],
    ['Sleep Efficiency', 'Steps'])

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

rename_cols = {
    'sleepEfficiency': 'Sleep Efficiency',
    'Steps': 'Steps',
    'deep': 'Deep Sleep',
    'light': 'Light Sleep',
    'rem': 'REM Sleep',
    'wake': 'Awake',
    'minutesFairlyActive': 'Fairly Active',
    'minutesLightlyActive': 'Lightly Active',
    'minutesSedentary': 'Sedentary',
    'minutesVeryActive': 'Very Active'

}
df_normalized = df_normalized.rename(columns=rename_cols)

if resample == 'No resample/smooth':
    smoothed_ts_resampled = df_normalized[options].dropna()
else:
    smoothed_ts_resampled = df_normalized[options].resample(resample).median()

colors_dict = fun.get_new_complete_colors()
with row[1]:
    fig = go.Figure()
    for col in smoothed_ts_resampled.columns:
        trace = go.Scatter(x=smoothed_ts_resampled.index,
                           y=smoothed_ts_resampled[col],
                           name=col,
                           line_color=colors_dict[col])
        fig.add_trace(trace)
    st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------------
st.subheader(f'AutoRegression for Steps and Sleep Efficiency')
col1, col2 = st.columns(2)

with col1:
   targetVars = ['sleepEfficiency', 'Steps']
   target = st.selectbox('Select Target Variable', targetVars)

with col2:
   steps = number = st.number_input('Select Forecast Horizon', value=2)

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
# LSTM Plot
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
   lstm_nodes = number = st.number_input('Insert LSTM Nodes', value=25)

with col2:
   epochs = number = st.number_input('Insert Epochs', value=25)

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

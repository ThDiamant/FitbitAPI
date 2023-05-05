import datetime as dt
import streamlit as st
import pandas as pd
import pymongo as mongo
import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects as go

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


def connect_to_db():
    """
    Returns the collection specified by the DATA_COLLECTION_NAME global variable of the DB_NAME MongoDb.
    If it does not exist it throws an error.
    """
    global DB_NAME
    global DATA_COLLECTION_NAME

    client = mongo.MongoClient('localhost', 27017)
    fitbitDb = client[DB_NAME]
    if DATA_COLLECTION_NAME in fitbitDb.list_collection_names():
        return fitbitDb[DATA_COLLECTION_NAME]
    else:
        raise Exception(f"Collection {DATA_COLLECTION_NAME} does not exist.")


def to_datetime(date, time=""):
    """
    Converts date (str or datetime.date) into datetime.datetime. If a time argument is given, it includes it
    in the datetime.datetime object it returns.
    """

    if isinstance(date, str):
        datetimeObj = dt.datetime.fromisoformat(date)
    elif isinstance(date, dt.date):
        datetimeObj = dt.datetime.combine(date, dt.datetime.min.time())
    else:
        raise ValueError("Unsupported type for date. It should be either a string or a datetime.date object.")

    if time != "":
        datetimeObj = dt.datetime.combine(datetimeObj, dt.datetime.strptime(time, '%H:%M:%S').time())

    return datetimeObj


def check_dType(dType):
    fitbitCollection = connect_to_db()
    distinctTypes = fitbitCollection.distinct("type")
    if dType is not None:
        if dType not in distinctTypes:
            raise ValueError(f"dType needs to be one of {distinctTypes}.")


def load_data(dType=None, date=None):
    """
    Inputs:
        - dType <str>: The 'type' key we want to pull from MongoDB.
        - date <str, datetime>: The date we want the data of.

    Creates a query based on dType and date and returns its result.
    """
    fitbitCollection = connect_to_db()
    check_dType(dType)

    if date is None and dType is None:
        raise ValueError("One of dType or date must be specified.")
    elif date is None:
        myquery = {'type': dType}
    elif dType is None:
        myquery = {'data.dateTime': to_datetime(date)}
    else:
        myquery = {
            'type': dType,
            'data.dateTime': to_datetime(date)
        }

    return fitbitCollection.find(myquery)


def get_min_max_dates():
    """
    This function goes through all the documents in our MongoDb collection and finds the
    min and max dates we have data for. We use the USER_UUID to query the db since we
    only have data for one user.
    """
    global USER_UUID
    global DATE_FORMAT

    fitbitCollection = connect_to_db()
    # Get all distinct values of the 'type' key in each document
    distinctTypes = fitbitCollection.distinct("type")

    # Get all the data
    myquery = {'id': USER_UUID}
    query_result = list(fitbitCollection.find(myquery))

    # Create dictionary that for each 'type' contains all the datetime objects under 'data'
    dateTimes = {}
    for dType in distinctTypes:
        dateTimes[dType] = []
        for doc in query_result:
            if 'intraday' not in doc['type'].lower():
                dateTimes[dType].append(doc['data']['dateTime'])

    # For each 'type' get the earliest and latest date we have data for
    minMaxDates = {}
    for dType in dateTimes.keys():
        minMaxDates[dType] = {
            'min': min(dateTimes[dType]).strftime('%Y-%m-%d'),
            'max': max(dateTimes[dType]).strftime('%Y-%m-%d')
        }

    # Get the ealiest date for each key in a list and convert to set to remove duplicates
    minDates = set([minMaxDates[dType]['min'] for dType in minMaxDates.keys()])
    # Same for latest date
    maxDates = set([minMaxDates[dType]['max'] for dType in minMaxDates.keys()])

    # If there is only one min date then return it else raise an Exception
    if len(minDates) == 1:
        minDate = dt.datetime.strptime(next(iter(minDates)), '%Y-%m-%d')
        minDate = minDate.strftime(DATE_FORMAT)
    else:
        raise Exception(f"minDates contains more than 1 element: {minDates}")

    if len(maxDates) == 1:
        maxDate = dt.datetime.strptime(next(iter(maxDates)), '%Y-%m-%d')
        maxDate = maxDate.strftime(DATE_FORMAT)
    else:
        raise Exception(f"maxDates contains more than 1 element: {maxDates}")

    return minDate, maxDate


def add_datetime_columns(df):
    if "dateTime" in df.columns:
        df["month"] = df["dateTime"].dt.month
        df["day"] = df["dateTime"].dt.day
        df["hour"] = df["dateTime"].dt.hour
        df["minute"] = df["dateTime"].dt.minute
        df["second"] = df["dateTime"].dt.second
        # Get month name
        df["month_name"] = df["dateTime"].dt.month_name()
        month_cat_dtype = pd.CategoricalDtype(
            categories=['January', 'February', 'March', 'April',
                        'May', 'June', 'July', 'August',
                        'September', 'October', 'November', 'December'],
            ordered=True)
        df['month_name'] = df['month_name'].astype(month_cat_dtype)
        # Get day name
        df["day_name"] = df["dateTime"].dt.day_name()
        day_cat_dtype = pd.CategoricalDtype(
            categories=['Monday', 'Tuesday', 'Wednesday',
                        'Thursday', 'Friday', 'Saturday', 'Sunday'],
            ordered=True)

        df['day_name'] = df['day_name'].astype(day_cat_dtype)
        return df
    else:
        raise Exception("Need a dateTime column in df.")


def get_df(dType=None, date=None, addDateTimeCols=False):
    """
    Loads data from MongoDB into a dataframe.
    """
    query_result = load_data(dType=dType, date=date)
    query_sample_data = query_result[0]['data']
    cols = [key for key in query_sample_data.keys()]
    data = [doc['data'] for doc in query_result]
    df = pd.DataFrame(data,
                      columns=cols)
    if addDateTimeCols:
        df = add_datetime_columns(df)
    return df

# Connect to MongoDB collection where the data are stored
fitbitCollection = connect_to_db()
distinctTypes = fitbitCollection.distinct("type")

START_DATE, END_DATE = get_min_max_dates()
# Number of days considered
NUM_DAYS = (dt.datetime.strptime(END_DATE, DATE_FORMAT) - dt.datetime.strptime(START_DATE, DATE_FORMAT)).days
print(START_DATE, END_DATE, NUM_DAYS)

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

# ----- Avg sleep duration (total)
def get_avg_sleep_duration(period):
    dType = 'sleep-duration'
    sleep_duration_df = get_df(dType=dType)
    # Filter based on the selected period
    sleep_duration_df = sleep_duration_df.iloc[-period:]

    # Convert sleep duration from ms to hours
    sleep_duration_df['value'] = sleep_duration_df['value'] / 3600000
    tot_avg_sleep_duration = round(sleep_duration_df['value'].mean(), 1)
    return tot_avg_sleep_duration

tot_avg_sleep_duration = get_avg_sleep_duration(period)
print(tot_avg_sleep_duration)


# ----- Sleep start time (most common one)
def get_most_common_sleep_start_time(period):
    dType = 'sleep-startTime'
    sleep_stime_df = get_df(dType=dType)
    # Filter based on the selected period
    sleep_stime_df = sleep_stime_df.iloc[-period:]
    # Create column containing the hour
    sleep_stime_df['hour'] = sleep_stime_df['value'].dt.hour
    # Get the count of each hour
    hour_counts_df = (sleep_stime_df
                      .groupby('hour')
                      .count()
                      .sort_values(by='value', ascending=False))
    # nNights is the number of nights with sleep start time equal to the most common hour
    most_common_hour, nNights = hour_counts_df['value'].head(1).index[0], hour_counts_df['value'].iloc[0]

    return most_common_hour, nNights

most_common_hour, nNights = get_most_common_sleep_start_time(period)
print(most_common_hour, nNights, period)

# ----- Avg sleep efficiency
def get_avg_sleep_eff(period):
    dType = 'sleep-efficiency'
    sleep_efficiency_df = get_df(dType=dType)
    # Filter based on the selected period
    sleep_efficiency_df = sleep_efficiency_df.iloc[-period:]
    avg_sleep_efficiency = round(sleep_efficiency_df['value'].mean(), 1)

    return avg_sleep_efficiency

avg_sleep_efficiency = get_avg_sleep_eff(period)

# ----- Avg number of steps
def get_avg_steps(period):
    dType = 'steps'
    steps_df = get_df(dType=dType)
    # Filter based on the selected period
    steps_df = steps_df.iloc[-period:]
    avg_steps = int(round(steps_df['value'].mean(), 0))

    return avg_steps

avg_steps = get_avg_steps(period)


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
def avg_num_min_each_stage_ser(period):
    dType = "sleepSummary-stages"
    sleep_level_summary_df = get_df(dType=dType)
    # Filter based on the selected period
    sleep_level_summary_df = sleep_level_summary_df.iloc[-period:]
    # Get series with total avg time (min) spent in each sleep stage
    avg_min_stage_ser = (sleep_level_summary_df
                        .drop('dateTime', axis=1)
                        .mean()
                        .round(0)
                        .astype(int))

    return avg_min_stage_ser

avg_min_stage_df = avg_num_min_each_stage_ser(period)

def plot_bar_from_ser(ser, title, x_axis_title, y_axis_title):
    # Create a bar chart using plotly.graph_objects
    fig = go.Figure(go.Bar(x=ser.values,
                                         y=ser.index,
                                         orientation='h',
                                         marker_color='green'))
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        font=dict(family='Arial', size=12)
    )

    return fig

# Plot
title = 'Avg time spent in each Sleep Stage'
x_axis_title = 'Minutes'
y_axis_title = 'Sleep Stage'
avg_min_stage_fig = plot_bar_from_ser(avg_min_stage_df, title, x_axis_title, y_axis_title)

# st.plotly_chart(avg_min_stage_fig)

# ----------------------------------------------------------------------------------------------------------------------
# Bar chart - Avg minutes in different activity zones

def get_activity_df():
    # Get data
    dType = 'minutesSedentary'
    minutesSedentary_df = get_df(dType=dType)
    minutesSedentary_df = minutesSedentary_df.rename(columns={'value': 'Sedentary'})

    dType = 'minutesLightlyActive'
    minutesLightlyActive_df = get_df(dType=dType)
    minutesLightlyActive_df = minutesLightlyActive_df.rename(columns={'value': 'Lightly active'})

    dType = 'minutesFairlyActive'
    minutesFairlyActive_df = get_df(dType=dType)
    minutesFairlyActive_df = minutesFairlyActive_df.rename(columns={'value': 'Fairly active'})

    dType = 'minutesVeryActive'
    minutesVeryActive_df = get_df(dType=dType)
    minutesVeryActive_df = minutesVeryActive_df.rename(columns={'value': 'Very active'})

    # Merge data
    activity_df = minutesSedentary_df.merge(minutesLightlyActive_df, on='dateTime', how='left')
    activity_df = activity_df.merge(minutesFairlyActive_df, on='dateTime', how='left')
    activity_df = activity_df.merge(minutesVeryActive_df, on='dateTime', how='left')

    activity_df['Sedentary'] = activity_df['Sedentary'].astype(int)
    activity_df['Lightly active'] = activity_df['Lightly active'].astype(int)
    activity_df['Fairly active'] = activity_df['Fairly active'].astype(int)
    activity_df['Very active'] = activity_df['Very active'].astype(int)

    return activity_df

def get_avg_min_activity_ser(period):
    activity_df = get_activity_df()

    # Filter based on the selected period
    activity_df = activity_df.iloc[-period:]
    avg_min_activity_ser = (activity_df
                            .drop('dateTime', axis=1)
                            .mean().round(0)
                            .astype(int))

    return avg_min_activity_ser

avg_min_activity_ser = get_avg_min_activity_ser(period)

# Plot
title = 'Avg time spent in each Activity Type'
x_axis_title = 'Minutes'
y_axis_title = 'Activity Type'
avg_min_activity_fig = plot_bar_from_ser(avg_min_activity_ser, title, x_axis_title, y_axis_title)

# st.plotly_chart(avg_min_activity_fig)

# Show the metrics side by side
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(avg_min_stage_fig)
with col2:
    st.plotly_chart(avg_min_activity_fig)

# ----------------------------------------------------------------------------------------------------------------------
# Plot time series for sleep level data
def get_sleep_start_end(dates):
    """
    Input:
        - dates <list>: Contains the dates for which we want to gather data.

    Returns a list of dictionaries, where each dictionary contains the sleep start and end time
    for the given date.
    """

    dTypes = ['sleep-startTime', 'sleep-endTime']
    sleepStartEnd_list = []
    for date in dates:
        sleepStartEnd = {}
        sleepStartEnd["date"] = date
        for dType in dTypes:
            query_result = load_data(dType=dType, date=date)
            query_sample_data = query_result[0]['data']

            sleepStartEnd[dType] = query_sample_data['value']

        sleepStartEnd_list.append(sleepStartEnd)

    return sleepStartEnd_list


def expand_time_series(query_result, step=10):
    """
    Inputs:
        - query_result: The result of querying the MongoDB to get the data we want.
        - step <int>:
    Since the query_result contains information in the form
    (<time the sleep level was entered>, <sleep level>, <duration in the sleep level (sec)>)
    we use this information to get a proper time series of the form (<dateTime>, <sleep level>).
    The expansion takes place so that for each stage, we add a point every step seconds.
    """

    # Construct sleep level time series
    sleepLevelTimeSeries = []
    for doc in query_result:
        docData = doc['data']
        # Sleep level
        level = docData['level']
        # Datetime when level was initially entered
        dateTime = docData['dateTime']
        # Total seconds spent in level
        totalSecInLevel = docData['value']
        # Create new point every sec seconds
        step = 10
        # Number of points that will be added in the time series
        nTimePeriods = int(totalSecInLevel / step) - 1

        # First element is zero so that the first value is the datetime when level
        # was initially entered (i.e. dateTime)
        timePeriods = [0] + [step] * (nTimePeriods)
        for sec in timePeriods:
            # print(f"sec: {sec}")
            dateTime += dt.timedelta(seconds=sec)
            # print(type(dateTime))
            dataPoint = (dateTime, level)
            sleepLevelTimeSeries.append(dataPoint)

    return sleepLevelTimeSeries


def create_sleep_level_ts_df(sleepLevelTimeSeries):
    """
    Creates a dataframe from the sleep level time series.
    """
    # Create sleep level time series dataframe
    sleepLevelTimeSeries_df = pd.DataFrame(sleepLevelTimeSeries, columns=["dateTime", "sleepStage"])
    # Change level names
    new_level_names = {
        "wake": "Awake",
        "rem": "REM",
        "light": "Light",
        "deep": "Deep"
    }
    sleepLevelTimeSeries_df["sleepStage"] = sleepLevelTimeSeries_df["sleepStage"].apply(lambda x: new_level_names[x])
    # Define sleepStage as a categorical variable
    cat_dtype = pd.CategoricalDtype(
        categories=['Deep', 'Light', 'REM', 'Awake'], ordered=True)
    sleepLevelTimeSeries_df['sleepStage'] = sleepLevelTimeSeries_df['sleepStage'].astype(cat_dtype)
    # Create numeric column for sleep stages for plotting
    sleepLevelTimeSeries_df['sleepStageNum'] = sleepLevelTimeSeries_df['sleepStage'].cat.codes

    return sleepLevelTimeSeries_df


def get_sleep_level_timeseries_data(sleepStartEnd_list):
    """
    Input:
        - sleepStartEnd_list <list>: List of dictionaries each containing three keys
                                     (date, sleep-startTime, sleep-endTime).

    Returns for each element of the input list the time series of sleep levels. First,
    the MongoDB is queried to get the relevant data. The data we get from Mongo contain
    the information as (<time the sleep level was entered>, <sleep level>,
    <duration in the sleep level (sec)>). For this reason, we then expand the time series
    with the expand_time_series() function.
    """

    dType = 'sleepLevelsData-data'
    # For each dictionary in sleepStartEnd_list (i.e. for each date)
    for sleepStartEnd in sleepStartEnd_list:
        # Create the query that will give us the time series data
        sleepStartTime = sleepStartEnd['sleep-startTime']
        sleepEndTime = sleepStartEnd['sleep-endTime']
        query = {
            'type': dType,
            'data.dateTime': {
                '$gte': sleepStartTime,
                '$lte': sleepEndTime
            }
        }

        # Get time series with time spent in each sleep level
        query_result = fitbitCollection.find(query)

        # Expand the time series
        sleepLevelTimeSeries = expand_time_series(query_result, step=10)
        # Create dataframe for expanded time serires
        sleepLevelTimeSeries_df = create_sleep_level_ts_df(sleepLevelTimeSeries)

        # Plot the timeseries
        plot_sleep_level_time_series(sleepLevelTimeSeries_df)


def plot_sleep_level_time_series(sleepLevelTimeSeries_df):
    global DATE_FORMAT

    colors = {'Awake': 'red', 'REM': 'lightblue', 'Light': 'blue', 'Deep': 'darkblue'}
    # Sleep date
    date = sleepLevelTimeSeries_df['dateTime'].iloc[0].strftime("%d %B %Y")

    # Create figure
    fig = px.scatter(sleepLevelTimeSeries_df, x="dateTime", y="sleepStageNum",
                     color="sleepStage", color_discrete_map=colors)

    fig.update_layout(
        title=f'Sleep Stages for {date}',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Sleep Stage',
                   tickmode='array',
                   tickvals=[0, 1, 2, 3],
                   ticktext=['Deep', 'Light', 'REM', 'Awake']),
        plot_bgcolor='white',
        height=500
    )

    # fig.show()
    return fig

# date = testDate
# dates = [date]
# sleepStartEnd_list = get_sleep_start_end(dates)
# sleepStartEnd_list
# fig = get_sleep_level_timeseries_data(sleepStartEnd_list)


"""
TO DO:
MAKE SURE THE USER CAN SELECT THE DATE THEY WANT TO SEE THE DATA FOR preferably using a calendar widget if possible.


1) Figure out what to do with the slider.
2) Try to make side-by-side plots not overlapping.
2) Create some more visualizations for activity and sleep (maybe that line plot from the app with smoothing options
3) Try to create correlations
4) ML if you have time.
"""




# COSTAS FROM HERE ON
# ----------------------------------------------------------------------------------------------------------------------
# Bar plot for the number of steps aggregated by day
st.subheader(f'Average number of steps per day')
# Get data
dType = 'steps'
steps_per_day_df = get_df(dType=dType, addDateTimeCols=True)

steps_per_day_df = steps_per_day_df.groupby('day').agg({'value': 'mean'}).reset_index(drop=False)
steps_per_day_df['value'] = steps_per_day_df['value'].astype(int).round()
steps_per_day_df = steps_per_day_df.rename(columns={'value': 'Steps', 'day': 'Day'})
steps_per_day_df = steps_per_day_df.sort_values(by='Day')
fig = px.bar(steps_per_day_df, x='Day', y='Steps')
st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------------
# Stacked bar chart for the sleep levels duration
st.subheader(f'Average duration of sleep levels per day')
# Get data
dType = 'sleepLevelsData-data'
sleep_levels_data = get_df(dType=dType)

sleep_levels_data['dateTime'] = sleep_levels_data['dateTime'].dt.date
sleep_data_grouped = sleep_levels_data.groupby(['dateTime', 'level'])['value'].sum()
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
sleep_data_pivot['dateTime'] = pd.to_datetime(sleep_data_pivot['dateTime'])
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

# ----------------------------------------------------------------------------------------------------------------------
# Area chart for the activity levels duration
st.subheader(f'Activity status over time')

# Get data
dType = 'minutesFairlyActive'
minutesFairlyActive_df = get_df(dType=dType)
minutesFairlyActive_df = minutesFairlyActive_df.rename(columns={'value': 'fairly active'})

dType = 'minutesVeryActive'
minutesVeryActive_df = get_df(dType=dType)
minutesVeryActive_df = minutesVeryActive_df.rename(columns={'value': 'very active'})

dType = 'minutesLightlyActive'
minutesLightlyActive_df = get_df(dType=dType)
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

# ----------------------------------------------------------------------------------------------------------------------
# Timeseries comparison between the activity and sleep duration
st.subheader(f'Timeseries comparison between the activity and sleep duration')

dType = 'sleep-duration'
totalMinutesAsleep_df = get_df(dType=dType)

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

dType = 'sleep-duration'
sleep_duration_df = get_df(dType=dType)
sleep_duration_df = sleep_duration_df.rename(columns={'value': 'Sleep duration'})
sleep_duration_df['Sleep duration'] = sleep_duration_df['Sleep duration'] / 60000

dType = 'steps'
steps_per_day_df = get_df(dType=dType)
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
sleep_level_summary_df = get_df(dType=dType)

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

sleep_level_summary_df = sleep_level_summary_df.drop_duplicates()

fig = px.scatter_3d(sleep_level_summary_df, x='Deep sleep minutes', y='Light sleep minutes', z='REM minutes',
                    color='Steps', color_continuous_scale='redor')

st.plotly_chart(fig)


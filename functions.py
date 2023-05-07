import datetime as dt
import streamlit as st
import pandas as pd
import pymongo as mongo
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ar_model import AutoReg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_percentage_error

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

# Connect to MongoDB collection where the data are stored
fitbitCollection = connect_to_db()
distinctTypes = fitbitCollection.distinct("type")

def to_date(x):
    """
    Returns x as a datetime.date() object.
    """
    global DATE_FORMAT
    if isinstance(x, str):
        # If input is a string, parse it as a date
        try:
            return dt.datetime.strptime(x, "%Y-%m-%d").date()
        except:
            return dt.datetime.strptime(x, DATE_FORMAT).date()
    elif isinstance(x, dt.date):
        # If input is already a date object, return it
        return x
    else:
        # Otherwise, raise an error
        raise ValueError("Input must be a string or datetime.date object")


def daterange(start_date, end_date):
    """
    Inputs:
        start_date, end_date: Can be either string or datetime.date

    Returns all dates in [start_date, end_date] as datetime.date objects.
    """
    start_date = to_date(start_date)
    end_date = to_date(end_date)
    for n in range(int((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)

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

def get_most_common_sleep_start_time(period):
    dType = 'sleep-startTime'
    sleep_stime_df = get_df(dType=dType)

    # Filter based on the selected period
    sleep_stime_df['dateTime'] = pd.to_datetime(sleep_stime_df['dateTime'])
    sleep_stime_df = sleep_stime_df.set_index('dateTime')
    # Filter based on the selected period
    sleep_stime_df = sleep_stime_df.loc[(sleep_stime_df.index >= period[0]) & (sleep_stime_df.index <= period[1])]

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

def get_avg_sleep_eff(period):
    dType = 'sleep-efficiency'
    sleep_efficiency_df = get_df(dType=dType)

    sleep_efficiency_df['dateTime'] = pd.to_datetime(sleep_efficiency_df['dateTime'])
    sleep_efficiency_df = sleep_efficiency_df.set_index('dateTime')
    # Filter based on the selected period
    sleep_efficiency_df = sleep_efficiency_df.loc[(sleep_efficiency_df.index >= period[0]) & (sleep_efficiency_df.index <= period[1])]

    avg_sleep_efficiency = round(sleep_efficiency_df['value'].mean(), 1)

    return avg_sleep_efficiency

def get_avg_steps(period):
    dType = 'steps'
    steps_df = get_df(dType=dType)

    steps_df['dateTime'] = pd.to_datetime(steps_df['dateTime'])
    steps_df = steps_df.set_index('dateTime')
    # Filter based on the selected period
    steps_df = steps_df.loc[(steps_df.index >= period[0]) & (steps_df.index <= period[1])]

    avg_steps = int(round(steps_df['value'].mean(), 0))

    return avg_steps

def avg_num_min_each_stage_ser(period):
    dType = "sleepSummary-stages"
    sleep_level_summary_df = get_df(dType=dType)

    sleep_level_summary_df['dateTime'] = pd.to_datetime(sleep_level_summary_df['dateTime'])
    sleep_level_summary_df = sleep_level_summary_df.set_index('dateTime')
    # Filter based on the selected period
    sleep_level_summary_df = sleep_level_summary_df.loc[(sleep_level_summary_df.index >= period[0]) & (sleep_level_summary_df.index <= period[1])]

    # Get series with total avg time (min) spent in each sleep stage
    avg_min_stage_ser = (sleep_level_summary_df
                        .mean()
                        .round(0)
                        .astype(int))

    return avg_min_stage_ser

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

def plot_pie_from_ser(ser, title):
    # Create a bar chart using plotly.graph_objects
    fig = go.Figure(data=go.Pie(labels=ser.index, values=ser.values))
    fig.update_layout(
        title=title,
        font=dict(family='Arial', size=12)
    )

    return fig

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

    activity_df['dateTime'] = pd.to_datetime(activity_df['dateTime'])
    activity_df = activity_df.set_index('dateTime')
    # Filter based on the selected period
    activity_df = activity_df.loc[(activity_df.index >= period[0]) & (activity_df.index <= period[1])]


    avg_min_activity_ser = (activity_df
                            .mean().round(0)
                            .astype(int))

    return avg_min_activity_ser

# Functions for sleep level data ----------------------------------------
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


def get_sleep_level_timeseries_df(sleepStartEnd):
    """
    Input:
        - sleepStartEnd <dict>: Dict containing three keys (date, sleep-startTime, sleep-endTime).

    Returns for each element of the input list the time series of sleep levels. First,
    the MongoDB is queried to get the relevant data. The data we get from Mongo contain
    the information as (<time the sleep level was entered>, <sleep level>,
    <duration in the sleep level (sec)>). For this reason, we then expand the time series
    with the expand_time_series() function.
    """

    dType = 'sleepLevelsData-data'

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

    return sleepLevelTimeSeries_df


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
        height=350
    )

    # fig.show()
    return fig


# Functions for activity level data ----------------------------------------
def get_activity_detail_timeseries(date):
    dateTimeStart = dt.datetime.strptime(date, '%Y-%m-%d')
    dateTimeEnd = dt.datetime.strptime(date, '%Y-%m-%d')
    t = dt.time(hour=23, minute=59)
    dateTimeEnd = dt.datetime.combine(dateTimeEnd.date(), t)

    activityTypeTimeseries = {}
    dTypes = ['minutesSedentary-intraday',
          'minutesLightlyActive-intraday',
          'minutesFairlyActive-intraday',
          'minutesVeryActive-intraday']
    for dType in dTypes:

        query = {
            'type': dType,
            'data.dateTime': {
                '$gte': dateTimeStart,
                '$lte': dateTimeEnd
            }
        }

        # Get time series with time spent in each sleep level
        query_result = fitbitCollection.find(query)
        activityTimeseries = []
        for doc in query_result:
            activityType = doc['type'].split('-')[0].replace('minutes', '').strip()
            docData = doc['data']

            dateTime = docData['dateTime']
            minutes = docData['value']
            if minutes != 0:
                # First element is zero so that the first value is the datetime when level
                # was initially entered (i.e. dateTime)
                timePeriods = [0] + [1] * (minutes - 1)
                for min in timePeriods:
                    dateTime += dt.timedelta(minutes=min)
                    dataPoint = (dateTime, activityType)
                    activityTimeseries.append(dataPoint)
            else:
                continue
        activityTypeTimeseries[activityType] = activityTimeseries

    fullActivityTimeseries = []
    for key in activityTypeTimeseries.keys():
        fullActivityTimeseries += activityTypeTimeseries[key]

    return fullActivityTimeseries


def get_activity_timeseries_df(fullActivityTimeseries):
    activity_timeseries_df = pd.DataFrame(fullActivityTimeseries, columns=["dateTime", "activityLevel"]).sort_values(
        by='dateTime')
    # Change level names
    new_level_names = {
        "Sedentary": "Sedentary",
        "LightlyActive": "Lightly Active",
        "FairlyActive": "Fairly Active",
        "VeryActive": "Very Active"
    }
    activity_timeseries_df["activityLevel"] = activity_timeseries_df["activityLevel"].apply(
        lambda x: new_level_names[x])
    # Define sleepStage as a categorical variable
    cat_dtype = pd.CategoricalDtype(
        categories=['Sedentary', 'Lightly Active', 'Fairly Active', 'Very Active'], ordered=True)
    activity_timeseries_df['activityLevel'] = activity_timeseries_df['activityLevel'].astype(cat_dtype)
    # Create numeric column for sleep stages for plotting
    activity_timeseries_df['activityLevelNum'] = activity_timeseries_df['activityLevel'].cat.codes

    return activity_timeseries_df


def plot_activity_level_timeseries(activity_timeseries_df):
    colors = {'Sedentary': 'red', 'Lightly Active': 'lightblue', 'Fairly Active': 'blue', 'Very Active': 'darkblue'}

    fig = px.scatter(activity_timeseries_df, x="dateTime", y="activityLevelNum",
                     color="activityLevel", color_discrete_map=colors)

    # Sleep date
    date = activity_timeseries_df['dateTime'].iloc[0].strftime("%d %B %Y")

    fig.update_layout(
        title=f'Activity Levels for {date}',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Activity Level',
                   tickmode='array',
                   tickvals=[0, 1, 2, 3],
                   ticktext=['Sedentary', 'Lightly Active', 'Fairly Active', 'Very Active']),
        plot_bgcolor='white',
        height=350
    )

    # fig.show()
    return fig

def merge_dataframes(df_list, common_col, how="outer"):
    """
    Merges a list of dataframes on a common column.
    
    Args:
    - df_list: list of pandas dataframes to merge
    - common_col: name of the common column to merge on
    - how: merge method (default: "outer")
    
    Returns:
    - pandas dataframe containing the merged data
    """
    # make a copy of the first dataframe in the list
    merged_df = df_list[0].copy()
    
    # loop through the remaining dataframes and merge on the common column
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on=common_col, how=how)
    
    # return the merged dataframe
    return merged_df

def heatmpaPlots():
    """
    The function retrieves data from various sources and merges them into a single dataframe for
    plotting.
    
    :return: The function `heatmpaPlots()` is returning a merged dataframe that includes data on sleep
    efficiency, steps, sleep summary stages, and activity summary stages. The data is merged based on
    the common column "dateTime".
    """

    sleepEfficiency_df = get_df("sleep-efficiency").rename(columns={'value': 'sleepEfficiency'})

    steps_df = get_df(dType="steps").rename(columns={'value': 'Steps'})

    sleepSummary_stages_df = get_df(dType="sleepSummary-stages")

    minutesFairlyActive_df = get_df(dType="minutesFairlyActive").rename(columns={'value': 'minutesFairlyActive'})
    minutesFairlyActive_df['minutesFairlyActive'] = minutesFairlyActive_df['minutesFairlyActive'].astype(int)

    minutesLightlyActive_df = get_df(dType="minutesLightlyActive").rename(columns={'value': 'minutesLightlyActive'})
    minutesLightlyActive_df['minutesLightlyActive'] = minutesLightlyActive_df['minutesLightlyActive'].astype(int)

    minutesSedentary_df = get_df(dType="minutesSedentary").rename(columns={'value': 'minutesSedentary'})
    minutesSedentary_df['minutesSedentary'] = minutesSedentary_df['minutesSedentary'].astype(int)

    minutesVeryActive_df = get_df(dType="minutesVeryActive").rename(columns={'value': 'minutesVeryActive'})
    minutesVeryActive_df['minutesVeryActive'] = minutesVeryActive_df['minutesVeryActive'].astype(int)

    df_list = [minutesFairlyActive_df, minutesLightlyActive_df, minutesSedentary_df, minutesVeryActive_df]

    activitySummary_stages_df = merge_dataframes(df_list=df_list, common_col="dateTime", how="outer")

    df_list = [sleepEfficiency_df, steps_df, sleepSummary_stages_df, activitySummary_stages_df]

    return merge_dataframes(df_list=df_list, common_col="dateTime", how="outer")


def AutoReg_TS(df, target, lag, steps):
    # Load your data
    ts = df[['dateTime', target]].copy()
    
    # Convert dateTime column to datetime type
    ts['dateTime'] = pd.to_datetime(ts['dateTime'])

    # Sort the dataframe by dateTime column
    ts = ts.sort_values(by='dateTime')

    # Set the dateTime column as the index
    ts = ts.set_index('dateTime')

    # Fill missing values with the mean of the rolling window
    window_size = 3
    ts = ts.rolling(window=window_size, min_periods=1).mean().fillna(method='bfill')

    # Split the data into train and test sets
    train = ts.iloc[:]
    # test = ts.iloc[-1:]
    last_index = df.index[-1]

    # Train an autoregression model with lag=1 (predicting tomorrow's steps)
    model = AutoReg(train[target], lags=steps)
    model_fit = model.fit()

    # Make a forecast for tomorrow's steps
    forecast = model_fit.forecast(steps=steps)

    # assuming the variable is named 'forecast'
    df_forecast = forecast.to_frame().reset_index()

    # rename the columns
    df_forecast.columns = ['dateTime', 'forecast']
    df_forecast['dateTime'] = pd.to_datetime(df_forecast['dateTime'])
    df_forecast = df_forecast.sort_values(by='dateTime')
    df_forecast = df_forecast.set_index('dateTime')

    ts = pd.concat([ts, df_forecast])

    # Print the forecasted steps value
    return ts, last_index



# Calculate the mean absolute percentage error
def mape(actuals, preds):
    return np.mean(np.abs((actuals - preds) / actuals)) * 100

def LSTM_model(df, lstm_nodes=50, epochs=200):
    # select the relevant columns for the model
    # ts = df.copy()
    # ts['target'] = ts['sleepEfficiency']
    # ts.drop(columns='sleepEfficiency', inplace=True)

    ts = df[['Steps', 'minutesFairlyActive', 'minutesLightlyActive', 'minutesSedentary', 'minutesVeryActive', 'target']].values

    # split the data into train and test sets
    train_data = ts[:df.shape[0]-5, :]
    test_data = ts[df.shape[0]-5:, :]

    # create the input sequences and corresponding output values for the train set
    X_train = train_data[:, :5]
    y_train = train_data[:, 5:]
    X_train, y_train = np.array(X_train), np.array(y_train)

    # create the input sequences and corresponding output values for the test set
    X_test = test_data[:, :5]
    y_test = test_data[:, 5:]
    X_test, y_test = np.array(X_test), np.array(y_test)

    # reshape input to be 3D [samples, timesteps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(lstm_nodes, activation='relu', input_shape=(1, 5)))
    model.add(Dense(25))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit the LSTM model
    model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0)

    # make predictions
    y_pred = model.predict(X_test)

    # print("Actuals: ", y_test)
    # print("Predicted: ", y_pred)

    # make predictions
    y_pred = model.predict(X_test)

    # calculate error for each prediction
    errors = np.empty(len(y_pred))
    for i in range(len(y_pred)):
        error = mean_absolute_percentage_error(y_test[i], y_pred[i]) * 100
        errors[i] = error
        # print(f"Error for prediction {i+1}: {error}")

    # Calculate the MAPE between the predictions and actuals
    mape_val = mape(y_test, y_pred)

    # print('MAPE:', mape_val)

    result = df[['target']]
    result['forecast'] = np.nan
    result['error'] = np.nan
    # new_values = y_pred.ravel()
    result.iloc[-len(y_pred.ravel()):, result.columns.get_loc('forecast')] = y_pred.ravel()
    result.iloc[-len(y_pred.ravel()):, result.columns.get_loc('error')] = errors

    return result, mape_val
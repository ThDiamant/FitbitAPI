import datetime as dt
import streamlit as st
import pandas as pd
import pymongo as mongo
import plotly.express as px
import plotly.graph_objs as go

# Options to be able to see all columns when printing
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

USER_UUID = "3cc4e2ee-8c2f-4c25-955b-fe7f6ffcbe44"
DB_NAME = "fitbit"
DATA_COLLECTION_NAME = "fitbitCollection"


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

# Streamlit Dashboard title
st.title('Fitbit Sleep-Activity Insights')

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
level_summary_df = get_df(dType=dType)

level_summary_df = level_summary_df.drop("wake", axis=1)
rename_cols = {
    "deep": "Deep sleep minutes",
    "light": "Light sleep minutes",
    "rem": "REM minutes"
}
level_summary_df = level_summary_df.rename(columns=rename_cols)

steps = bubble_df[['dateTime', 'Steps']]
level_summary_df = level_summary_df.merge(steps, on='dateTime', how='left')

level_summary_df['Light sleep minutes'] = level_summary_df['Light sleep minutes'].astype(int)
level_summary_df['Deep sleep minutes'] = level_summary_df['Deep sleep minutes'].astype(int)
level_summary_df['REM minutes'] = level_summary_df['REM minutes'].astype(int)
level_summary_df['Steps'] = level_summary_df['Steps'].astype(float)

level_summary_df = level_summary_df.drop_duplicates()

fig = px.scatter_3d(level_summary_df, x='Deep sleep minutes', y='Light sleep minutes', z='REM minutes',
                    color='Steps', color_continuous_scale='redor')

st.plotly_chart(fig)


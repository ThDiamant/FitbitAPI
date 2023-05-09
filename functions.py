import datetime as dt
import fitbit
import requests as req
import pandas as pd
import pymongo as mongo
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_percentage_error

# Globals
USER_UUID = "3cc4e2ee-8c2f-4c25-955b-fe7f6ffcbe44"
DB_NAME = "fitbit"
DATA_COLLECTION_NAME = "fitbitCollection"
DATE_FORMAT = "%d %B %Y"
START_DATE = "2023-03-29"
SLEEP_LEVEL_ORDER = {
    "Sedentary": 0,
    "REM": 1,
    "Light": 2,
    "Deep": 3
}
ACTIVITY_LEVEL_ORDER = {
    "Sedentary": 0,
    "Lightly Active": 1,
    "Fairly Active": 2,
    "Very Active": 3
}
SLEEP_LEVEL_COLORS = {
    'Awake': 'red',
    'REM': 'lightblue',
    'Light': 'blue',
    'Deep': 'darkblue'
}
ACTIVITY_LEVEL_COLORS = {
    'Sedentary': 'grey',
    'Lightly Active': 'lightgreen',
    'Fairly Active': 'gold',
    'Very Active': 'orange'
}
STEPS_COLOR = {'Steps': 'purple'}


def check_create_collection(mongoDb, collection):
    """ 
    Checks if collection exists in mongoDb, and if it doesn't it creates it.
    """
    if collection in mongoDb.list_collection_names():
        print(f"Collection {collection} already exists, proceeding.")
    else:
        mongoDb.create_collection(collection)
        print(f"Collection {collection} created.")

    return mongoDb[collection]


def check_create_index(collection, index, indexName):
    """ 
    Checks if index with indexName exists in collection, and if it doesn't it creates it.
    """
    # Check if the index exists
    if indexName not in [index['name'] for index in collection.list_indexes()]:
        # Create the index if it does not exist
        collection.create_index(index, name = indexName, unique=True)
        print(f"Index {indexName} created.")
    else:
        print(f"Index {indexName} already exists, proceeding.")

def create_document(documentType, dataDict):
    """ 
    Inputs:
        > documentType <str>: The type entry of the document to be created.
        > dataDict <dict>: The 'data' entry of the document to be created.

    Creates a document to be inserted into MongoDB.
    """
    global USER_UUID

    myDocument = {}
    myDocument["id"] = USER_UUID
    myDocument["type"] = documentType
    myDocument["data"] = dataDict

    return myDocument

     
def save_document(myCollection, myDocument):
    """
    Inputs:
        > myCollection: MongoDB collection in which we want to add myDocument.
        > myDocument <dict>: Document to be inserted.
    Adds myDocument in myCollection and checks if it was inserted successfully. If
    If myDocument already exists in myCollection, if it cannot find it, a ValueError
    is raised.
    """

    try:
        # Insert myDocument in Mongo
        result = myCollection.insert_one(myDocument)
        # Check if the document was inserted successfully
        if not result.inserted_id:
            raise Exception(f"Document {myDocument} not inserted.")
    # If record already exists
    except mongo.errors.DuplicateKeyError:
        # Try to find the document in the DB
        query = {
            'type': myDocument['type'],
            'data.dateTime': myDocument['data']['dateTime']
        }

        existing_doc = myCollection.find_one(query)
        if existing_doc is None:
            # Something went wrong, raise an error
            raise ValueError("Cannot find existing document in the collection.")
        else:
            # Document already exists, ignore it
            pass
    except Exception as e:
        print('Error: %s' % e)
    

def create_and_save_document(fitbitCollection, documentType, dataDict):
    """ 
    Creates and saves document into fitbitCollection
    """
    dataDocument = create_document(documentType, dataDict)
    save_document(fitbitCollection, dataDocument)


def to_datetime(date, time = ""):
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


def get_summary_key_data(sleepSummaryData, sleepSummaryDataKey, single_date):
    """
    Input:
        > sleepSummaryData <dict>: Contains the sleep data from the Fitbit API reponse under the 'summary' key.
        > single_date <datetime.date>: The date the data of which we parse.
    Output:
        > documentType <str>: The 'type' key of the document to be inserted in MongoDB.
        > dataDict <dict>: The 'data' key of the document to be insterted in MongoDB.

    This function parses through the data the Fitbit API gives us for a single day and returns
    only the data under the 'summary' key that we are interested in keeping in our MongoDB. 
    More specifically, it returns the type of the entry, and the relevant data for each key we decide to keep.
    """

    # Dictionary that will hold the 'data' entry of the document
    dataDict = {}
    documentType = "sleepSummary-{}".format(sleepSummaryDataKey)

    dataDict["dateTime"] = to_datetime(single_date)
    if sleepSummaryDataKey == "stages":
        for stage in sleepSummaryData[sleepSummaryDataKey].keys():
            dataDict[stage] = sleepSummaryData[sleepSummaryDataKey][stage]
    else:
        dataDict[sleepSummaryDataKey] = sleepSummaryData[sleepSummaryDataKey]

    return documentType, dataDict

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

def to_date(x):
    """
    Returns x as a datetime.date() object.
    """
    if isinstance(x, str):
        # If input is a string, parse it as a date
        return dt.datetime.strptime(x, "%Y-%m-%d").date()
    elif isinstance(x, dt.date):
        # If input is already a date object, return it
        return x
    else:
        # Otherwise, raise an error
        raise ValueError("Input must be a string or datetime.date object")


def get_sleep_data(date, accessToken):
    """
    Inputs:
        - date <str> (yyyy-mm-dd): Date for which we want to pull data.
        
    Returns the data related to sleep for <date> from the Fitbit API. 
    """

    global START_DATE
        
    date = to_date(date)
    start_date = to_date(START_DATE)

    # Check date value
    if date < start_date:
        raise ValueError("date cannot be before {}".format(start_date))
    
    # Make API get request
    headers = {
        'accept': 'application/json',
        'authorization': 'Bearer {}'.format(accessToken),
    }
    try:
        response = req.get('https://api.fitbit.com/1.2/user/-/sleep/date/{}.json'.format(date), 
                           headers = headers)
    except fitbit.exceptions.HTTPTooManyRequests as e:
        tryAfterMin = e.retry_after_secs/60
        errorMessage = str(e) + ", please try again after {:.1f} min.".format(tryAfterMin)
        raise Exception(errorMessage)

    return response.json()


def get_activity_data(date, authClient):
    """
    Inputs:
        - date <str> (yyyy-mm-dd): Date we want the data of.
    
    Returns activity data for the specified date. Activity is quantified in terms of the elements of the resources list
    that is defined inside the function.
    """
    
    global START_DATE

    date = to_date(date)
    start_date = to_date(START_DATE)

    # Check date value
    if date < start_date:
        raise ValueError("date cannot be before {}".format(start_date))
   
    # Dictionary where data returned by the API will be stored
    data = {}

    # Different kinds of resources that quantify activity
    resources = [
        "minutesSedentary",
        "minutesLightlyActive",
        "minutesFairlyActive",
        "minutesVeryActive",
        "steps"
    ]
    
    try:
        # A separate API call is made for each resource
        for resource in resources:
            resourceString = "activities/" + resource
            # detailString can be one of 1min, 5min, 15min
            if resource == "steps":
                # Thought this might make more sense, feel free to change it if you think otherwise
                detailString = "1min"
            else:
                detailString = "15min"
            
            # Use fitbit module to make the API get request
            data[resource] = authClient.intraday_time_series(resourceString, 
                                                               date, 
                                                               detail_level = detailString)
    except fitbit.exceptions.HTTPTooManyRequests as e:
        tryAfterMin = e.retry_after_secs/60
        errorMessage = str(e) + ", please try again after {:.1f} min.".format(tryAfterMin)
        raise Exception(errorMessage)
        
    return data

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

def get_avg_sleep_duration(period):
    dType = 'sleep-duration'
    sleep_duration_df = get_df(dType=dType)
    # Filter based on the selected period
    sleep_duration_df['dateTime'] = pd.to_datetime(sleep_duration_df['dateTime'])
    sleep_duration_df = sleep_duration_df.set_index('dateTime')
    sleep_duration_df = sleep_duration_df.loc[(sleep_duration_df.index >= period[0]) & (sleep_duration_df.index <= period[1])]

    # Convert sleep duration from ms to hours
    sleep_duration_df['value'] = sleep_duration_df['value'] / 3600000
    tot_avg_sleep_duration = round(sleep_duration_df['value'].mean(), 1)
    return tot_avg_sleep_duration

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

    # Rename columns
    new_col_names = {
        "wake": "Awake         ",# Extra spaces so that the two bar charts appear as the same size
        "rem": "REM           ",# Hacking, am I right? :P
        "light": "Light          ",
        "deep": "Deep          "
    }
    sleep_level_summary_df = sleep_level_summary_df.rename(columns=new_col_names)

    sleep_level_summary_df['dateTime'] = pd.to_datetime(sleep_level_summary_df['dateTime'])
    sleep_level_summary_df = sleep_level_summary_df.set_index('dateTime')
    # Filter based on the selected period
    sleep_level_summary_df = sleep_level_summary_df.loc[(sleep_level_summary_df.index >= period[0]) &
                                                        (sleep_level_summary_df.index <= period[1])]

    # Get series with total avg time (min) spent in each sleep stage
    avg_min_stage_ser = (sleep_level_summary_df
                        .mean()
                        .round(0)
                        .astype(int))

    return avg_min_stage_ser

def plot_pie_from_ser(ser, title, colors):
    # For some reason marker_colors can't be a dict, but either a list or a pd.series
    colors = pd.Series(list(colors.values()),
                       index=pd.MultiIndex.from_tuples(colors.keys()))
    # Create a bar chart using plotly.graph_objects
    fig = go.Figure(data=go.Pie(labels=ser.index,
                                values=ser.values,
                                marker_colors=colors))
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
    minutesLightlyActive_df = minutesLightlyActive_df.rename(columns={'value': 'Lightly Active'})

    dType = 'minutesFairlyActive'
    minutesFairlyActive_df = get_df(dType=dType)
    minutesFairlyActive_df = minutesFairlyActive_df.rename(columns={'value': 'Fairly Active'})

    dType = 'minutesVeryActive'
    minutesVeryActive_df = get_df(dType=dType)
    minutesVeryActive_df = minutesVeryActive_df.rename(columns={'value': 'Very Active'})

    # Merge data
    activity_df = minutesSedentary_df.merge(minutesLightlyActive_df, on='dateTime', how='left')
    activity_df = activity_df.merge(minutesFairlyActive_df, on='dateTime', how='left')
    activity_df = activity_df.merge(minutesVeryActive_df, on='dateTime', how='left')

    activity_df['Sedentary'] = activity_df['Sedentary'].astype(int)
    activity_df['Lightly Active'] = activity_df['Lightly Active'].astype(int)
    activity_df['Fairly Active'] = activity_df['Fairly Active'].astype(int)
    activity_df['Very Active'] = activity_df['Very Active'].astype(int)

    return activity_df

def get_avg_min_activity_ser(period):
    activity_df = get_activity_df()

    activity_df['dateTime'] = pd.to_datetime(activity_df['dateTime'])
    activity_df = activity_df.set_index('dateTime')
    # Filter based on the selected period
    activity_df = activity_df.loc[(activity_df.index >= period[0]) & (activity_df.index <= period[1])]

    # Get the average minutes spent in each activity level
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
            query_result = list(load_data(dType=dType, date=date))
            if len(list(query_result)) > 0:
                query_sample_data = query_result[0]['data']
                sleepStartEnd[dType] = query_sample_data['value']
            else:
                raise ValueError(f":exclamation: No sleep data found. It's possible Fitbit was not worn on "
                                 f"{date} :exclamation:.")

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

    sleepLevelTimeSeries_df["sleepStage"] = (sleepLevelTimeSeries_df["sleepStage"]
                                             .apply(lambda x: new_level_names[x]))
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

def plot_sleep_level_time_series(sleepLevelTimeSeries_df, colors):
    global DATE_FORMAT

    # Sleep date
    date = sleepLevelTimeSeries_df['dateTime'].iloc[0].strftime("%d %B %Y")
    # Sleep duration
    maxTime = sleepLevelTimeSeries_df['dateTime'].max()
    minTime = sleepLevelTimeSeries_df['dateTime'].min()
    total_seconds = int((maxTime - minTime).total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    # Create figure
    fig = px.scatter(sleepLevelTimeSeries_df, x="dateTime", y="sleepStageNum",
                     color="sleepStage", color_discrete_map=colors)

    fig.update_layout(
        title=f'Sleep Stages for {date}. Sleep duration: {hours}h {minutes}min',
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

    # Get the activity levels that exist (e.g. Very Active may not be present for a day)
    existingStages = get_existing_activity_levels(activity_timeseries_df)

    # Define sleepStage as a categorical variable
    cat_dtype = pd.CategoricalDtype(
        categories=existingStages, ordered=True)
    activity_timeseries_df['activityLevel'] = activity_timeseries_df['activityLevel'].astype(cat_dtype)
    # Create numeric column for sleep stages for plotting
    activity_timeseries_df['activityLevelNum'] = activity_timeseries_df['activityLevel'].cat.codes

    return activity_timeseries_df

def get_existing_activity_levels(activity_timeseries_df):
    global ACTIVITY_LEVEL_ORDER

    val_counts = activity_timeseries_df['activityLevel'].value_counts()
    existingStages = list(val_counts[val_counts != 0].index)
    existingStagesOrd = sorted(existingStages, key=lambda x: ACTIVITY_LEVEL_ORDER[x])

    return existingStagesOrd

def plot_activity_level_timeseries(activity_timeseries_df):
    global ACTIVITY_LEVEL_COLORS

    # Get the activity levels that exist (e.g. Very Active may not be present for a day)
    existingStages = get_existing_activity_levels(activity_timeseries_df)
    colors = {activityStage: ACTIVITY_LEVEL_COLORS[activityStage] for activityStage in existingStages}

    fig = px.scatter(activity_timeseries_df, x="dateTime", y="activityLevelNum",
                     color="activityLevel", color_discrete_map=colors)

    # Sleep date
    date = activity_timeseries_df['dateTime'].iloc[0].strftime("%d %B %Y")

    fig.update_layout(
        title=f'Activity Levels for {date}',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Activity Level',
                   tickmode='array',
                   tickvals=[i for i in range(len(existingStages))],
                   ticktext=existingStages),
        plot_bgcolor='white',
        height=350
    )

    # fig.show()
    return fig

def get_sleep_data_pivot():
    dType = 'sleepLevelsData-data'
    sleep_levels_data = get_df(dType=dType)

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
        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ordered=True)
    sleep_data_pivot['day'] = sleep_data_pivot['day'].astype(cat_dtype)
    sleep_data_pivot = (sleep_data_pivot
                        .groupby('day')
                        .agg({'deep': 'mean', 'rem': 'mean', 'light': 'mean', 'wake': 'mean'})
                        .reset_index(drop=False))
    sleep_data_pivot = sleep_data_pivot.reset_index(drop=True)
    sleep_data_pivot.set_index('day', inplace=True)
    sleep_data_pivot = sleep_data_pivot.loc[cat_dtype.categories]

    return sleep_data_pivot

def get_activity_data_pivot():
    minutesFairlyActive_df = get_df(dType="minutesFairlyActive").rename(columns={'value': 'Fairly Active'})
    minutesFairlyActive_df['Fairly Active'] = minutesFairlyActive_df['Fairly Active'].astype(int)
    minutesLightlyActive_df = get_df(dType="minutesLightlyActive").rename(columns={'value': 'Lightly Active'})
    minutesLightlyActive_df['Lightly Active'] = minutesLightlyActive_df['Lightly Active'].astype(int)
    minutesSedentary_df = get_df(dType="minutesSedentary").rename(columns={'value': 'Sedentary'})
    minutesSedentary_df['Sedentary'] = minutesSedentary_df['Sedentary'].astype(int)
    minutesVeryActive_df = get_df(dType="minutesVeryActive").rename(columns={'value': 'Very Active'})
    minutesVeryActive_df['Very Active'] = minutesVeryActive_df['Very Active'].astype(int)
    df_list = [minutesFairlyActive_df, minutesLightlyActive_df, minutesSedentary_df, minutesVeryActive_df]
    activitySummary_stages_df = merge_dataframes(df_list=df_list, common_col="dateTime", how="outer")

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
        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ordered=True)
    activity_data_pivot['day'] = activity_data_pivot['day'].astype(cat_dtype)
    activity_data_pivot = (activity_data_pivot
                           .groupby('day')
                           .agg({'Fairly Active': 'mean',
                                 'Lightly Active': 'mean',
                                 'Sedentary': 'mean',
                                 'Very Active': 'mean'})
                           .reset_index(drop=False))
    activity_data_pivot = activity_data_pivot.reset_index(drop=True)
    activity_data_pivot.set_index('day', inplace=True)
    activity_data_pivot = activity_data_pivot.loc[cat_dtype.categories]

    return activity_data_pivot

def get_bubble_df():
    dType = 'sleep-duration'
    sleep_duration_df = get_df(dType=dType)
    sleep_duration_df = sleep_duration_df.rename(columns={'value': 'Sleep duration'})
    sleep_duration_df['Sleep duration'] = sleep_duration_df['Sleep duration'] / 60000

    dType = 'steps'
    steps_per_day_df = get_df(dType=dType)
    steps_per_day_df = steps_per_day_df.rename(columns={'value': 'Steps'})

    dType = 'minutesVeryActive'
    minutesVeryActive_df = get_df(dType=dType)
    minutesVeryActive_df = minutesVeryActive_df.rename(columns={'value': 'Very Active'})

    bubble_df = steps_per_day_df.merge(minutesVeryActive_df, on='dateTime', how='left')
    bubble_df = bubble_df.merge(sleep_duration_df, on='dateTime', how='left')
    bubble_df = bubble_df.dropna()

    bubble_df['Steps'] = bubble_df['Steps'].astype(int)
    bubble_df['Very Active'] = bubble_df['Very Active'].astype(int)
    bubble_df['Sleep duration'] = bubble_df['Sleep duration'].astype(int)

    return bubble_df

def get_sleep_level_summary_df():
    dType = "sleepSummary-stages"
    sleep_level_summary_df = get_df(dType=dType)

    rename_cols = {
        "deep": "Deep sleep minutes",
        "light": "Light sleep minutes",
        "rem": "REM minutes"
    }
    sleep_level_summary_df = (sleep_level_summary_df
                              .drop("wake", axis=1)
                              .rename(columns=rename_cols))

    bubble_df = get_bubble_df()
    steps = bubble_df[['dateTime', 'Steps']]

    sleep_level_summary_df = sleep_level_summary_df.merge(steps, on='dateTime', how='left')

    sleep_level_summary_df['Light sleep minutes'] = sleep_level_summary_df['Light sleep minutes'].astype(int)
    sleep_level_summary_df['Deep sleep minutes'] = sleep_level_summary_df['Deep sleep minutes'].astype(int)
    sleep_level_summary_df['REM minutes'] = sleep_level_summary_df['REM minutes'].astype(int)
    sleep_level_summary_df['Steps'] = sleep_level_summary_df['Steps'].astype(float)

    sleep_level_summary_df = sleep_level_summary_df.drop_duplicates().dropna()

    return sleep_level_summary_df

def get_new_complete_colors():
    global SLEEP_LEVEL_COLORS
    global ACTIVITY_LEVEL_COLORS
    global STEPS_COLOR

    # Rename keys to match what's used in the widget
    sleep_level_colors = {key + " Sleep": val for key, val in SLEEP_LEVEL_COLORS.items() if key != 'Awake'}
    sleep_level_colors['Awake'] = SLEEP_LEVEL_COLORS['Awake']
    # Get all colours in one dictionary
    colors = {}
    for d in [sleep_level_colors, ACTIVITY_LEVEL_COLORS, STEPS_COLOR]:
        for k, v in d.items():
            colors[k] = v
    colors['Sleep Efficiency'] = 'green'

    return colors

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

def heatmapPlots():
    """
    The function retrieves data from various sources and merges them into a single dataframe for
    plotting.
    
    :return: The function `heatmapPlots()` is returning a merged dataframe that includes data on sleep
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
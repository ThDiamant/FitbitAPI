# Import necessary modules
import time as tm
import warnings
import gather_keys_oauth2 as Oauth2 # This is a python file you need to have in the same directory as your code so you can import it
import functions as fun
import fitbit
import pymongo as mongo
import datetime as dt
from globals import *



# Enter CLIENT_ID and CLIENT_SECRET
# CLIENT_ID = '23QRRC'
# CLIENT_SECRET = '51922a48a2df4434cc20afaac4ee97b8'

# Authorize user
server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
server.browser_authorize()
# Save access and refresh tokens
ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])
EXPIRES_AT = str(server.fitbit.client.session.token['expires_at'])

# Create Fitbit object which will be used to get the data
auth2_client = fitbit.Fitbit(client_id=CLIENT_ID,
                             client_secret=CLIENT_SECRET,
                             expires_at=EXPIRES_AT,
                             oauth2=True,
                             access_token=ACCESS_TOKEN,
                             refresh_token=REFRESH_TOKEN)

client = mongo.MongoClient('localhost', 27017)

def importData(targetDate):
    now = dt.datetime.now() - dt.timedelta(hours=0, minutes=15)
    # Get data from the fitbit API
    oneDaySleepData = fun.get_sleep_data(targetDate, ACCESS_TOKEN)

    # Check if sleep data exist for the date we are looking at
    if (len(oneDaySleepData["sleep"]) > 0):
        # Get data related to general sleep info as well as the sleep time series
        sleepData = oneDaySleepData["sleep"][0]
        # Define which keys we want to keep
        sleepDataKeys = [key for key in sleepData.keys() if key not in skipKeys or key == "levels"]
        # For each key containing general information on sleep
        for sleepDataKey in sleepDataKeys:  
            # Dictionary that will hold the 'data' entry of the document
            dataDict = {}    
            # 'levels' contains the time series data
            if sleepDataKey == "levels":
                sleepLevelsData = sleepData[sleepDataKey]
                for key in sleepLevelsData.keys():
                    if key == "data" or key == "shortData":
                        documentType = f"sleepLevelsData-{key}"
                        for dataPoint in sleepLevelsData[key]:
                            dataDict = {}
                            # Convert string date to datetime.datetime so that's saved correctly in mongo
                            dataDict["dateTime"] = fun.to_datetime(dataPoint["dateTime"])
                            if (dataDict["dateTime"] > now):
                                continue
                            dataDict["level"] = dataPoint["level"]
                            dataDict["value"] = dataPoint["seconds"]
                            fun.create_and_save_document(fitbitCollection, documentType, dataDict)
            else:
                documentType = "sleep-{}".format(sleepDataKey)
                # Convert datetime.date to datetime.datetime so that's saved correctly in mongo
                dataDict["dateTime"] = fun.to_datetime(targetDate)
                if (dataDict["dateTime"] > now):
                    continue
                else:
                    if isinstance(sleepData[sleepDataKey], str):
                        dataDict["value"] = fun.to_datetime(sleepData[sleepDataKey])
                    else:
                        dataDict["value"] = sleepData[sleepDataKey]
                    fun.create_and_save_document(fitbitCollection, documentType, dataDict)
        # Get data related to summary sleep info
        sleepSummaryData = oneDaySleepData["summary"]
        # For each key containing summary information on sleep
        for sleepSummaryDataKey in sleepSummaryData.keys():
            documentType, dataDict = fun.get_summary_key_data(sleepSummaryData, sleepSummaryDataKey, targetDate)
            fun.create_and_save_document(fitbitCollection, documentType, dataDict)
    else:
        warnings.warn(f"Could not find sleep data for {targetDate}.")
    print(f'Loaded sleep data until {now}.')

    # Get activity data from the fitbit API
    oneDayActivityData = fun.get_activity_data(targetDate, auth2_client)

    # For each kind of activity
    for activityTypeKey in oneDayActivityData.keys():
        for key in oneDayActivityData[activityTypeKey].keys():
            # Check if activity data exist for the date and type of activity we are looking at
            if len(oneDayActivityData[activityTypeKey][key]) > 0:
                documentType = key.replace('activities-',"")
                if "intraday" not in key:
                    dataDict = {}
                    dataDict["dateTime"] = fun.to_datetime(oneDayActivityData[activityTypeKey][key][0]["dateTime"])
                    if (dataDict["dateTime"] > now):
                        continue
                    dataDict["value"] = int(oneDayActivityData[activityTypeKey][key][0]["value"])
                    fun.create_and_save_document(fitbitCollection, documentType, dataDict)
                else:
                    for dataPoint in oneDayActivityData[activityTypeKey][key]["dataset"]:
                        dataDict = {}
                        dataDict["dateTime"] = fun.to_datetime(targetDate, time = dataPoint["time"])
                        if (dataDict["dateTime"] > now):
                            continue
                        dataDict["value"] = int(dataPoint["value"])
                        fun.create_and_save_document(fitbitCollection, documentType, dataDict)
            else:
                warnings.warn(f"Could not find {activityTypeKey}-{key} data for {targetDate}.")
    print(f'Loaded activity data until {now}.')

# Check if the connection to the db was successful
try:
    db = client.admin
    server_info = db.command('serverStatus')
    print('Connection to MongoDB server successful.')
    
except mongo.errors.ConnectionFailure as e:
    print('Connection to MongoDB server failed: %s' % e)


USER_UUID = "3cc4e2ee-8c2f-4c25-955b-fe7f6ffcbe44"
DB_NAME = "fitbit"
DATA_COLLECTION_NAME = "fitbitCollection"


# Connect to the fitbitDb and the collection where the data are stored or create them if they don't exist
fitbitDb = client[DB_NAME]
fitbitCollection = fun.check_create_collection(fitbitDb, DATA_COLLECTION_NAME)

# Keys in the returned
skipKeys = ["levels", "infoCode", "logId", "logType", "type", "dateOfSleep"]

currentDate = dt.datetime.now().date()
while True:
    try:
        # Use now before 15 minutes. This is due to the fact that fitbit does not update it's data right away 
        now = dt.datetime.now() - dt.timedelta(hours=0, minutes=15)
        single_date = now.date()

        if (now.day != currentDate.day):
            # Make sure that the date from the previous date have been imported
            importData(currentDate)
            currentDate = now
        importData(single_date)
    except Exception as e:
        print('Error: %s' % e)

    # Wait for 15 minutes before making the next request
    # By doing so we can make sure that threshold of 150 queries per hour wont be reached
    tm.sleep(15*60)


# Exploring Fitbit data with MongoDb and Python Streamlit

Internet of Things is a highly increased domain of computer science. On part of it's data is generated through digital watches like Fitbit. Fitbit has an API which can be used in order to fetch the collected data such as steps, sleeps stages, heartbeat rate etc. 

With the help of MongoDb, a document based database, we can save those data in order to process them according to our needs.

In this tutorial, we will go through the process of installing and set up MongoDb, creating Fitbit account, saving Fitbit data to a MongoDb and using the saved data to display the data in python by using Streamlit library.

More specifically this article contains the following sections:

A. Installing and setting up MongoDb.

B. Creation of Fitbit account.

C. Fetching of Fitbit data

D. Saving of data to MongoDb.

E. Streamlit setup to display the data

F. Usage of Machine Learing algorith in order to generate knowledge

G. Implementation of python code to simulate streaming

## Installing and setting up MongoDb

To get started, download MongoDB Community Edition from the official website (https://www.mongodb.com/try/download/community).

Select the desired version and OS system. For Windows execute the MSI file and install MongoDB Compass

Then we need to add the bin folder (usually the path to the folder will look like `C:\Program Files\MongoDB\Server\6.0\bin`) to the environmental variables of Windows.

Open a terminal and execute the following to make sure that MongoDB is installed correctly:

``` bash
    mongod --version
```

Create a data directory for MongoDB. Open a terminal and run the following command:

``` bash
    mkdir C:\data\db
```

This command will create a directory named "db" in the "data" directory on your C drive. MongoDB uses this directory to store its data.

Then to start the server you need to execute the following:

``` bash
    mongod
```

## Creation of Fitbit developer account

## Interacting with the Fitbit API

In this section we will be using the `python-fitbit` and the `requests` modules to get data from the Fitbit API. This is not the only way to do it, for example, a simple alternative would be to use the [Fitbit Web API Explorer](https://dev.fitbit.com/build/reference/web-api/explore/).

The steps taken here are largely outlined in this (https://towardsdatascience.com/using-the-fitbit-web-api-with-python-f29f119621ea) Towards Data Science article.

First we have to proceed with the authorization from the Fitbit API and also define the Fitbit object in python which will be used to make some GET requests to the Fitbit API. In order to perform the authentication we will need the `CLIENT_ID` and `CLIENT_SECRET` which we had save during the creation of the Fitbit account. So, to proceed with authentication the following code should be executed:

``` python
# Authorize user
server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
server.browser_authorize()
# Save access and refresh tokens
ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])
EXPIRES_AT = str(server.fitbit.client.session.token['expires_at'])
```

Upon execution of this codeblock, we will be redirected to another tab and we will be asked to login into our Fitbit account. Upon doing that we will see a page that should say something like "You are now authorized to access the Fitbit API!".

Given that the authentication has been completed as expected now we have an access token, a refresh token and the datetime when the access token will expire. By using them, we can proceed with the initialization of Fitbit object. So, to initialize it, the following code should be executed:

``` python
auth2_client = fitbit.Fitbit(client_id = CLIENT_ID,
                             client_secret = CLIENT_SECRET,
                             expires_at = EXPIRES_AT,
                             oauth2 = True,
                             access_token = ACCESS_TOKEN,
                             refresh_token = REFRESH_TOKEN)
```

## Fetching of Fitbit data and saving to MongoDb

Now we are set and ready to proceed with the fetching Fitbit data. In this tutorial we will focus into two categories of data, the Sleep related data and the Activity related data or to be more presise the data which have to do with the level of our activity and the data which have to do with the number of our steps.

To get the Sleep related data we will use `requests` module, due to the fact that `python-fitbit` module has not been updated for a long time and it is using an older API version which is hardcoded in it's codebase. So, as we need to use version 1.2 instead of version 1 in order to get the Sleep related data in the desired format, we will perform this request manually through `requests` module.

Therefore, what we will do is to use the Fitbit Web API Explorer to get the CURL of the endpoint we want to draw data from, converted it to python using the requests module and get the data we need.

An important note here is that there is a rate limit for each user who has consented to share their data. This limit is **150 API requests per hour** and it resets at the top of each hour.

Having said that, we can move on to the construction of the request. We are gonna need a header object for our request which should contains the access token. An example of the request construction is visible below:

``` python
    # Make API get request
    headers = {
        'accept': 'application/json',
        'authorization': 'Bearer {}'.format(ACCESS_TOKEN),
    }
    try:
        response = req.get('https://api.fitbit.com/1.2/user/-/sleep/date/{}.json'.format(date), 
                            headers = headers)
    except fitbit.exceptions.HTTPTooManyRequests as e:
        tryAfterMin = e.retry_after_secs/60
        errorMessage = str(e) + ", please try again after {:.1f} min.".format(tryAfterMin)
        raise Exception(errorMessage)
```

The response object contains every information we need for the Sleep related data of the given date. 

Before checking out how to add those data to MongoDb, let's also have a look on how we fetch the Activity data. In order to fetch them we used the `python-fitbit` module and we will target to specific resources (steps and minutes active/sedentary). The first thing we have to do is to create a list with the target resources in order to minimize the code duplication e.g. "activities/steps", "activities/minutesVeryActive" etc. Then based on the selected resource, we will determine which should be the detail level of the data (1, 5 or 15 minutes). For steps we select 1 minute in have as more detailed information as possible while for the rest resource we select 15 minutes due to the fact that 1 minute will return binary values (1 or 0) based on the activity on this minute.

So, to sum-up, the code which we executed in order to perform this task was the following:

``` python
    for resource in resources:
        if resource == "steps":
            detailString = "1min"
        else:
            detailString = "15min"
        
        # Use fitbit module to make the API get request
        data[resource] = auth2_client.intraday_time_series(resource, date, detail_level = detailString)
```

## Set up MongoDb and saving Fitbit data to a database

In this tutorial we will use `pymongo` module in order to communicate with the MongoDb we installed in a previous section. The first thing we have to do, it to establic connection with the database. To do so execute:

``` python
import pymongo as mongo
client = mongo.MongoClient('localhost', 27017)

# Check if the connection to the db was successful
try:
    db = client.admin
    server_info = db.command('serverStatus')
    print('Connection to MongoDB server successful.')
    
except mongo.errors.ConnectionFailure as e:
    print('Connection to MongoDB server failed: %s' % e)
```

The next step is to create a new database and a collection in it in order to save the data. To do so, execute the following code:

``` python
fitbitDb = client[DB_NAME]
fitbitCollection = fitbitDb.create_collection(COLLECTION_NAME)
```

One last step before inserting the data to our Mongo database is to create an index. This index will help up both in performance and especially in the validity of our data. With the usage of a custom index, we are able to use some fields in order to check if there is already a document with the same values in them. In our case, we can use `resource type` and `dateTime` fields for the index as we already know that if a document has the same value in both fields with a document in our database and tries to be added to the database then this document should be rejected to avoid duplicates.

So, to do so, we executed the following code:

``` python
fitbitIndex = [('type', mongo.ASCENDING), ('data.dateTime', mongo.ASCENDING)]
# Check if the index exists
if indexName not in [fitbitIndex['name'] for fitbitIndex in collection.list_indexes()]:
    # Create the index if it does not exist
    collection.create_index(fitbitIndex, name = indexName, unique=True)
```

Now we have ensure that we wont have duplicate data to our database so we are ready to proceed with the loading of Fitbit data to MongoDb. By using the manual request as well as the `fitbit-python` module we perform multiple requests for every day starting from 28/03/2023 until 30/04/2023. For each request we manipulate the data in order to create a dictionary with the following format:


Each object represents a document. The last part to save a document to our database is to use the collection which was created before. So, by using this collection we execute the following code for each document:

``` python
try:
    # Insert myDocument in Mongo
    result = fitbitCollection.insert_one(myDocument)
    # Check if the document was inserted successfully
    if not result.inserted_id:
        raise Exception(f"Document {myDocument} not inserted.")
    # If record already exists an exception is raised due to the index
except mongo.errors.DuplicateKeyError:
    pass
```








pip install plotly scikit-learn scipy matplotlib
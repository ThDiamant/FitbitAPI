# Exploring Fitbit data with MongoDb and Python Streamlit

Internet of Things is a highly increased domain of computer science. On part of it's data is generated through digital watches like Fitbit. Fitbit has an API which can be used in order to fetch the collected data such as steps, sleeps stages, heartbeat rate etc. 

With the help of MongoDb, a document based database, we can save those data in order to process them according to our needs.

In this tutorial, we will go through the process of installing and set up MongoDb, creating Fitbit account, saving Fitbit data to a MongoDb and using the saved data to display the data in python by using Streamlit library.

More specifically this article contains the following sections:

A. Installing and setting up MongoDb.

B. Creation of Fitbit account.

C. Fetching of Fitbit data and saving to MongoDb.

D. Streamlit setup to display the data

E. Usage of Machine Learing algorith in order to generate knowledge

F. Implementation of python code to simulate streaming

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

Upon execution of this codeblock, we will be redirected to another tab and we will be asked to login into our Fitbit account. Upon doing that we will see a page that should say something like "Authentication Complete, you may close this tab".

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

Therefore, what we will do is to use the Fitbit Web API Explorer to get the CURL of the endpoint we want to draw data from, converted it to python using the requests module, and get the data we need.

For Activity data we used the `python-fitbit` module. We have used several different resources to quantify activity (steps, minutes active/sedentary).

An important note here is that there is a rate limit for each user who has consented to share their data, and this limit is **150 API requests per hour**. This resets at the top of each hour.

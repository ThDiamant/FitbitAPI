import datetime
import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient

def load_data(date):
    client = MongoClient('localhost', 27017)
    fitbitDb = client['fitbit']
    # Create a new collection called "fitbitCollection"
    fitbitCollection = fitbitDb['fitbitCollection']

    myquery = { 'type': 'sleepLevelsData', 'data.dateTime': {'$regex':'^{}'.format(date)} }
    mydoc = fitbitCollection.find(myquery)
    # df = pd.DataFrame(list(mydoc))
    df = pd.DataFrame(columns=['dateTime', 'level', 'seconds'])
    for doc in mydoc:
        new = pd.DataFrame.from_dict([doc["data"]])
        df = pd.concat([df, new])
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    return df

st.title('Fitbit App')

d = st.date_input(
    "Select date for your data",
    datetime.date(2023, 4, 20))

data = load_data(d)
# st.subheader('Raw data')
# st.write(data)
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


# Histogram
st.subheader('Number of pickups by hour')
hist_values = np.histogram(
    data["dateTime"].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)


# Map
# st.subheader('Map of all pickups')
# st.map(data)
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data["dateTime"].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)
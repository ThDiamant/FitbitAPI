import datetime
import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient

def load_data(date=None, dType=None, columns=None):
    client = MongoClient('localhost', 27017)
    fitbitDb = client['fitbit']
    # Create a new collection called "fitbitCollection"
    fitbitCollection = fitbitDb['fitbitCollection']

    if dType=='sleepLevelsData':
        myquery = { 'type': dType, 'data.dateTime': {'$regex':'^{}'.format(date)} }
    else:
        myquery = { 'type': dType }
    mydoc = fitbitCollection.find(myquery)
    # df = pd.DataFrame(list(mydoc))
    df = pd.DataFrame(columns=columns)
    for doc in mydoc:
        new = pd.DataFrame.from_dict([doc["data"]])
        df = pd.concat([df, new])
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    return df

st.title('Fitbit App')

today = datetime.date(2023, 4, 20)
start_clnd = datetime.date(2023, 3, 29)

date = st.date_input(
    "Select date for your data",
    datetime.date(2023, 4, 20))

dType = st.selectbox(
    'What type of data would you like to check?',
    ('sleepLevelsData', 'sleep-duration', 'sleep-efficiency', 'sleep-endTime', 'sleep-isMainSleep', 'sleep-minutesAfterWakeup', 'sleep-minutesAsleep'))

if dType=='sleepLevelsData':
    columns=['dateTime', 'level', 'seconds']

data = load_data(date, dType, columns)

# st.subheader('Raw data')
# st.write(data)
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader(f'Sleep Level Data')
dType ='sleepLevelsData'
columns=['dateTime', 'level', 'seconds']
barChart_sleepLevelsData = load_data(date, dType, columns)
barChart_sleepLevelsData['dateTime'] = pd.to_datetime(barChart_sleepLevelsData['dateTime']).dt.date
barChart_sleepLevelsData = barChart_sleepLevelsData.groupby('level').agg({'seconds': 'sum'})
st.bar_chart(barChart_sleepLevelsData)

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

################## Date Slider to retrieve TS Data ##################

date_range = st.slider(
    "When do you start?",
    value=(start_clnd, today),
    format="MM/DD/YY")
# st.write("Date range:", date_range)

start_date = date_range[0]
end_date = date_range[1]
st.write(f"Start Date: {start_date}")
st.write(f"End Date: {end_date}")


################## Sleep Duration Line Chart ##################
st.subheader(f'Sleep Duration')
date = today
dType ='sleep-duration'
columns=['dateTime', 'value']
lineChart_sleepDuration = load_data(date, dType, columns)

lineChart_sleepDuration.set_index('dateTime', inplace=True)
lineChart_sleepDuration['Sleep Duration'] = lineChart_sleepDuration['value']/3600000
lineChart_sleepDuration.drop(columns=['value'], inplace=True)
st.line_chart(lineChart_sleepDuration)
################## Sleep Duration Line Chart ##################

################## Sleep Efficiency Line Chart ##################
st.subheader(f'Sleep Efficiency')
date = today
dType ='sleep-efficiency'
columns=['dateTime', 'value']
lineChart_sleepEfficiency = load_data(date, dType, columns)

lineChart_sleepEfficiency.set_index('dateTime', inplace=True)
lineChart_sleepEfficiency['Sleep Efficiency'] = lineChart_sleepEfficiency['value']
lineChart_sleepEfficiency.drop(columns=['value'], inplace=True)
st.line_chart(lineChart_sleepEfficiency)
################## Sleep Efficiency Line Chart ##################
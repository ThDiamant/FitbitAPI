import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

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
# PACKAGE INSTALLATION
# pip install google_play_scraper
# pip install transformers
# pip install seaborn

# LIBRARIES AND MODULES
import pandas as pd
import seaborn as sns
from google_play_scraper import app
from google_play_scraper import reviews_all
from google_play_scraper import Sort

# EXTRACTION OF REVIEWS FROM APP - Snapseed
app_rw = reviews_all("com.niksoftware.snapseed",sleep_milliseconds = 0,lang='en',country = 'us',sort = Sort.NEWEST)

# EXTRACTED REVIEWS IN JSON FORMAT CONVERTED TO NORMALIZED DATA STRUCTURE
app_rw_df = pd.json_normalize(app_rw)

print(app_rw_df.head())

# TRANSFORMER MODEL INITALIZATION
from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

print(app_rw_df.dtypes)

# CONVERT REVIEWS TO STRING DATATYPE IF IN OBJECT DATATYPE
app_rw_df['content'] = app_rw_df['content'].astype('str')

# APPLY LAMBDA FUNCTION ON CONTENT COLUMN AND STORE IT IN RESULT VARIABLE COLUMN
app_rw_df['result'] = app_rw_df['content'].apply(lambda x: sentiment_analysis(x))

app_rw_df['sentiment'] = app_rw_df['result'].apply(lambda x: (x[0]['Label']))
app_rw_df['s_score'] = app_rw_df['result'].apply(lambda x: (x[0]['score']))

s_result = app_rw_df['sentiment'].value_counts(normalize = True)

print(s_result)
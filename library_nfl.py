# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import requests
from bs4 import BeautifulSoup

# Own libraries
from library_nfl import *

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Show all columns in pandas
pd.set_option('display.max_columns', 500) 

# Graphing style
plt.style.use('seaborn-colorblind')

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def pull_data_stats(urls):
    # Dictionary to store all tables
    dataframes = {}

    # Loop through each url and import data
    for url in urls: 
        # The read_html function returns a list of all the tables found on the web page
        tables = pd.read_html(url)

        # Assuming the table you want is the first one in the list, you can access it like this
        table_df = tables[0]

        position = url.split('/')[5].split('.')[0].upper()

        # Drop the first level of the MultiIndex, keeping only the second level
        if position in ['QB', 'RB', 'WR', 'TE']:
                table_df.columns = table_df.columns.droplevel(level=0)


        table_df.columns = [x.lower() for x in table_df.columns]

        table_df['pos'] = position

        # Extract team
        table_df['team'] = table_df['player'].apply(lambda x: x.split('(')[1].replace(')', '') if '(' in x else None)

        # Remove team from name
        table_df['player'] = table_df['player'].replace('\s\(.+\)', '', regex=True)

        dataframes[position] = table_df

    # Return dictionary 
    return dataframes

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def pull_data_target_distribution(urls):
    # Loop through each url and import data
    for url in urls: 
        # The read_html function returns a list of all the tables found on the web page
        tables = pd.read_html(url)

        # Assuming the table you want is the first one in the list, you can access it like this
        table_df = tables[0]

        table_df.columns = [x.lower() for x in table_df.columns]

        # Convert avg to numeric
        table_df.rename(columns={'team': 'team_name'}, inplace=True)


    # Return dictionary 
    return table_df

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
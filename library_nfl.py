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
#plt.style.use('seaborn-colorblind')

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
        table_df['player'] = table_df['player'].replace(r'\s\(.+\)', '', regex=True)

        dataframes[position] = table_df

        # Drop column and rename team name
        table_df.drop(columns=['rost'], inplace=True)

    # Return dictionary 
    return dataframes

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def pull_data_snapcount(urls):
    # Dictionary to store all tables
    dataframes = {}

    # Loop through each url and import data
    for url in urls: 
        # The read_html function returns a list of all the tables found on the web page
        tables = pd.read_html(url)

        # Assuming the table you want is the first one in the list, you can access it like this
        table_df = tables[0]

        # Get position from URL
        position = url.split('/')[6].split('.')[0].upper()

        # Remove %%!
        table_df = table_df.replace(r'\%', '', regex=True)
        # Remove byes!
        table_df = table_df.replace('bye', '200', regex=True)

        table_df.columns = [x.lower() for x in table_df.columns]

        # Keep only relevant columns
        # table_df = table_df[['player', 'team', 'pos', 'avg']]

        # Convert avg to numeric
        table_df.rename(columns={'avg': 'avg_snap_pct'}, inplace=True)

        if '18' in table_df.columns:
            int_columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'ttl','avg_snap_pct']
        else:
            int_columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', 'ttl','avg_snap_pct']


        table_df[int_columns] = table_df[int_columns].astype(float)

        table_df = table_df.replace(200, np.nan)

        # Custom function to calculate the average snap count for each x when played
        def average_snapcount(x, pos1, pos2, limit=0):
            snapcounts = [snapcount for snapcount in x[pos1: pos2] if snapcount > limit]  # Selecting only week columns (from 3rd column onwards)
            return sum(snapcounts) / len(snapcounts) if len(snapcounts) > 0 else None

        # Create the new column with avg snap pct when played
        if '18' in table_df.columns:
            table_df['avg_snap_pct_played'] = table_df.apply(lambda x: average_snapcount(x, 2, 20), axis=1)
        else: 
            table_df['avg_snap_pct_played'] = table_df.apply(lambda x: average_snapcount(x, 2, 19), axis=1)

        # Create the new column with avg snap pct global
        if '18' in table_df.columns:
            table_df['avg_snap_pct_global'] = table_df.apply(lambda x: average_snapcount(x, 2, 20, -1), axis=1)
        else: 
            table_df['avg_snap_pct_global'] = table_df.apply(lambda x: average_snapcount(x, 2, 19, -1), axis=1)

        # Create position variable
        table_df['pos'] = position

        # Drop column
        table_df.drop(columns='avg_snap_pct', inplace=True)

        # Return dataframe
        dataframes[position] = table_df

    # Return dictionary 
    return dataframes

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def pull_data_target_distribution(urls, nfl_teams_dict):
    # Loop through each url and import data
    for url in urls: 
        # The read_html function returns a list of all the tables found on the web page
        tables = pd.read_html(url)

        # Assuming the table you want is the first one in the list, you can access it like this
        table_df = tables[0]

        # Clean column names
        table_df.columns = [x.lower() for x in table_df.columns]

        table_df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

        table_df.rename(columns=lambda x: x.replace('%', 'pct'), inplace=True)

        # Rename columns
        table_df.rename(columns={'team': 'team_name'}, inplace=True)

        # Create 
        table_df['team'] = table_df['team_name'].map(nfl_teams_dict)


    # Return dictionary 
    return table_df

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def get_percentiles(df, col):
    percentiles = []
    for i in range(10, 100, 5):
        percentiles.append(np.percentile(df[col], i))

    return percentiles


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Function to assign score
def score_column(x, df, col):
    # Find percentiles
    percentiles = get_percentiles(df, col)
    # Assign grade to each player
    score = 1
    for p in percentiles:
        if x[col] < p:
            return score
        score = score + 0.5
        
    return 10

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def merge_score(df1, df2, col):
    # Create provisional score column
    df1['score'] = df1.apply(lambda x: score_column(x, df1, col), axis=1)
    # Merge with new score df
    df2 = df2.merge(df1[['player', 'score']], how='left', on='player').rename(columns={'score': f'score_{col}'})

    # Check result
    return df2


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def create_radar_diagram(df, player, team):
    # Filter the DataFrame for the specific player and team
    player_data = df[(df['player'] == player) & (df['team'] == team)]

    # List of column names
    columns = df.columns.drop(['player', 'team'])

    # Get the scores for the player and team
    scores = player_data[columns].values[0]

    # Number of variables
    num_variables = len(columns)

    # Calculate the angle for each variable
    angles = np.linspace(0, 2 * np.pi, num_variables, endpoint=False)

    # Make the plot circular
    angles = np.concatenate((angles, [angles[0]]))
    scores = np.concatenate((scores, [scores[0]]))

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, linewidth=2, linestyle='solid', c='lightseagreen')
    ax.fill(angles, scores, alpha=0.1, c='lightseagreen')

    # Set the labels for each variable
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([x.replace('score_', '') for x in  columns])

    # Set the title
    ax.set_title(f"Radar Diagram for {player} ({team})")

    # Display the plot
    plt.show()
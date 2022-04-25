import pandas as pd

career = pd.read_csv('Datasets/Basketball_Reference/BR_Seasonal_Stats.csv')

career = career[(~career['MP'].str.contains('Did', na=False))]

career.head()

double_columns = ['G','GS','AGE','MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

career[double_columns] = career[double_columns].astype('double')

career.fillna(method = 'bfill', inplace=True)

career_stats = career.groupby('name').mean()

career_stats.drop('Unnamed: 0',axis=1, inplace=True)

#career_stats.to_csv('BR_Career_stats.csv')

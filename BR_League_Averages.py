from bs4 import BeautifulSoup as bs
import pandas as pd
import requests

url = 'https://www.basketball-reference.com/leagues/NBA_stats_per_game.html'

html = requests.get(url)

soup = bs(html.content,'html.parser')

headers = [th.getText() for th in soup.findAll('tr', limit=2)[1].findAll('th')]
headers.pop(0)
rows = soup.findAll('tr')[2:]
rows_data = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]

league_avg = pd.DataFrame(rows_data,columns=headers)

league_avg.to_csv('league_avgs.csv')

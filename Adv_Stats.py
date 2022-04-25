import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import duckdb

url = 'https://www.basketball-reference.com/leagues/NBA_2022_advanced.html'
html = requests.get(url)
soup = bs(html.content,'html.parser')
headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
headers.pop(0)
rows = soup.findAll('tr')[1:]
rows_data = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]
df = pd.DataFrame(rows_data,columns=headers)
df['Season'] = '2021-2022'

for i in range(1950,2022):
    url = f'https://www.basketball-reference.com/leagues/NBA_{i}_advanced.html'
    html = requests.get(url)
    soup = bs(html.content,'html.parser')

    rows = soup.findAll('tr')[1:]
    rows_data = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]
    pre_df = pd.DataFrame(rows_data,columns=headers)
    pre_df['Season'] = f'{i-1}-{i}'
    
    df = pd.concat([df,pre_df])

df.dropna(inplace=True)

pre_rank = duckdb.query("""select *,row_number() over( partition by Player,Season order by G desc) as row from df order by Player,Season""").to_df()

adv_df = duckdb.query("""select * from pre_rank where row = 1 order by Player,Season""").to_df()  

columns = list(adv_df.columns)

adv_df.drop('row',axis=1,inplace=True)
adv_df.drop(columns[18],axis=1,inplace=True)
adv_df.drop(columns[23],axis=1,inplace=True)

adv_df['Seasonnum'] = adv_df.groupby(['Player'])['Season'].rank(method='dense')

five_seasons = list(adv_df[adv_df['Seasonnum'] == 5]['Player'])

five_season_stats = adv_df[adv_df['Player'].isin(five_seasons)]
five_season_stats = five_season_stats[(five_season_stats['Seasonnum'] <= 5)]

league_avg_per = df[['Season','PER']]
league_avg_per['PER'] = league_avg_per['PER'].replace('','0').astype('float')
league_avg_per_grouped = league_avg_per.groupby('Season').mean().reset_index()
league_avg_per_grouped = league_avg_per_grouped[league_avg_per_grouped['Season']>'1950-1951']

faulty_players = ['Abdel Nader','Cheick Diallo','Damyean Dotson','Wes Iwundu','Shaquille Harrison','Semi Ojeleye','Sam Dekker','Malik Monk','Luke Kornet','Larry Nance Jr.','Justin Jackson','Juancho Hernangomez','Josh Hart','Jordan Bell','George King','Gary Payton II','Emmanuel Mudiay','Dejounte Murray','DJ Wilson']

adv_df = adv_df[~adv_df['Player'].isin(faulty_players)]
# adv_df.to_csv('BR_Adv_stats.csv')
# five_season_stats.to_csv('BR_Adv_Stats_Five.csv')
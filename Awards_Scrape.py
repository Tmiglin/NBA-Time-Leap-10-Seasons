from bs4 import BeautifulSoup as bs
import pandas as pd
import requests 
#%%
# HOF Players


url = 'https://www.basketball-reference.com/awards/hof.html'

html = requests.get(url)

soup = bs(html.content,'html.parser')

headers = [th.getText() for th in soup.findAll('tr', limit=2)[1].findAll('th')]
rows = soup.findAll('tr')[2:]
rows_data = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]

print(rows_data)
headers.pop(0)

hof = pd.DataFrame(rows_data,columns=headers).dropna()

hof_players = list(hof[hof['Category'] == 'Player'].Name)

hof_list_space = [" ".join(player.split(' ')[:2]) for player in hof_players]

removable = ['WNBA','Player',"Int'l",'CBB','NBL','Coach']

for i,player in enumerate(hof_list_space):
    for item in removable:
        hof_list_space[i] = hof_list_space[i].replace(item," ")

hof_list = [item[:-4] for item in hof_list_space]


#%%
# All-NBA Teams


url = 'https://www.basketball-reference.com/awards/all_league.html'

html = requests.get(url)

soup = bs(html.content,'html.parser')

headers = [th.getText() for th in soup.findAll('tr')[0].findAll('th')]
headers.pop(0)
headers = ['lg','tm','vt','C','F','F','G','G']
rows = soup.findAll('tr')[1:]
rows_data = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]

all_nba = pd.DataFrame(rows_data,columns=headers)

all_nba.drop('vt',axis=1,inplace=True)
#%%
# All-Defensive Teams

url = 'https://www.basketball-reference.com/awards/all_defense.html'

html = requests.get(url)

soup = bs(html.content,'html.parser')

headers = ['lg','tm','1','2','3','4','5']
rows = soup.findAll('tr')[1:]
rows_data = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]

all_defense = pd.DataFrame(rows_data,columns=headers)

#%%
# All-Rookie Teams

url = 'https://www.basketball-reference.com/awards/all_rookie.html'

html = requests.get(url)

soup = bs(html.content,'html.parser')

headers = ['lg','tm','G','G','F','F','F']
rows = soup.findAll('tr')[1:]
rows_data = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]

all_rookie = pd.DataFrame(rows_data,columns=headers)

#%%
# Individual Awards
awardees = []
player_awards = {} 
awards = ['mvp',\
        'dpoy',\
        'smoy',\
        'mip',\
        'finals_mvp']

roy_url = 'https://www.basketball-reference.com/awards/roy.html'
html = requests.get(roy_url)
soup = bs(html.content, 'html.parser')

headers = [th.getText() for th in soup.findAll('tr')[1].findAll('th')]
rows = soup.find('table', {"id" : "roy_NBA"}).findAll('tr')[2:]
rows_data = [[td.getText() for td in rows[i]] for i in range(len(rows))]
roy = pd.DataFrame(rows_data,columns=headers)
awardees.append(list(roy['Player']))
  
for award in awards:
    url = f'https://www.basketball-reference.com/awards/{award}.html'
    html = requests.get(url)
    soup = bs(html.content, 'html.parser')
    headers = [th.getText() for th in soup.find('table', {"id" : f"{award}_summary"}).findAll('tr')[0].findAll('th')]
    rows = soup.find('table', {"id" : f"{award}_summary"}).findAll('tr')[1:]
    print(rows)
    rows_data = [[td.getText() for td in rows[i]] for i in range(len(rows))]
    names = pd.DataFrame(rows_data,columns=headers)
    awardees.append(list(zip(names['Player'],names['Count'])))

player_awards['roy'] = awardees[0]
player_awards['mvp'] = awardees[1]
player_awards['dpoy'] = awardees[2]
player_awards['smoy'] = awardees[3]
player_awards['mip'] = awardees[4]
player_awards['finals_mvp'] = awardees[5]
    




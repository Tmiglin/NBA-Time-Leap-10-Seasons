from Awards_Scrape import player_awards, hof_list
import pandas as pd
import duckdb


seasonal_stats = pd.read_csv('Datasets/Basketball_Reference/Adv_Stats/BR_Adv_Stats_Five.csv')
general_info = pd.read_csv('Datasets/Basketball_Reference/Adv_Stats/BR_Adv_Stats_Five.csv')[['Player','Pos','Tm']]

mvp = pd.DataFrame(player_awards['mvp'], columns = ['player','mvp_count'])
fmvp = pd.DataFrame(player_awards['finals_mvp'], columns = ['player','finals_mvp_count'])
roy = pd.DataFrame(player_awards['roy'], columns = ['player'])
roy['roy_count'] = 1
mip = pd.DataFrame(player_awards['mip'], columns = ['player','mip_count'])
dpoy = pd.DataFrame(player_awards['dpoy'], columns = ['player', 'dpoy_count'])
smoy = pd.DataFrame(player_awards['smoy'], columns = ['player', 'smoy_count'])

mvp['mvp_count'] = mvp['mvp_count'].astype('int64')
fmvp['finals_mvp_count'] = fmvp['finals_mvp_count'].astype('int64')
mip['mip_count'] = mip['mip_count'].astype('int64')
dpoy['dpoy_count'] = dpoy['dpoy_count'].astype('int64')
smoy['smoy_count'] = smoy['smoy_count'].astype('int64')

award_list = [mvp,fmvp,roy,mip,dpoy,smoy]

awards = pd.concat(award_list).fillna(0)

awards = awards.groupby('player').sum()

awards['total_awards'] = awards.sum(axis=1)
awards.index = awards.index.str.replace('\\*','')
awards.index = awards.index.str.replace('\\(Tie\\)','')

seasonal_stats.drop('Unnamed: 0',axis=1,inplace=True)

seasonal_stats_group = seasonal_stats.groupby('Player').mean()

seasonal_stats_group['HOF'] = [1 if name in hof_list else 0 for name in seasonal_stats_group.index]

cluster_set = seasonal_stats_group.join(awards)
cluster_set.fillna(0,inplace=True)
cluster_set.drop('Seasonnum',axis=1,inplace=True)

general_info['Player'] = general_info['Player'].str.replace('\\*','')
general_info = general_info.groupby('Player')['Pos'].value_counts().reset_index(name='Freq')
general_info = duckdb.query('''select Player,Pos
                            from (select *, row_number() over(partition by Player order by Freq desc) as rn from general_info) a
                            where rn = 1 order by Player''').to_df()
general_info.index = general_info['Player']
cluster_set.index = general_info.index
seasonal_stats['Player'] = [item for item in general_info.index for i in range(5)]

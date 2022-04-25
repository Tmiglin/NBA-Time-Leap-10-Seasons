import pandas as pd
import numpy as np
import duckdb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.ar_model import AutoReg
from kneed import KneeLocator
from Dataset_Merge import cluster_set, general_info, seasonal_stats,awards
from Awards_Scrape import hof_list
from Adv_Stats import adv_df

#%%
# Create the normalized feature set to be used for Clustering by Position
all_five_df = seasonal_stats
cluster_set = cluster_set.join(general_info).drop('Player',axis=1)

cluster_set.loc[cluster_set['Pos'] == 'F-C','Pos'] = 'PF'
cluster_set.loc[cluster_set['Pos'] == 'F-G','Pos'] = 'SF'
cluster_set.loc[cluster_set['Pos'] == 'G-F','Pos'] = 'SG'

cluster_positional_sets_avg = {pos:{'Cluster_Set' : cluster_set[cluster_set['Pos']==pos].drop('Pos',axis=1).drop('VORP',axis=1).drop('G',axis=1).drop('Age',axis=1)} for pos in cluster_set['Pos'].unique()}
#%%
# Use the elbow method to decide on the proper amount of clusters (N=5)
i=1
for pos,item in cluster_positional_sets_avg.items():
    sse = []
    silhouette_coefficients = []
    for k in range(2,11):
        kmeans = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(item['Cluster_Set'])
        sse.append(kmeans.inertia_)
        score = silhouette_score(item['Cluster_Set'], kmeans.labels_)
        silhouette_coefficients.append(score)
    cluster_positional_sets_avg[pos]['SSE'] = sse
    cluster_positional_sets_avg[pos]['Sil_Cof'] = silhouette_coefficients
    knee_range = [i for i in range(2,11)]
    knee = KneeLocator(knee_range,sse, curve='convex',direction='decreasing')
    cluster_positional_sets_avg[pos]['Clusters'] = knee.knee
    kmeans = KMeans(init="random", n_clusters=item['Clusters'], n_init=10, max_iter=300, random_state=42)
    kmeans.fit(item['Cluster_Set'])
    item['Cluster_Set']['Clusters_init'] = kmeans.labels_
    item['Player_Cluster_info'] = item['Cluster_Set'].reset_index()[['Player','Clusters_init','PER','DWS','OWS','TOV%']]
    item['Player_Cluster_info']['Pos'] = pos
    if i==1:
        initial_clustering = item['Player_Cluster_info']
    else:
        initial_clustering = pd.concat([initial_clustering,item['Player_Cluster_info']])
    i+=1

#%%
# Create Subsets for Time Series Prediction and Analysis
current_players_under_10 = adv_df.loc[(adv_df['Season']=='2021-2022')&(adv_df['Seasonnum']>=5)&(adv_df['Seasonnum']<10),['Player','Pos']]
ten_seasons = [item.replace('*','') for item in adv_df[adv_df['Seasonnum']>=10]['Player'].unique()]

grab_predictive_players = adv_df[adv_df['Player'].isin(current_players_under_10['Player'])]
Player_Sets = {player:{'Feature_Set': grab_predictive_players.loc[grab_predictive_players['Player']== player].iloc[:,5:].reset_index(drop=True).drop('Season',axis=1).drop('VORP',axis=1)} for player in current_players_under_10['Player']}

pred_subset = grab_predictive_players[['Player','PER','Seasonnum']].sort_values(['Seasonnum','Player'])

pred_subset2 = pred_subset.copy()
pred_subset2['Previous_Year_PER'] = pred_subset2.groupby(['Player'])['PER'].shift().astype('float')
pred_subset2['PER'] = pred_subset2['PER'].astype('float')
pred_subset2['Change_In_PER'] = pred_subset2['PER']-pred_subset2['Previous_Year_PER']
pred_subset2 = pred_subset2[pred_subset2['Seasonnum']>1]
pred_subset2['Player_ID'] = pred_subset2.groupby('Player').ngroup()
#%%
# Sliding window validation
# This means that we are going to simulate training the model in all the seasons up to the one we want to forecast, and evaluate our score in the new season.

def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))

mean_error = []
for season in range(3,6):
    train = pred_subset2[pred_subset2['Seasonnum'] < season]
    val = pred_subset2[pred_subset2['Seasonnum'] == season]

    p = abs(val['Previous_Year_PER'].values)

    error = rmsle(abs(val['PER'].values), p)
    print('Season %d - Error %.5f' % (season, error))
    mean_error.append(error)
print('Mean Error = %.5f' % np.mean(mean_error))

#%%
# Try to beat the baseline now
# As a first model, let's train a Random Forest. Besides being a strong model with structured data (like the one we have), we usually can already get a very good result by just setting a high number of trees.

mean_error = []
for season in range(3,6):
    train = pred_subset2[pred_subset2['Seasonnum'] < season]
    val = pred_subset2[pred_subset2['Seasonnum'] == season]

    xtr, xts = train.drop(['PER','Player'], axis=1), val.drop(['PER','Player'], axis=1)
    ytr, yts = abs(train['PER'].values), abs(val['PER'].values)

    mdl = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
    mdl.fit(xtr, ytr)

    p = mdl.predict(xts)

    error = rmsle(yts, p)
    print('Week %d - Error %.5f' % (season, error))
    mean_error.append(error)
print('Mean Error = %.5f' % np.mean(mean_error))

#%%
# Test the Autoreg model against known seasons
fails=[]
for player,item in Player_Sets.items():
    predictor_list = list(item['Feature_Set'].iloc[:,:-1])
    print(player)
    for predictor in predictor_list:
        df = item['Feature_Set'].reset_index()
        df[df[predictor]==''] = '0'
        df[predictor] = df[predictor].astype('float')
        
        train = df[predictor][:len(df)-2]
        test = df[predictor][len(df)-2:]
        
        try:
            ar_model = AutoReg(train, lags=1).fit()
        except:
            fails.append(f'Failure to fit model for {player} on predictor {predictor}')
            
        pred = ar_model.predict(start=len(train), end=(len(df)-1), dynamic=False)
        error = rmsle(abs(test),abs(pred))
            
        if predictor == 'MP':
            item['Prediction_Test'] = {predictor : {'PredVTest': [pred,test],'Error' : error}}
            if error > .50:
                item['Prediction_Test'][predictor]['Large_error'] = 1
            else:
                item['Prediction_Test'][predictor]['Large_error'] = 0
        else:
            item['Prediction_Test'][predictor] = {'PredVTest': [pred,test],'Error' : error}
            if error > .50:
                item['Prediction_Test'][predictor]['Large_error'] = 1
            else:
                item['Prediction_Test'][predictor]['Large_error'] = 0
        
        
#%%
# Predict enough seasons so every eligible player has 10 seasons
for player,item in Player_Sets.items():
    predictor_list = list(item['Feature_Set'].iloc[:,:-2])
    df = item['Feature_Set'].reset_index()
    print(player)
    for predictor in predictor_list:
        df[df[predictor]==''] = '0'
        df[predictor] = df[predictor].astype('float')
        
        try:
            ar_model = AutoReg(df[predictor], lags=1).fit()
        except:
            fails.append(f'Failure to fit model for {player} on predictor {predictor}')
            
        pred = ar_model.predict(start=len(df)+1, end=10, dynamic=False)
        
        if predictor == 'MP':
            item['Predictions'] = {predictor:pred}
        else:
            item['Predictions'][predictor] = pred
#%%
# With the predictions now done, a new cluster set needs to be merged that will cluster on an average of over 10 seasons


for player,item in Player_Sets.items():
    Player_Sets[player]['MergePredictions'] = pd.DataFrame(Player_Sets[player]['Predictions'])
    Player_Sets[player]['Ten_Seasons'] = pd.concat([Player_Sets[player]['Feature_Set'],Player_Sets[player]['MergePredictions']]).reset_index(drop=True)
    tens = Player_Sets[player]['Ten_Seasons']
    for column in tens.columns:
        tens[tens[column]==''] = '0'
        tens[column] = tens[column].astype('float')
    Player_Sets[player]['Ten_Seasons']['Player'] = player
    Player_Sets[player]['Ten_Seasons']['Seasonnum'] = tens.index+1
    if player == 'Aaron Gordon':
        Ten_Seasons = Player_Sets[player]['Ten_Seasons']
    else:
        Ten_Seasons = Ten_Seasons.append(Player_Sets[player]['Ten_Seasons'], ignore_index=True)

#%%
# Group together two datasets to make for an easier concatenation
Ten_Seasons.iloc[:,:-2] = Ten_Seasons.iloc[:,:-2].astype('float')
Ten_Seasons['MP'] = Ten_Seasons['MP'].astype('int')
current_players_pos = pd.concat([current_players_under_10]*10).sort_values('Player').reset_index(drop=True)
Ten_Seasons['Pos'] = current_players_pos['Pos']

advc_df = pd.read_csv('Datasets/Basketball_Reference/Adv_Stats/BR_Adv_stats.csv')
Ten_or_more = advc_df[advc_df['Player'].isin(ten_seasons)].drop('Season',axis=1).drop('VORP',axis=1).drop('Unnamed: 0',axis=1)
Ten_or_more = Ten_or_more[Ten_or_more['Seasonnum']<=10].fillna(0)


faulty_players = duckdb.query('''select player,count(*) from Ten_or_more group by player having count(*)<10''').to_df()['Player']

Ten_or_more = Ten_or_more[~Ten_or_more['Player'].isin(faulty_players)]

Ten_or_more_pos = Ten_or_more.groupby('Player')['Pos'].value_counts().reset_index(name='Freq')
Ten_or_more_pos = duckdb.query('''select Player,Pos
                            from (select *, row_number() over(partition by Player order by Freq desc) as rn from Ten_or_more_pos) a
                            where rn = 1 order by Player''').to_df()
                            
Ten_or_more['Pos'] = [item for item in Ten_or_more_pos['Pos'] for i in range(10)]

Ten_or_more_avg = Ten_or_more.groupby(['Player','Pos']).mean().round(2).reset_index()

Ten_or_more_avg['HOF'] = [1 if name in hof_list else 0 for name in Ten_or_more_avg['Player']]

Ten_Seasons_avg = Ten_Seasons.groupby(['Player','Pos']).mean().round(2).reset_index()
Ten_Seasons_avg['HOF'] = 0
#%%
#now join the current players set with the known 10 season players
FullTen = pd.concat([Ten_Seasons_avg,Ten_or_more_avg])

FullTen.loc[FullTen['Pos'] == 'C-PF','Pos'] = 'C'
FullTen.loc[FullTen['Pos'] == 'SF-SG','Pos'] = 'SF'
FullTen.loc[FullTen['Pos'] == 'SG-SF','Pos'] = 'SG'
FullTen.loc[FullTen['Pos'] == 'SG-PG','Pos'] = 'SG'
FullTen.loc[FullTen['Pos'] == 'PF-SF','Pos'] = 'PF'
FullTen.loc[FullTen['Pos'] == 'PF-C','Pos'] = 'PF'

Cluster_Set_Ten = FullTen.join(awards)

Cluster_Set_Ten.index = Cluster_Set_Ten['Player']
Cluster_Set_Ten.drop('Player',axis=1,inplace=True)
Cluster_Set_Ten
cluster_ten_sets_avg = {pos:{'Cluster_Set' : Cluster_Set_Ten[Cluster_Set_Ten['Pos']==pos].drop('Pos',axis=1)} for pos in Cluster_Set_Ten['Pos'].unique()}

#%%
# Recluster the Ten Season Data now

i=1
for pos,item in cluster_ten_sets_avg.items():
    sse = []
    silhouette_coefficients = []
    for k in range(2,11):
        kmeans = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(item['Cluster_Set'])
        sse.append(kmeans.inertia_)
        score = silhouette_score(item['Cluster_Set'], kmeans.labels_)
        silhouette_coefficients.append(score)
    cluster_ten_sets_avg[pos]['SSE'] = sse
    cluster_ten_sets_avg[pos]['Sil_Cof'] = silhouette_coefficients
    knee_range = [i for i in range(2,11)]
    knee = KneeLocator(knee_range,sse, curve='convex',direction='decreasing')
    cluster_ten_sets_avg[pos]['Clusters'] = knee.knee
    kmeans = KMeans(init="random", n_clusters=item['Clusters'], n_init=10, max_iter=300, random_state=42)
    kmeans.fit(item['Cluster_Set'])
    item['Cluster_Set']['Clusters_new'] = kmeans.labels_
    item['Player_Cluster_info'] = item['Cluster_Set'].reset_index()[['Player','Clusters_init','PER','DWS','OWS','TOV%']]
    item['Player_Cluster_info']['Pos'] = pos
    if i==1:
        new_clustering = item['Player_Cluster_info']
    else:
        new_clustering = pd.concat([new_clustering,item['Player_Cluster_info']])
    i+=1
    
full_clustering = new_clustering.merge(initial_clustering,on='Player',suffixes=('_Ten','_Five'))

full_clustering['Current'] = [1 if name in list(current_players_under_10['Player']) else 0 for name in full_clustering['Player']]

full_clustering.to_csv('Prediction_Results.csv')

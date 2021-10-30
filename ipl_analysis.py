
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
# %matplotlib inline
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

matches = pd.read_csv('IPL_matches.csv')

matches.shape

matches.head()

matches.tail()

matches.columns

matches.dtypes

matches.win_by_runs.mean()

#Change Datatypes as per requirement

matches.dl_applied = matches.dl_applied.astype(bool)
matches.date = pd.to_datetime(matches.date)

matches.dtypes

matches.isnull().sum()

matches.drop(columns = ["umpire1","umpire2","umpire3"],axis = 1,inplace = True)

matches.shape

matches.isnull().sum()

#matches.fillna(np.nan)
mode_city = max(matches.city.value_counts())
mode_city

matches.city.value_counts()

matches.city.fillna(mode_city,inplace = True)

matches.city.isnull().sum()

matches[matches.winner.isnull()]

no_result_matches = matches[matches.result == 'no result'] #storing no result matches in a variable
matches = matches[matches.winner.isnull() == False] #drop 4 rows and keep 752 rows

matches[matches.winner.isnull()]

matches.describe()

matches.winner.value_counts()

matches.info()

matches['id'].nunique()

matches['Season'].unique()

matches.Season.nunique()

matches.iloc[matches['win_by_runs'].idxmax()]

matches.iloc[matches['win_by_wickets'].idxmax()]

matches.iloc[matches[matches['win_by_runs'].ge(1)].win_by_runs.idxmin()]

matches[matches[matches['win_by_runs'].ge(1)].win_by_runs.min() == matches['win_by_runs']]['winner']  #to handle the issue of only one team being shown

matches[matches[matches['win_by_wickets'].ge(1)].win_by_wickets.min() == matches['win_by_wickets']][['winner','win_by_wickets']]  #to handle the issue of only one team being shown

sns.countplot(x='Season', data=matches)
plt.show()

data = matches.winner.value_counts()
sns.barplot(y = data.index, x = data, orient='h')

top_players = matches.player_of_match.value_counts()[:10]
fig, ax = plt.subplots()
ax.set_ylim([0,30])
ax.set_ylabel("Count of player of match")
ax.set_xlabel("Player")
ax.set_title("Top player of the match Winners")
#top_players.plot.bar()
sns.barplot(x = top_players.index, y = top_players, orient='v',palette="Blues")
plt.show()

ss = matches['toss_winner'] == matches['winner']

ss.groupby(ss).size()

sns.countplot(ss)

s = round(ss.groupby(ss).size() / ss.count() * 100,2)
s

s.plot.pie(autopct="%.1f%%")

#Using matplotlib
pie, ax = plt.subplots(figsize=[10,6])
labels = s.keys()
plt.pie(x=s, autopct="%.1f%%", labels=labels, pctdistance=0.5)
plt.title("Toss winning helps match winning", fontsize=14);
#plt.savefig("Toss.png")

sns.countplot(x='Season',hue='toss_decision',data=matches,palette=sns.color_palette('bright'))
plt.title("Decision to bat or field across seasons")
plt.show()

#did fielding decision help in winning matches?
match_winner_by_field=matches[matches["toss_decision"]=='field'][['toss_winner', 'winner']]
match_winner_by_field['toss_win_flag']=match_winner_by_field['toss_winner']==match_winner_by_field['winner']
match_winner_by_field.groupby(match_winner_by_field.toss_win_flag).size()

a=match_winner_by_field.groupby(match_winner_by_field.toss_win_flag).size()
labels=a.keys()
plt.pie(x=a,autopct="%.1f%%",labels=labels,pctdistance=0.5)
plt.title("Fielding first helps in match winning",fontsize=14)

#Venue and city analysis
print("Total no. of cities played:",matches['city'].nunique())
print("Total no. of venues played:",matches['venue'].nunique())

ax=matches['venue'].value_counts().sort_values(ascending=True).plot.barh(width=.9,color=sns.color_palette("inferno",40))
ax.set_xlabel("Count")
ax.set_ylabel("Grounds")
plt.title("Venues played (from most to least)")
plt.show()

ipldelivery = pd.read_csv('deliveries.csv')

ipldelivery.head()

ipldelivery.shape

ipldelivery.columns

batsman_grp = ipldelivery.groupby(["match_id", "inning", "batting_team", "batsman"])
batsmen = batsman_grp["batsman_runs"].sum().reset_index()
batsmen.head()

balls_faced = ipldelivery[ipldelivery["wide_runs"] == 0]
balls_faced = balls_faced.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()
balls_faced.columns = ["match_id", "inning", "batsman", "balls_faced"]
balls_faced.head()

batsmen = batsmen.merge(balls_faced, left_on=["match_id", "inning", "batsman"], 
                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen.head()

fours = ipldelivery[ ipldelivery["batsman_runs"] == 4]
sixes = ipldelivery[ ipldelivery["batsman_runs"] == 6]

fours_per_batsman = fours.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()
sixes_per_batsman = sixes.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()

fours_per_batsman.columns = ["match_id", "inning", "batsman", "4s"]
sixes_per_batsman.columns = ["match_id", "inning", "batsman", "6s"]

fours_per_batsman.head()

batsmen = batsmen.merge(fours_per_batsman, left_on=["match_id", "inning", "batsman"], 
                        right_on=["match_id", "inning", "batsman"], how="left")
batsmen = batsmen.merge(sixes_per_batsman, left_on=["match_id", "inning", "batsman"], 
                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen.head()

batsmen['SR'] = np.round(batsmen['batsman_runs'] / batsmen['balls_faced'] * 100, 2)

for col in ["batsman_runs", "4s", "6s", "balls_faced", "SR"]:
    batsmen[col] = batsmen[col].fillna(0)

dismissals = ipldelivery[ pd.notnull(ipldelivery["player_dismissed"])]
dismissals = dismissals[["match_id", "inning", "player_dismissed", "dismissal_kind", "fielder"]]
dismissals.rename(columns={"player_dismissed": "batsman"}, inplace=True)
batsmen = batsmen.merge(dismissals, left_on=["match_id", "inning", "batsman"], 
                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen = matches[['id','Season']].merge(batsmen, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
batsmen.head(10)

## Bowlers grouped by sets of data
# Data is grouped for bowlers to provide greater depth of information. Very important for the regression analysis.

bowler_grp = ipldelivery.groupby(["match_id", "inning", "bowling_team", "bowler", "over"])
bowlers = bowler_grp["total_runs", "wide_runs", "bye_runs", "legbye_runs", "noball_runs"].sum().reset_index()

bowlers["runs"] = bowlers["total_runs"] - (bowlers["bye_runs"] + bowlers["legbye_runs"])
bowlers["extras"] = bowlers["wide_runs"] + bowlers["noball_runs"]

del( bowlers["bye_runs"])
del( bowlers["legbye_runs"])
del( bowlers["total_runs"])

dismissal_kinds_for_bowler = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]
dismissals = ipldelivery[ipldelivery["dismissal_kind"].isin(dismissal_kinds_for_bowler)]
dismissals = dismissals.groupby(["match_id", "inning", "bowling_team", "bowler", "over"])["dismissal_kind"].count().reset_index()
dismissals.rename(columns={"dismissal_kind": "wickets"}, inplace=True)

bowlers = bowlers.merge(dismissals, left_on=["match_id", "inning", "bowling_team", "bowler", "over"], 
                        right_on=["match_id", "inning", "bowling_team", "bowler", "over"], how="left")
bowlers["wickets"] = bowlers["wickets"].fillna(0)

bowlers_over = bowlers.groupby(['match_id', 'inning', 'bowling_team', 'bowler'])['over'].count().reset_index()
bowlers = bowlers.groupby(['match_id', 'inning', 'bowling_team', 'bowler']).sum().reset_index().drop('over', 1)
bowlers = bowlers_over.merge(bowlers, on=["match_id", "inning", "bowling_team", "bowler"], how = 'left')
bowlers['Econ'] = np.round(bowlers['runs'] / bowlers['over'] , 2)
bowlers = matches[['id','Season']].merge(bowlers, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)

bowlers.head(10)

runs_scored=batsmen.groupby(['batsman'])['batsman_runs'].sum()
runs_scored=runs_scored.sort_values(ascending=False)
top10runs = runs_scored.head(10)
top10runs.plot(kind = 'barh')

#boxplot of runs
fig,ax=plt.subplots()
ax.set_xlabel("Runs")
ax.set_title("Winning by runs - Team Performance")
sns.boxplot(y='winner',x='win_by_runs',data=matches[matches['win_by_runs']>0],orient='h')
plt.show()

#Encoding
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors','Delhi Capitals']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW','DC'],inplace=True)

encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DCh':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13, 'DC':14},
          'team2': {'MI':1,'KKR':2,'RCB':3,'DCh':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13, 'DC':14},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DCh':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13, 'DC':14},
          'winner': {'MI':1,'KKR':2,'RCB':3,'DCh':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13, 'DC':14,'Draw':15}}
matches.replace(encode, inplace=True)
matches.head(2)

matches = matches[['team1','team2','city','toss_decision','toss_winner','venue','winner']]
matches.head(2)

df = pd.DataFrame(matches)
df.head()

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  model.fit(data[predictors],data[outcome])
  predictions = model.predict(data[predictors])
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
  
  model.fit(data[predictors],data[outcome])

#from sklearn.ensemble import RandomForestRegressor
outcome_var=['winner']
predictor_var = ['team1','team2','toss_winner']
model = LogisticRegression(max_iter = 1000)
classification_model(model, df,predictor_var,outcome_var)

#Building predictive model
from sklearn.preprocessing import LabelEncoder
var_mod=['city','toss_decision','venue']
le=LabelEncoder()
for i in var_mod:
    df[i]=le.fit_transform(df[i].astype('str'))
df.dtypes

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
outcome_var=['winner']
predictor_var = ['team1','team2','venue','toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var)

dicVal=encode['winner']
print(dicVal['MI']) #key value
print(list(dicVal.keys())[list(dicVal.values()).index(1)])

team1="RCB"
team2="KKR"
toss_winner="RCB"
input=[dicVal[team1],dicVal[team2],'14',dicVal[toss_winner],'2','1']
input=np.array(input).reshape((1,-1))
output=model.predict(input)
print(list(dicVal.keys())[list(dicVal.values()).index(output)])

imp_input=pd.Series(model.feature_importances_,index=predictor_var).sort_values(ascending=False)
print(imp_input)

# Creating a pickle file for the classifier
'''import pickle
filename = 'match_winner_model.pkl'
pickle.dump(model, open(filename, 'wb'))'''


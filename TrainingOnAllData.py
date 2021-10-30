import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split # For splitting the data into train and validation set


matches = pd.read_csv('IPL_matches.csv')

matches.city.fillna(max(matches.city.value_counts()),inplace = True)


matches = matches[matches.winner.isnull() == False] 
matches.replace(['Delhi Daredevils'],['Delhi Capitals'],inplace=True)
matches['team1'].unique()
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Capitals', 'Sunrisers Hyderabad'] 

consistent_venues = ['M Chinnaswamy Stadium','Eden Gardens', 'Feroz Shah Kotla', 'MA Chidambaram Stadium, Chepauk',
       'Punjab Cricket Association Stadium, Mohali', 
       'Wankhede Stadium', 'Sawai Mansingh Stadium',
       'Rajiv Gandhi International Stadium, Uppal']

matches = matches[(matches['team1'].isin(consistent_teams)) & (matches['team2'].isin(consistent_teams)) & (matches['toss_winner'].isin(consistent_teams)) & (matches['winner'].isin(consistent_teams))]
'''print(matches['team1'].unique())
print(matches['team2'].unique())
print(matches['toss_winner'].unique())
print(matches['winner'].unique())'''
matches= matches[(matches['venue'].isin(consistent_venues))]


matches = matches[['team1','team2','toss_decision','toss_winner','venue','winner']]

encoded_df = pd.get_dummies(data=matches, columns=['team1','team2','toss_decision','toss_winner','venue','winner'])

#x=encoded_df.drop(columns='winner', axis=1)
#y=encoded_df['winner'].values
#print(encoded_df.columns)
#X_train, X_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state = 42)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#pd.set_option('display.max_columns',35)
#print(encoded_df.columns)
#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  model.fit(data[predictors],data[outcome])
  predictions = model.predict(data[predictors])
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
  
  model.fit(data[predictors],data[outcome])
  
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
outcome_var=['winner_Chennai Super Kings', 'winner_Delhi Capitals', 'winner_Kings XI Punjab', 'winner_Mumbai Indians', 'winner_Royal Challengers Bangalore', 'winner_Kolkata Knight Riders', 'winner_Rajasthan Royals', 'winner_Sunrisers Hyderabad']
predictor_var = ['team1_Chennai Super Kings', 'team1_Delhi Capitals',
       'team1_Kings XI Punjab', 'team1_Kolkata Knight Riders',
       'team1_Mumbai Indians', 'team1_Rajasthan Royals',
       'team1_Royal Challengers Bangalore', 'team1_Sunrisers Hyderabad',
       'team2_Chennai Super Kings', 'team2_Delhi Capitals',
       'team2_Kings XI Punjab', 'team2_Kolkata Knight Riders',
       'team2_Mumbai Indians', 'team2_Rajasthan Royals',
       'team2_Royal Challengers Bangalore', 'team2_Sunrisers Hyderabad',
       'toss_decision_bat', 'toss_decision_field',
       'toss_winner_Chennai Super Kings', 'toss_winner_Delhi Capitals',
       'toss_winner_Kings XI Punjab', 'toss_winner_Kolkata Knight Riders',
       'toss_winner_Mumbai Indians', 'toss_winner_Rajasthan Royals',
       'toss_winner_Royal Challengers Bangalore',
       'toss_winner_Sunrisers Hyderabad', 'venue_Eden Gardens',
       'venue_Feroz Shah Kotla', 'venue_M Chinnaswamy Stadium',
       'venue_MA Chidambaram Stadium, Chepauk',
       'venue_Punjab Cricket Association Stadium, Mohali',
       'venue_Rajiv Gandhi International Stadium, Uppal',
       'venue_Sawai Mansingh Stadium', 'venue_Wankhede Stadium',
       'winner_Chennai Super Kings', 'winner_Delhi Capitals',
       'winner_Kings XI Punjab', 'winner_Kolkata Knight Riders',
       'winner_Mumbai Indians', 'winner_Rajasthan Royals',
       'winner_Royal Challengers Bangalore', 'winner_Sunrisers Hyderabad']
classification_model(model,encoded_df,predictor_var,outcome_var)

'''model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])    
model.fit(X, y_train,epochs=10, batch_size=10)'''
# Creating a pickle file for the classifier
import pickle
filename = 'match_winner_model.pkl'
pickle.dump(model, open(filename, 'wb'))

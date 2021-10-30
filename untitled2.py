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
matches = matches[(matches['team1'].isin(consistent_teams)) & (matches['team2'].isin(consistent_teams)) & (matches['toss_winner'].isin(consistent_teams)) & (matches['winner'].isin(consistent_teams))]
'''print(matches['team1'].unique())
print(matches['team2'].unique())
print(matches['toss_winner'].unique())
print(matches['winner'].unique())'''

matches = matches[['team1','team2','city','toss_decision','toss_winner','venue','winner']]

encoded_df = pd.get_dummies(data=matches, columns=['team1','team2','city','toss_decision','toss_winner','venue'])

x=encoded_df.drop(columns='winner', axis=1)
y=encoded_df['winner'].values

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
outcome_var=['winner']
predictor_var = ['team1_Chennai Super Kings', 'team1_Delhi Capitals',
       'team1_Kings XI Punjab', 'team1_Kolkata Knight Riders',
       'team1_Mumbai Indians', 'team1_Rajasthan Royals',
       'team1_Royal Challengers Bangalore', 'team1_Sunrisers Hyderabad',
       'team2_Chennai Super Kings', 'team2_Delhi Capitals',
       'team2_Kings XI Punjab', 'team2_Kolkata Knight Riders',
       'team2_Mumbai Indians', 'team2_Rajasthan Royals',
       'team2_Royal Challengers Bangalore', 'team2_Sunrisers Hyderabad',
       'city_101', 'city_Abu Dhabi', 'city_Ahmedabad', 'city_Bangalore',
       'city_Bengaluru', 'city_Bloemfontein', 'city_Cape Town',
       'city_Centurion', 'city_Chandigarh', 'city_Chennai', 'city_Cuttack',
       'city_Delhi', 'city_Dharamsala', 'city_Durban', 'city_East London',
       'city_Hyderabad', 'city_Indore', 'city_Jaipur', 'city_Johannesburg',
       'city_Kimberley', 'city_Kolkata', 'city_Mohali', 'city_Mumbai',
       'city_Port Elizabeth', 'city_Pune', 'city_Raipur', 'city_Ranchi',
       'city_Sharjah', 'city_Visakhapatnam', 'toss_decision_bat',
       'toss_decision_field', 'toss_winner_Chennai Super Kings',
       'toss_winner_Delhi Capitals', 'toss_winner_Kings XI Punjab',
       'toss_winner_Kolkata Knight Riders', 'toss_winner_Mumbai Indians',
       'toss_winner_Rajasthan Royals',
       'toss_winner_Royal Challengers Bangalore',
       'toss_winner_Sunrisers Hyderabad', 'venue_ACA-VDCA Stadium',
       'venue_Barabati Stadium', 'venue_Brabourne Stadium',
       'venue_Buffalo Park', 'venue_De Beers Diamond Oval',
       'venue_Dr DY Patil Sports Academy',
       'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'venue_Dubai International Cricket Stadium', 'venue_Eden Gardens',
       'venue_Feroz Shah Kotla', 'venue_Feroz Shah Kotla Ground',
       'venue_Himachal Pradesh Cricket Association Stadium',
       'venue_Holkar Cricket Stadium', 'venue_IS Bindra Stadium',
       'venue_JSCA International Stadium Complex', 'venue_Kingsmead',
       'venue_M Chinnaswamy Stadium', 'venue_M. A. Chidambaram Stadium',
       'venue_M. Chinnaswamy Stadium', 'venue_MA Chidambaram Stadium, Chepauk',
       'venue_Maharashtra Cricket Association Stadium',
       'venue_New Wanderers Stadium', 'venue_Newlands',
       'venue_OUTsurance Oval',
       'venue_Punjab Cricket Association IS Bindra Stadium, Mohali',
       'venue_Punjab Cricket Association Stadium, Mohali',
       'venue_Rajiv Gandhi International Stadium, Uppal',
       'venue_Rajiv Gandhi Intl. Cricket Stadium',
       'venue_Sardar Patel Stadium, Motera', 'venue_Sawai Mansingh Stadium',
       'venue_Shaheed Veer Narayan Singh International Stadium',
       'venue_Sharjah Cricket Stadium', 'venue_Sheikh Zayed Stadium',
       'venue_St George\'s Park', 'venue_Subrata Roy Sahara Stadium',
       'venue_SuperSport Park', 'venue_Wankhede Stadium']
classification_model(model,encoded_df,predictor_var,outcome_var)

'''model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])    
model.fit(X, y_train,epochs=10, batch_size=10)'''
# Creating a pickle file for the classifier
#filename = 'score_predictor_model.pkl'
#pickle.dump(lasso_regressor, open(filename, 'wb'))

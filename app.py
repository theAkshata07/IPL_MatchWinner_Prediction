# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'match_winner_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('dropdown.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
        batting_team = request.form['Team_1']
        if batting_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Capitals':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif batting_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif batting_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif batting_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        bowling_team = request.form['Team_2']
        if bowling_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif bowling_team == 'Delhi Capitals':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif bowling_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif bowling_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif bowling_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif bowling_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
        Venue = request.form['venue']
        if Venue == 'M Chinnaswamy Stadium':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif Venue == 'Eden Gardens':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif Venue == 'Feroz Shah Kotla':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif Venue == 'MA Chidambaram Stadium, Chepauk':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif Venue == 'Punjab Cricket Association Stadium, Mohali':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif Venue == 'Wankhede Stadium':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif Venue == 'Sawai Mansingh Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif Venue == 'Rajiv Gandhi International Stadium, Uppal':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
        
        Toss_decision=request.form['Toss_decision']
        if Toss_decision=='Bat':
             temp_array = temp_array + [1,0]
        elif Toss_decision=='field':
            temp_array = temp_array + [0,1]
            
        Toss_won_by=reuest.form['Toss_won_by']
        if Toss_won_by== 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif Toss_won_by== 'Delhi Capitals':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif Toss_won_by == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif Toss_won_by== 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif Toss_won_by== 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif Toss_won_by== 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif Toss_won_by== 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif Toss_won_by == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
        
       # data = np.array([temp_array])
        #my_prediction = int(classifier.predict(temp_array)[0])
        # prediction function 
def ValuePredictor(temp_array): 
    to_predict = np.array(temp_array)
    filename = 'match_winner_model.pkl'
    classifier = pickle.load(open(filename, 'rb')) 
    result =classifier.predict(to_predict) 
    return result[0] 
  
@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        temp_array = request.form.to_dict() 
        temp_array = list(temp_array.values()) 
        temp_array = list(map(int, temp_array)) 
        result = ValuePredictor(temp_array)
        #print(result)
        #result=int(result)
        #result =(classifier.predict(temp_array)[0])         
        if result==[1,0,0,0,0,0,0,0]: 
            prediction='Chennai Super Kings'
        elif result==[0,1,0,0,0,0,0,0]:
            prediction ='Delhi Capitals'
        elif result==[0,0,1,0,0,0,0,0]:
            prediction='Kings XI Punjab'
        elif result==[0,0,0,1,0,0,0,0]:
            prediction='Kolkata Knight Riders'
        elif result==[0,0,0,0,1,0,0,0]:
            prediction='Mumbai Indians'
        elif result==[0,0,0,0,0,1,0,0]:
            prediction='Rajasthan Royals'
        elif result==[0,0,0,0,0,0,1,0]:
            prediction='Royal Challengers Bangalore'   
        elif result ==[0,0,0,0,0,0,0,1]:
            prediction='Sunrisers Hyderabad'
            
        if result==1: 
            prediction='Chennai Super Kings'
        elif result==2:
            prediction ='Delhi Capitals'
        elif result==3:
            prediction='Kings XI Punjab'
        elif result==4:
            prediction='Kolkata Knight Riders'
        elif result==5:
            prediction='Mumbai Indians'
        elif result==6:
            prediction='Rajasthan Royals'
        elif result==7:
            prediction='Royal Challengers Bangalore'   
        elif result ==8:
            prediction='Sunrisers Hyderabad' 
            
        return render_template('result.html',prediction=prediction )



if __name__ == '__main__':
	app.run(debug=False)
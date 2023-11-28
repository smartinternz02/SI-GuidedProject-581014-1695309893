import pandas as pd
import numpy as np
file_path = 'C:\Users\varan\Downloads\t20i_info.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path,encoding='utf-8')
df.head()
df[df['city'].isnull()]['venue'].value_counts()
cities = np.where(df['city'].isnull(),df['venue'].str.split().apply(lambda x : x[0]), df['city'])
df['city'] = cities
df.isnull().sum()
df['city'].value_counts()
eligible_cities = df['city'].value_counts()[df['city'].value_counts()>600].index.tolist()
df = df[df['city'].isin(eligible_cities)]
df['current_score'] = df.groupby('match_id').cumsum()['runs']
df['over'] = df['ball'].apply(lambda x : str(x).split(".")[0])
df['ball_no'] = df['ball'].apply(lambda x : str(x).split(".")[1])
df['balls_bowled'] = (df['over'].astype('int')*6 + df['ball_no'].astype('int'))
df
df['balls_left'] = 120-df['balls_bowled']
df['balls_left'] = df['balls_left'].apply(lambda x: 0 if x< 0 else x)
df.head
df['player_dimissed'] = df['player_dismissed'].apply(lambda x: 1 if x !=0 else 0)
df.sample(5)
df['player_dismissed'] = df['player_dismissed'].astype('int')

df['player_dismissed'] = df.groupby('match_id').cumsum()['player_dismissed']

df['wickets_left'] = 10-df['player_dismissed']

df.sample
df['crr'] = (df['current_score']*6) / df['balls_bowled']

df.head()
groups = df.groupby('match_id')

match_ids = df['match_id'].unique()
last_five = []
for id in match_ids:
  last_five.extend(groups.get_group(id).rolling(window=30).sum()['runs'].values.tolist())
  df['last_five'] = last_five
df.sample(5)
final_df = df.groupby('match_id').sum()['runs'].reset_index().merge(df,on='match_id')
final_df.head()
final_df = final_df[['batting_team','bowling_team','city','current_score','balls_left','wickets_left','crr','last_five','runs_x']]
final_df.dropna(inplace=True)
final_df.isnull().sum()
final_df = final_df.sample(final_df.shape[0])
final_df = final_df.sample(final_df.shape[0])
final_df
X = final_df.drop(columns=['runs_x'])
y = final_df['runs_x']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
X_train
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

! pip install xgboost
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')
pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',StandardScaler()),
    ('step3',XGBRegressor(n_estimators=1000,learning_rate=0.2,max_depth=12,random_state=1))
])
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
import pickle

# Save the model to the root directory
with open('/pipe.pkl', 'wb') as file:
    pickle.dump(pipe, file)
    import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Define teams and cities
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']
cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

with open('/pipe.pkl', 'rb') as file:
    pipe = pickle.load(file)

# Streamlit app
st.title('Cricket Score Predictor')

# Layout columns
col1, col2 = st.columns(2)

# Dropdown for selecting batting team
with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))

# Dropdown for selecting bowling team
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

# Dropdown for selecting city
city = st.selectbox('Select city', sorted(cities))

# Layout columns for additional input fields
col3, col4, col5 = st.columns(3)

# Input field for current score
with col3:
    current_score = st.number_input('Current Score')

# Input field for overs done
with col4:
    overs = st.number_input('Overs done (works for over > 5)')

# Input field for wickets out
with col5:
    wickets = st.number_input('Wickets out')

# Input field for runs scored in last 5 overs
last_five = st.number_input('Runs scored in last 5 overs')

# Predict button
if st.button('Predict Score'):
    balls_left = (120 - (overs * 6))
    wickets_left = (10 - wickets)
    crr = current_score / overs

    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'current_score': [current_score],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'last_five': [last_five]
    })

    # Make prediction
    result = pipe.predict(input_df)

    # Display predicted score
    st.header("Predicted Score: " + str(int(result[0])))

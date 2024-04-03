#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


train_data = pd.read_excel(r"C:\Users\Rishabh\Downloads\drive-download-20240226T070455Z-001\Data_Train.xlsx")


# In[7]:


train_data


# In[8]:


train_data.head()


# In[9]:


train_data.tail()


# In[10]:


train_data.info()


# In[11]:


#checking nan values

train_data.isnull().sum()


# In[12]:


train_data['Total_Stops'].isnull()


# In[13]:


train_data[train_data['Total_Stops'].isnull()]


# In[14]:


train_data.dropna(inplace=True)


# In[15]:


train_data.isnull().sum()


# In[16]:


train_data.dtypes


# In[17]:


train_data.info(memory_usage='deep')


# In[18]:


df = train_data.copy()


# In[19]:


df


# In[20]:


df.columns


# In[21]:


def change_into_datetime(col):
    df[col] = pd.to_datetime(df[col])


# In[22]:


df


# In[23]:


for feature in ['Arrival_Time','Date_of_Journey','Dep_Time']:
    change_into_datetime(feature)


# In[24]:


df.dtypes


# In[25]:


df['Journey_day'] = df['Date_of_Journey'].dt.day


# In[26]:


df['Journey_month'] = df['Date_of_Journey'].dt.month


# In[27]:


df['Journey_year'] = df['Date_of_Journey'].dt.year


# In[28]:


df.head()


# In[29]:


#making 2 new columns
def extract_hour_min(dataframe , col):
    df[col+"_hour"] = df[col].dt.hour
    df[col+"_minute"] = df[col].dt.minute
    return df.head(3)


# In[30]:


df.columns


# In[31]:


df


# In[32]:


extract_hour_min(df,'Dep_Time')


# In[33]:


extract_hour_min(df,'Arrival_Time')


# In[34]:


## we have extracted derived attributes from ['Arrival_Time' , "Dep_Time"] , so lets drop both these features ..
cols_to_drop = ['Arrival_Time' , "Dep_Time"]

df.drop(cols_to_drop , axis=1 , inplace=True )


# In[35]:


df.head(3)


# In[36]:


df.shape


# In[37]:


df.columns


# In[38]:


#function to create flight information of Dep_Time_hour into propper time (morning,noon,evening,night)


def flight_dep_time(x):
    
    if (x>4) and (x<=8):
        return "Early Morning"
    
    elif(x>8) and (x<=12):
        return "Morning"
        
    elif(x>12) and (x<=16):
        return "Afternoon"
    
    elif(x>16) and (x<=20):
        return "Evening"
    
    elif(x>20) and (x<=24):
        return "Night"
    
    else:
        return "Late Night"


# In[39]:


df['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind="bar" , color="g")


# In[40]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install chart_studio')


# In[41]:


get_ipython().system('pip install cufflinks')


# In[42]:


import plotly 
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import plot, iplot, init_notebook_mode, download_plotlyjs
init_notebook_mode(connected = True)
cf.go_offline()


# In[43]:


df['Dep_Time_hour'].apply(flight_dep_time).value_counts().iplot(kind = "bar")


# In[44]:


df


# In[45]:


def preprocess_duration(x):
    if 'h' not in x:
        x = '0h' + ' ' + x
    
    elif 'm' not in x:
        x = x + ' ' + '0m'
    
    return x


# In[46]:


df['Duration'] = df['Duration'].apply(preprocess_duration)


# In[47]:


df['Duration']


# In[48]:


df['Duration'][0]


# In[49]:


'2h 50m'.split()


# In[50]:


'2h 50m'.split()[0]


# In[51]:


'2h 50m'.split()[0][0:-1]


# In[52]:


'2h 50m'.split()[1][0:-1]


# In[53]:


int('2h 50m'.split()[1][0:-1])


# In[54]:


int('2h 50m'.split()[0][0:-1])


# In[55]:


df['Duration_Mins'] = df['Duration'].apply(lambda x : int(x.split()[1][0:-1]))
df['Duration_Hours'] = df['Duration'].apply(lambda x : int(x.split()[0][0:-1]))


# In[56]:


df.head(2)


# In[57]:


pd.to_timedelta(df['Duration']).dt.components.hours


# In[58]:


df.head(2)


# In[59]:


df['Duration_Total_Mins'] = df['Duration'].str.replace('h' , "*60").str.replace(' ' , "+").str.replace('m' , "*1").apply(eval)


# In[60]:


df['Duration_Total_Mins']


# In[61]:


df


# In[62]:


df.columns


# In[63]:


sns.scatterplot(data=df,y='Price',x="Duration_Total_Mins",hue='Total_Stops')


# In[64]:


df['Airline'] == 'Jet Airways'


# In[65]:


df[df['Airline'] == 'Jet Airways'].groupby('Route').size().sort_values(ascending=False)


# In[66]:


df.columns


# In[67]:


sns.boxplot(x='Airline',y='Price',data=df.sort_values('Price', ascending=False))
plt.xticks(rotation='vertical')
plt.show()


# In[68]:


cat_col = [col for col in df.columns if df[col].dtype == 'object']


# In[69]:


num_col = [col for col in df.columns if df[col].dtype != 'object']


# In[70]:


cat_col


# In[71]:


num_col


# In[72]:


df['Source'].apply(lambda x : 1 if x == 'Banglore' else 0)


# In[73]:


for sub_category in df['Source'].unique() :
    df['Source_'+sub_category] = df['Source'].apply(lambda x : 1 if x == sub_category else 0)


# In[74]:


df.head(2)


# In[ ]:





# In[75]:


df.columns


# In[76]:


df['Airline'].nunique()


# In[77]:


df.groupby(['Airline'])['Price'].mean().sort_values()


# In[78]:


airlines = df.groupby(['Airline'])['Price'].mean().sort_values().index


# In[79]:


airlines


# In[80]:


airlines_dict = {key:index for index , key in enumerate(airlines, 0)}


# In[81]:


airlines_dict


# In[82]:


df['Airline'] = df['Airline'].map(airlines_dict)


# In[83]:


df['Airline']


# In[84]:


df['Destination'].unique()


# In[85]:


df['Destination'].replace('New Delhi','Delhi',inplace=True)


# In[86]:


df['Destination'].unique()


# In[87]:


dest = df.groupby(['Destination'])['Price'].mean().sort_values().index


# In[88]:


dest_dict = {key:index for index, key in enumerate(dest, 0)}


# In[89]:


dest_dict


# In[90]:


df['Destination'] = df['Destination'].map(dest_dict)


# In[91]:


df['Destination']


# In[92]:


df.head(2)


# ## MANUAL ENCODING 

# In[93]:


df.head(3)


# In[94]:


df.columns


# In[95]:


df['Total_Stops'].unique()


# In[96]:


stop = {'non-stop': 0, '2 stops': 2, '1 stop': 1, '3 stops': 3, '4 stops': 4}


# In[97]:


df['Total_Stops'] = df['Total_Stops'].map(stop)


# In[98]:


df['Total_Stops'].unique()


# ## Remove Unnecessary Columns

# In[99]:


df.columns


# In[100]:


df.head(2)


# In[101]:


#we can see that 'Additional_Info' may have no info, lets check that


# In[102]:


df['Additional_Info'].value_counts()


# In[103]:


#No info entry is almost 80% in the 'Additional_Info' . lets find other collumns also which are of no use in ML


# In[104]:


df.head(3)


# In[105]:


#we can remove source, route, date of journey, duration, date of journey, journey year, total duration min & additional info 


# In[106]:


df.columns


# In[107]:


cols_to_drop = ['Route','Duration','Date_of_Journey','Source','Additional_Info','Journey_year','Duration_Total_Mins']


# In[108]:


df.drop(cols_to_drop, inplace = True, axis = 1)


# In[109]:


df.head()


# In[110]:


df.columns.nunique()


# ### Let's perform outlier detection

# In[111]:


def plot(df, col):
    fig, (ax1,ax2,ax3) = plt.subplots(3,1)
    
    sns.distplot(df[col] , ax = ax1)
    sns.boxplot(x=df[col] , ax = ax2)
    sns.histplot(df[col] , ax = ax3,kde=False)


# In[112]:


plot(df,'Price')


# In[113]:


Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

maximum = Q3 + 1.5 * IQR
minimum = Q1 - 1.5 * IQR


# In[114]:


maximum


# In[115]:


minimum


# In[116]:


print([price for price in df['Price'] if price>maximum or price<minimum])


# In[117]:


len([price for price in df['Price'] if price>maximum or price<minimum])


# In[118]:


df['Price'] = np.where(df['Price']>=35000, df['Price'].median(), df['Price'])


# In[119]:


plot(df, 'Price')


# In[120]:


df.head(2)


# In[121]:


X = df.drop(['Price'], axis = 1)


# In[122]:


y = df['Price']


# In[123]:


from sklearn.feature_selection import mutual_info_regression


# In[124]:


imp = mutual_info_regression(X , y)


# In[125]:


imp


# In[126]:


imp_df = pd.DataFrame(imp, index = X.columns)


# In[127]:


imp_df.columns = ['Importance']


# In[128]:


imp_df


# In[129]:


imp_df.sort_values(by='Importance', ascending=False)


# ## Building ML Model

# In[130]:


from sklearn.model_selection import train_test_split


# In[131]:


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.25, random_state=42)


# In[132]:


from sklearn.ensemble import RandomForestRegressor


# In[133]:


ml_model = RandomForestRegressor()


# In[134]:


ml_model.fit(X_train , y_train)


# In[135]:


y_pred = ml_model.predict(X_test)


# In[136]:


y_pred


# In[137]:


from sklearn import metrics


# In[138]:


metrics.r2_score(y_test , y_pred)


# ## save the model

# In[139]:


pip install pickle4


# In[140]:


import pickle


# In[141]:


file = open('D:\Airlines/random.pkl', 'wb')


# In[142]:


pickle.dump(ml_model , file)


# In[143]:


model = open('D:\Airlines/random.pkl', 'rb')


# In[144]:


forest = pickle.load(model)


# In[145]:


forest


# In[146]:


y_pred2 = forest.predict(X_test)


# In[149]:


metrics.r2_score(y_test,y_pred2)


# ### making our own metric

# In[150]:


def mape(y_true, y_pred):
    y_true , y_pred = np.array(y_true) , np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[151]:


mape(y_test,y_pred)


# ### automate ml pipeline

# In[153]:


from sklearn import metrics 


# In[156]:


def predict(ml_model):
    model = ml_model.fit(X_train , y_train)
    print('Training Score : {}'.format(model.score(X_train , y_train)))
    y_prediction = ml_model.predict(X_test)
    print('Predictions are : {}'.format(y_prediction))
    print('\n')
    r2_score = metrics.r2_score(y_test , y_prediction)
    print('r2_score : {}'.format(r2_score))
    print('MSE : {}'.format(metrics.mean_squared_error(y_test , y_prediction)))
    print('MAE : {}'.format(metrics.mean_absolute_error(y_test , y_prediction)))
    print('RMSE : {}'.format(np.sqrt(metrics.mean_squared_error(y_test , y_prediction))))
    print('MAPE : {}'.format(mape(y_test , y_prediction)))
    sns.distplot(y_test - y_prediction)


# In[157]:


predict(RandomForestRegressor())


# In[158]:


from sklearn.tree import DecisionTreeRegressor


# In[161]:


predict(DecisionTreeRegressor())


# ## Hypertune the model

# In[164]:


from sklearn.model_selection import RandomizedSearchCV


# In[165]:


#estimator initialized
reg_rf = RandomForestRegressor()


# In[167]:


n_estimators = [int(x)for x in np.linspace(start = 100, stop = 1200, num = 6)]


# In[168]:


max_features = ["auto","sqrt"]


# In[169]:


max_depth = [int(x)for x in np.linspace(start = 5, stop = 30, num = 4)]


# In[171]:


max_depth


# In[173]:


min_samples_split = [5,10,15,100]


# ### creating random grid/ hyper-parameter space

# In[174]:


random_grid = {
    'n_estimators' : n_estimators ,
    'max_features' : max_features ,
    'max_depth' : max_depth ,
    'min_samples_split' : min_samples_split
}


# In[175]:


random_grid


# ### Initializing RandomizedSearchcv

# In[183]:


rf_random = RandomizedSearchCV(estimator=reg_rf, param_distributions=random_grid, cv=3, n_jobs=-1, verbose=2)


# In[184]:


rf_random.fit(X_train,y_train)


# In[185]:


rf_random.best_params_


# In[186]:


rf_random.best_estimator_


# In[187]:


rf_random.best_score_


# In[ ]:





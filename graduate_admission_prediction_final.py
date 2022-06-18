#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


admission_df = pd.read_csv('Admission_Predict.csv')


# In[3]:


admission_df.shape


# In[4]:


admission_df.head()


# In[5]:


# Deleting serial no. column because it's irrelevant
admission_df.drop('Serial No.', axis=1, inplace=True)
admission_df.head()


# In[6]:


# Data Analysis
# checking the null values
admission_df.isnull().sum()


# In[7]:


# Observation - Our data contains no null value


# In[8]:


# Check the dataframe information
admission_df.info()


# In[9]:


# Statistical summary of the dataframe
admission_df.describe()


# In[10]:


# Grouping by University ranking 
df_university = admission_df.groupby(by = 'University Rating').mean()
df_university


# In[11]:


# Data Visualization
admission_df.hist(bins = 30, figsize = (20, 20), color='r')


# In[12]:


sns.pairplot(admission_df)


# In[13]:


corr_matrix = admission_df.corr()
plt.figure(figsize=(12,12,))
sns.heatmap(corr_matrix, annot=True)
plt.show()


# In[ ]:





# In[14]:


admission_df.columns


# In[15]:


X = admission_df.drop(columns=['Chance of Admit '])
X


# In[16]:


y = admission_df['Chance of Admit ']
y


# In[17]:


X = np.array(X)
y = np.array(y)


# In[18]:


y = y.reshape(-1,1)


# In[19]:


y.shape


# In[20]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score


# In[23]:


linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)


# In[24]:


y_pred=linear_regression_model.predict(X_test)


# In[25]:


accuracy_LinearRegression = linear_regression_model.score(X_test, y_test)
accuracy_LinearRegression


# In[26]:


from sklearn.tree import DecisionTreeRegressor
decisionTree_model = DecisionTreeRegressor()
decisionTree_model.fit(X_train, y_train)


# In[27]:


accuracy_decisionTree = decisionTree_model.score(X_test, y_test)
accuracy_decisionTree


# In[28]:


from sklearn.ensemble import RandomForestRegressor
randomForest_model = RandomForestRegressor(n_estimators=100, max_depth=10)
randomForest_model.fit(X_train, y_train)


# In[29]:


accuracy_randomforest = randomForest_model.score(X_test, y_test)
accuracy_randomforest


# In[30]:


# Best Model Evaluation
y_pred = linear_regression_model.predict(X_test)
plt.plot(y_test, y_pred, '^', color='r')


# In[31]:


y_predict_orig = scaler_y.inverse_transform(y_pred)
y_test_orig = scaler_y.inverse_transform(y_test)


# In[32]:


k = X_test.shape[1]
n = len(X_test)
n


# In[33]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 


# In[ ]:





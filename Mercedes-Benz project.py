#!/usr/bin/env python
# coding: utf-8

# # Mercedes-Benz Greener Manufacturing
# Reduce the time a Mercedes-Benz spends on the test bench.
# 
# Problem Statement Scenario:
# Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include the passenger safety cell with a crumple zone, the airbag, and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers. Mercedes-Benz is the leader in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.
# 
# To ensure the safety and reliability of every unique car configuration before they hit the road, the company’s engineers have developed a robust testing system. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Mercedes-Benz’s production lines. However, optimizing the speed of their testing system for many possible feature combinations is complex and time-consuming without a powerful algorithmic approach.
# 
# You are required to reduce the time that cars spend on the test bench. Others will work with a dataset representing different permutations of features in a Mercedes-Benz car to predict the time it takes to pass testing. Optimal algorithms will contribute to faster testing, resulting in lower carbon dioxide emissions without reducing Mercedes-Benz’s standards.
# 
# ### Following actions should be performed:
# 
# 1.If for any column(s), the variance is equal to zero, then you need to remove those variable(s).
# 
# 2.Check for null and unique values for test and train sets.
# 
# 3.Apply label encoder.
# 
# 4.Perform dimensionality reduction.
# 
# 5.Predict your test_df values using XGBoost. 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


mb_train=pd.read_csv('train.csv')
mb_test=pd.read_csv('test.csv')


# In[3]:


mb_train.head()


# In[4]:


mb_train.shape


# In[5]:


mb_test.head()


# In[6]:


mb_test.shape


# In[7]:


mb_train.drop('ID',axis=1,inplace=True)


# In[8]:


mb_train.head(2)


# In[9]:


mb_test.drop('ID',axis=1,inplace=True)
mb_test.head()


# We can clearly see the sparsity in training and testing data.It means that there are more binary features as compared to categorical features.

# In[10]:


dtypes_df=mb_train.dtypes.reset_index()
dtypes_df.columns=['feature type','dtypes']
dtypes_df.groupby('dtypes').agg('count').reset_index()


# In[11]:


dtypes_test=mb_test.dtypes.reset_index()
dtypes_test.columns=['feature type','dtypes']
dtypes_test.groupby('dtypes').agg('count').reset_index()


# After analyzing the data type we can say,there are 368 binary features, 8 feature which have data type object which is *categorical feature* and one remaining feature is our target variable which is **y**.

# ### Analyzing categorical features-

# In[12]:


fig,ax=plt.subplots(2,1,figsize=(15,10))
sns.boxplot(x=mb_train['X0'],y=mb_train['y'],ax=ax[0])
sns.boxplot(x=mb_train['X1'],y=mb_train['y'],ax=ax[1])


# In[13]:


fig,ax=plt.subplots(2,1,figsize=(15,10))
sns.boxplot(x=mb_train['X2'],y=mb_train['y'],ax=ax[0])
sns.boxplot(x=mb_train['X3'],y=mb_train['y'],ax=ax[1])


# In[14]:


fig,ax=plt.subplots(2,1,figsize=(15,10))
sns.boxplot(x=mb_train['X4'],y=mb_train['y'],ax=ax[0])
sns.boxplot(x=mb_train['X5'],y=mb_train['y'],ax=ax[1])


# In[15]:


fig,ax=plt.subplots(2,1,figsize=(15,10))
sns.boxplot(x=mb_train['X6'],y=mb_train['y'],ax=ax[0])
sns.boxplot(x=mb_train['X8'],y=mb_train['y'],ax=ax[1])


# ##### **We obsereved that X4 has low variance so we can remove that feature from our dataset**

# In[16]:


mb_train1=mb_train.drop('X4',axis=1,inplace=True)
mb_test1=mb_test.drop('X4',axis=1,inplace=True)


# In[17]:


mb_train.head()


# ## Analysis of Binary Features-

# In[18]:


mb_train_num=mb_train.select_dtypes(include='int64')


# In[19]:


mb_train_num


# In[20]:


# removing features with 0 variance
temp = []
for i in mb_train_num.columns:
    if mb_train_num[i].var()==0:
        temp.append(i)
        
print(len(temp))
print(temp)


# From this, we observed that there are 12 features that have constant value across all data points. So, as per my assumption, these features will not contribute to the modeling.

# In[21]:


mb_train.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330' ,'X347'],axis=1,inplace=True)


# In[22]:


mb_train


# In[23]:


mb_test.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330' ,'X347'],axis=1,inplace=True)


# In[24]:


mb_test


# ## Analysis of Target variable(y):-

# In[25]:


sns.boxplot(y='y',data=mb_train)


# In[26]:


mb_train.reset_index().plot(kind='scatter', x='index', y='y')
plt.show()


# Here, we can see there are many duplicates values and thershold of target variable lies between 150 and above it can be considered as outliers.

# ### Check for duplicates features

# In[27]:


mb_train1 = mb_train.drop_duplicates(keep=False)


# In[28]:


mb_train1


# In[29]:


mb_test1=mb_test.drop_duplicates(keep=False)
mb_test1


# ### Check for null and unique values

# In[30]:


mb_train1.isnull().sum().any()


# In[31]:


mb_test1.isnull().sum().any()


# ## Apply Label Encoder

# In[32]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[33]:


mb_train_feature=mb_train1.drop(columns={'y'})
mb_train_target=mb_train1.y
print(mb_train_feature.shape)
print(mb_train_target.shape)


# In[34]:


mb_train_feature.describe(include='object')


# In[35]:


mb_train_feature['X0']=le.fit_transform(mb_train_feature.X0)
mb_train_feature['X1']=le.fit_transform(mb_train_feature.X1)
mb_train_feature['X2']=le.fit_transform(mb_train_feature.X2)
mb_train_feature['X3']=le.fit_transform(mb_train_feature.X3)
mb_train_feature['X5']=le.fit_transform(mb_train_feature.X5)
mb_train_feature['X6']=le.fit_transform(mb_train_feature.X6)
mb_train_feature['X8']=le.fit_transform(mb_train_feature.X8)


# In[36]:


mb_train_feature


# In[46]:


mb_test


# In[47]:


mb_test.describe(include='object')


# In[48]:


mb_test['X0']=le.fit_transform(mb_test.X0)
mb_test['X1']=le.fit_transform(mb_test.X1)
mb_test['X2']=le.fit_transform(mb_test.X2)
mb_test['X3']=le.fit_transform(mb_test.X3)
mb_test['X5']=le.fit_transform(mb_test.X5)
mb_test['X6']=le.fit_transform(mb_test.X6)
mb_test['X8']=le.fit_transform(mb_test.X8)


# In[49]:


mb_test


# ## Perform Dimensionality Reduction

# In[58]:


from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)


# In[59]:


pca.fit(mb_train_feature,mb_train_target)


# In[60]:


mb_train_feature1=pca.fit_transform(mb_train_feature)
print(mb_train_feature1.shape)


# In[61]:


pca.fit(mb_test)


# In[62]:


mb_test_trans=pca.fit_transform(mb_test)
mb_test_trans.shape


# ## Predict values using XGBoost

# In[63]:


get_ipython().system('pip install xgboost')


# In[64]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt


# In[65]:


train_x,test_x,train_y,test_y=train_test_split(mb_train_feature1,mb_train_target,test_size=.3,random_state=7)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# In[66]:


xgb_reg=xgb.XGBRegressor(objective='reg:linear',colsample_bytree=0.3,learning_rate=0.4,max_depth=10,alpha=6,n_estimators=20)
model=xgb_reg.fit(train_x,train_y)
print('RMSE=',sqrt(mean_squared_error(model.predict(test_x),test_y)))


# In[67]:


pred_test_y=model.predict(test_x)
plt.figure(figsize=(10,5))
sns.distplot(test_y[test_y<150],color="skyblue",label="Actual_value")
sns.distplot(pred_test_y[pred_test_y<150],color="yellow",label="predict_value")
plt.legend()
plt.tight_layout()


# In[68]:


test_pred=model.predict(mb_test_trans)
test_pred


# In[ ]:





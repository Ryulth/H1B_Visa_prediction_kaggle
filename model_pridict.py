
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
#from sklearn import tree 

#from sklearn.neural_network import MLPClassifier 
#from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics
from sklearn.externals import joblib


# In[5]:


file_path='./datas/'
file_name=file_path+"h1b_test_preprocessing.csv"
df_test_no_ylabel=pd.read_csv(file_name,keep_default_na=False)
temp=pd.read_csv('./datas/h1b_test_no_ylabel.csv')


# In[11]:


file_name=file_path+"MLP_Model_h1b"
mlp_model=joblib.load(file_name)


# In[13]:


df_test_dummy=pd.get_dummies(df_test_no_ylabel,columns=['YEAR', 'SOC_CODE', 'STATE','WAGE_CATEGORY'])
print("DATA DUMMY DONE")


# In[33]:


y_pred_mlp=mlp_model.predict(X)


# In[19]:


df_test_dummy.columns


# In[20]:


file_path='./datas/'
file_name=file_path+"h1b_train_preprocessing.csv"
temp=pd.read_csv(file_name,keep_default_na=False)


# In[21]:


temp=pd.get_dummies(temp,columns=['YEAR', 'SOC_CODE', 'STATE','WAGE_CATEGORY'])
print("DATA DUMMY DONE")


# In[23]:


temp.columns


# In[25]:


X = temp.drop('CASE_STATUS', axis=1)


# In[27]:


X.columns


# In[28]:


X.columnsdf_test_dummy.columns


# In[29]:


s = set(temp)
temp2 = [x for x in X.columns if x not in df_test_dummy ]


# In[30]:


temp2


# In[31]:


list(set(X.columns) - set(df_test_dummy)) 


# In[32]:


list(set(df_test_dummy) - set(X.columns)) 


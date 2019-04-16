
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import tree 

from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB 



from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics


# In[32]:


file_path='./datas/'
file_name=file_path+"h1b_train_final_data_ry.csv"
df=pd.read_csv(file_name,keep_default_na=False)
from sklearn.externals import joblib

#df.columns


# In[3]:


df_denied = df[df['CASE_STATUS'] == 1] #1 = DiNED


# In[4]:


df_certified = df[df['CASE_STATUS'] == 0] #1 = CERTIFIED


# In[33]:


df["CASE_STATUS"].value_counts()


# In[5]:


Input_Certified, Input_Certified_extra, y_certified, y_certified_extra = train_test_split(df[df.CASE_STATUS == 0],
                                                                                          df_certified.CASE_STATUS, train_size= 0.4, random_state=1)


# In[6]:


df_train=Input_Certified.append(df_denied)


# In[7]:


df_train.CASE_STATUS.value_counts()


# In[8]:


df_train=pd.get_dummies(df_train,columns=['YEAR', 'SOC_CODE', 'STATE'])


# In[20]:


#df_train


# In[11]:
X_train = df_train.drop('CASE_STATUS', axis=1)
y_train = df_train.CASE_STATUS
X_train=X_train.drop('ID',axis=1)
X_train=X_train.drop("PREVAILING_WAGE",axis=1)
'''
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
#X_train.columns
'''

# In[12]:


df_train['CASE_STATUS'].value_counts()


# In[13]:


dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train) 


# In[14]:



# In[16]:


mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20), max_iter=1000)
mlp.fit(X_train, y_train)


# In[23]:




# In[18]:


gaus_clf = GaussianNB()
gaus_clf.fit(X_train, y_train)


# In[19]:




# In[21]:



type="train2_"
save_dtree=file_path+type+'DTREE_Model_h1b'
save_mlp=file_path+type+'MLP_Model_h1b'
save_gaus=file_path+type+'GAUS_Model_h1b'


# In[22]:


joblib.dump(dtree, save_dtree)
print("done")
joblib.dump(mlp, save_mlp)
print("done")
joblib.dump(gaus_clf, save_gaus)
print("done")
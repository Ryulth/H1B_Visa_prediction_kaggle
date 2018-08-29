
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import tree 
#from sklearn import svm
#import sklearn.linear_model as linear_model
#from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB 

#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics
#from sklearn import preprocessing
import pickle

# In[2]:


file_path='./datas/'
file_name=file_path+"h1b_train_preprocessing.csv"
df=pd.read_csv(file_name,keep_default_na=False)
#df.columns
print("DATA LOAD DONE")

# In[3]:


df_denied = df[df['CASE_STATUS'] == 1] #1 = DiNED


# In[4]:


df_certified = df[df['CASE_STATUS'] == 0] #1 = CERTIFIED


# In[5]:


Input_Certified, Input_Certified_extra, y_certified, y_certified_extra = train_test_split(df[df.CASE_STATUS == 0],
                                                                                          df_certified.CASE_STATUS, train_size= 0.06, random_state=1)


# In[6]:


df_train=Input_Certified.append(df_denied)


# In[7]:


#df_train.CASE_STATUS.value_counts()


# In[8]:


df_train=pd.get_dummies(df_train,columns=['YEAR', 'SOC_CODE', 'STATE','WAGE_CATEGORY'])
print("DATA DUMMY DONE")

# In[20]:


#df_train


# In[10]:

'''
model = LogisticRegression()
rfe = RFE(model, 30)
fit = rfe.fit(df_train.iloc[:,1:], df_train.iloc[:,0])
support_rfe = rfe.support_
length_cols = list(df_train.iloc[:,1:].columns.values)
list_selected = []
for index in range(len(length_cols)):
    if support_rfe[index] == True:
        list_selected.append(length_cols[index])
    else:
        pass
print(list_selected)
print(rfe.ranking_) 
unique_listcols = [col.split('_')[0] for col in list_selected]
set(unique_listcols)
'''

# In[11]:


X = df_train.drop('CASE_STATUS', axis=1)
y = df_train.CASE_STATUS
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
#X_train.columns
print("DATA SPLIT DONE")

# In[12]:


df_train['CASE_STATUS'].value_counts()


# In[13]:


dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train) 


# In[14]:


y_pred = dtree.predict(X_test)
y_prob = dtree.predict_proba(X_test)


# In[15]:


print("DTREE")
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[16]:


mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20), max_iter=1000)
mlp.fit(X_train, y_train)


# In[17]:


y_pred_mlp = mlp.predict(X_test)
confusion = metrics.confusion_matrix(y_test, y_pred_mlp)
print("MLP")
print(confusion)
print(metrics.classification_report(y_test, y_pred_mlp))


# In[18]:


gaus_clf = GaussianNB()
gaus_clf.fit(X_train, y_train)


# In[19]:


y_pred_glb = gaus_clf.predict(X_test)
confusion = metrics.confusion_matrix(y_test, y_pred_glb)
print("GUAS")
print(confusion)
print(metrics.classification_report(y_test, y_pred_glb))

"""save model"""
save_dtree=file_path+'DTREE_Model_h1b.sav'
pickle.dump(dtree, open(save_dtree, 'wb'))
print("DTREE_Model_h1b SAVE DONE")

save_mlp=file_path+'MLP_Model_h1b.sav'
pickle.dump(dtree, open(save_mlp, 'wb'))
print("MLP_Model_h1b SAVE DONE")

save_gaus=file_path+'GAUS_Model_h1b.sav'
pickle.dump(dtree, open(save_gaus, 'wb'))
print("GAUS_Model_h1b SAVE DONE")

# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import tree 

from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB 



from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics
from sklearn.externals import joblib


# In[23]:


file_path='./datas/'
file_name=file_path+"h1b_train_final_data_ry.csv"
df=pd.read_csv(file_name,keep_default_na=False)
#df.columns


# In[24]:


df=df.drop("PREVAILING_WAGE",axis=1)


# In[25]:


df_denied = df[df['CASE_STATUS'] == 1] #1 = DiNED


# In[26]:


df_certified = df[df['CASE_STATUS'] == 0] #1 = CERTIFIED


# In[27]:


df["CASE_STATUS"].value_counts()


# In[28]:


Input_Certified, Input_Certified_extra, y_certified, y_certified_extra = train_test_split(df[df.CASE_STATUS == 0],
                                                                                          df_certified.CASE_STATUS, train_size= 0.32, random_state=1)


# In[29]:


df_train=Input_Certified.append(df_denied)
#df_train=df


# In[30]:


df_train.CASE_STATUS.value_counts()


# In[34]:


#df_train.columns


# In[35]:


#df_train=df_train[['CASE_STATUS','FULL_TIME_POSITION','YEAR', 'STATE','SOC_CODE','real_wage_mean_soc','real_medianwage_state','real_meanwage_state','median_WAGE_by_YEAR', 'ratio_by_SOC', 'ratio_by_YEAR']]
df_train=pd.get_dummies(df_train,columns=['YEAR', 'SOC_CODE', 'STATE',
       'FULL_TIME_POSITION'])


# In[36]:


df_train=df_train.drop('ID',axis=1)


# In[37]:


X = df_train.drop('CASE_STATUS', axis=1)
y = df_train.CASE_STATUS
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
#X_train.columns


# In[38]:


#df_train['CASE_STATUS'].value_counts()


# In[13]:


#X_train


# In[39]:


dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train) 


# In[40]:


y_pred = dtree.predict(X_test)
y_prob = dtree.predict_proba(X_test)


# In[41]:


print("test", y_test[:10])
print("pred", y_pred[:10])
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[42]:

'''
mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20), max_iter=1000)
mlp.fit(X_train, y_train)


# In[ ]:


y_pred_mlp = mlp.predict(X_test)
confusion = metrics.confusion_matrix(y_test, y_pred_mlp)
print(confusion)
print(metrics.classification_report(y_test, y_pred_mlp))


# In[19]:

'''
gaus_clf = GaussianNB()
gaus_clf.fit(X_train, y_train)


# In[20]:


y_pred_glb = gaus_clf.predict(X_test)
confusion = metrics.confusion_matrix(y_test, y_pred_glb)
print(confusion)
print(metrics.classification_report(y_test, y_pred_glb))


# In[21]:


type="train2_"
save_dtree=file_path+type+'DTREE_Model_h1b'
save_mlp=file_path+type+'MLP_Model_h1b'
save_gaus=file_path+type+'GAUS_Model_h1b'


# In[22]:

'''
joblib.dump(dtree, save_dtree) 
joblib.dump(mlp, save_mlp) 
joblib.dump(gaus_clf, save_gaus)
'''

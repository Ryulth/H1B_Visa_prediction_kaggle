
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from statistics import mode
import re
from xgboost import XGBClassifier


# In[2]:


file_path='./datas/'
file_name=file_path+"h1b_dev_preprocessing.csv"
df=pd.read_csv(file_name,keep_default_na=False)
df.columns


# In[3]:


df[df.STATE.isna()]


# In[4]:


df


# In[5]:


df[['CASE_STATUS', 'FULL_TIME_POSITION', 'YEAR','SOC_CODE', 'STATE']]=df[['CASE_STATUS', 'FULL_TIME_POSITION', 'YEAR','SOC_CODE', 'STATE']].apply(lambda x : x.astype('category'))


# In[8]:


X = df.drop('CASE_STATUS', axis=1)
y = df.CASE_STATUS
seed = 6
test_size = 0.40
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
X_train.columns


# In[9]:


X_train_encode = pd.get_dummies(X_train)
X_test_encode = pd.get_dummies(X_test)


# In[10]:


train_X = X_train_encode.as_matrix()
train_y = y_train.as_matrix()


# In[11]:


gbm=XGBClassifier(max_features='sqrt', subsample=0.8, random_state=10)


# In[12]:


from sklearn.model_selection import GridSearchCV


# In[13]:


parameters = [{'n_estimators': [10, 100]},
              {'learning_rate': [0.1, 0.01, 0.5]}]


# In[14]:


grid_search = GridSearchCV(estimator = gbm, param_grid = parameters, scoring='accuracy', cv = 3, n_jobs=-1)


# In[15]:


grid_search = grid_search.fit(train_X, train_y)


# In[16]:


grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_


# In[17]:


grid_search.best_estimator_


# In[18]:


gbm=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.5, max_delta_step=0,
       max_depth=3, max_features='sqrt', min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=10, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=0.8).fit(train_X, train_y)


# In[19]:


y_pred = gbm.predict(X_test_encode.as_matrix())


# In[20]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import scale


# In[6]:


file_path='./datas/'
file_name=file_path+"real_ultimate_train.csv"
df=pd.read_csv(file_name,keep_default_na=False)
#df.columns


# In[8]:


#df


# In[9]:


df=df.drop("ID",axis=1)
df=df.drop("EMPLOYER_NAME",axis=1)
df=df.drop("SOC_NAME",axis=1)
df=df.drop("JOB_TITLE",axis=1)
df=df.drop("WORKSITE",axis=1)
df=df.drop("lon",axis=1)
df=df.drop("lat",axis=1)
df=df.drop("CITY",axis=1)


# In[11]:


df.PREVAILING_WAGE=scale(df.PREVAILING_WAGE)


# In[12]:


df


# In[13]:


save_file=file_path+'h1b_train_final_data.csv'
df.to_csv(save_file,index=False)

type="train2_"
save_dtree=file_path+type+'DTREE_Model_h1b'
save_mlp=file_path+type+'MLP_Model_h1b'
save_gaus=file_path+type+'GAUS_Model_h1b'


# In[22]:


joblib.dump(dtree, save_dtree)
joblib.dump(mlp, save_mlp)
joblib.dump(gaus_clf, save_gaus)

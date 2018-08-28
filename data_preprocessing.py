
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file_path='./datas/'
file_name=file_path+"h1b_dev_match_soc_code.csv"
df=pd.read_csv(file_name)
#train_data=train_data.loc[(train_data['CASE_STATUS']=='CERTIFIED') | (train_data['CASE_STATUS']=='DENIED')]


# In[3]:


df.loc[1572]


# In[4]:


df['STATE'] = df.WORKSITE.str.split('\s+').str[-1]


# In[5]:


class_mapping = {'CERTIFIED':0, 'DENIED':1}
df["CASE_STATUS"] = df["CASE_STATUS"].map(class_mapping)


# In[6]:


df[df["SOC_CODE"]=="0o0"]


# In[7]:


df.isnull().sum()


# In[8]:


df=df.drop("Unnamed: 0",axis=1)
df=df.drop("EMPLOYER_NAME",axis=1)
df=df.drop("SOC_NAME",axis=1)
df=df.drop("JOB_TITLE",axis=1)
df=df.drop("WORKSITE",axis=1)
df=df.drop("lon",axis=1)
df=df.drop("lat",axis=1)


# In[9]:


df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].fillna(df['FULL_TIME_POSITION'].mode()[0])
df['PREVAILING_WAGE']=df['PREVAILING_WAGE'].fillna(0)


# In[10]:


df


# In[11]:


save_file=file_path+'h1b_dev_preprocessing.csv'
df.to_csv(save_file,index=False)


# In[12]:


df


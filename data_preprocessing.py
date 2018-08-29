
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file_path='./datas/'
file_name=file_path+"h1b_test_match_soc_code.csv"
df=pd.read_csv(file_name)
#train_data=train_data.loc[(train_data['CASE_STATUS']=='CERTIFIED') | (train_data['CASE_STATUS']=='DENIED')]


# In[3]:


def wage_categorization(wage):
    if wage <=50000 and wage>= 0:
        return "VERY LOW"
    elif wage >50000 and wage <= 70000:
        return "LOW"
    elif wage >70000 and wage <= 90000:
        return "MEDIUM"
    elif wage >90000 and wage<=150000:
        return "HIGH"
    elif wage >=150000:
        return "VERY HIGH"
    elif wage < 0 :
        return "NON"


# In[4]:


#df.loc[1572]


# In[5]:


df['STATE'] = df.WORKSITE.str.split('\s+').str[-1]


# In[6]:


df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].fillna(df['FULL_TIME_POSITION'].mode()[0])
df['PREVAILING_WAGE']=df['PREVAILING_WAGE'].fillna(-100)


# In[7]:


#case_mapping = {'CERTIFIED':0, 'DENIED':1}
fp_mapping = {"N" :0,"Y" : 1}
#df["CASE_STATUS"] = df["CASE_STATUS"].map(case_mapping)
df["FULL_TIME_POSITION"]=df["FULL_TIME_POSITION"].map(fp_mapping)


# In[8]:


df["YEAR"]=df["YEAR"].astype('int')


# In[9]:


#df[df["SOC_CODE"]=="0o0"]

#In
# In[10]:


df['WAGE_CATEGORY'] = df['PREVAILING_WAGE'].apply(wage_categorization)


# In[11]:


df.isnull().sum()


# In[12]:


df=df.drop("Unnamed: 0",axis=1)
df=df.drop("EMPLOYER_NAME",axis=1)
df=df.drop("SOC_NAME",axis=1)
df=df.drop("JOB_TITLE",axis=1)
df=df.drop("WORKSITE",axis=1)
df=df.drop("lon",axis=1)
df=df.drop("lat",axis=1)
df=df.drop('PREVAILING_WAGE',axis=1)


# In[14]:


save_file=file_path+'h1b_test_preprocessing.csv'
df.to_csv(save_file,index=False)


# In[15]:




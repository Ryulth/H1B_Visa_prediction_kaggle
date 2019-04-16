
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time # 코드 수행 시간


# In[ ]:


file_path='./datas/'
file_name=file_path+"h1b_train.csv"
train=pd.read_csv(file_name)
train=train.loc[(train['CASE_STATUS']=='CERTIFIED') | (train['CASE_STATUS']=='DENIED')]


# In[ ]:


def init_train_groupby () :
    train_groupby=pd.DataFrame({'COUNT': train.groupby(['EMPLOYER_NAME','SOC_NAME','JOB_TITLE','FULL_TIME_POSITION']).size().sort_values(ascending=False)}).reset_index()
    train_groupby["FREQUENCY"]=0
    return train_groupby


# In[ ]:


train_groupby=init_train_groupby()
train_groupby


# In[ ]:


train[train.SOC_NAME.isna()]


# In[ ]:


def match_soc_name(employer_name,job_title,full_time_position):
    """
    최빈값으로  socname 입력해주기
    EMPLOYER SAME -> 4 add
    JOB_TITLE -> 2 add
    FULL_TIME SAME -> 1 add
    가중치설정
    """
    train_groupby["FREQUENCY"]=0
    #print (employer_name,job_title,full_time_position)
    #print (type((employer_name)))
    train_groupby.loc[(train_groupby['EMPLOYER_NAME']==employer_name),'FREQUENCY'] +=4
    train_groupby.loc[(train_groupby['JOB_TITLE']==job_title),'FREQUENCY']+=2
    train_groupby.loc[(train_groupby['FULL_TIME_POSITION']==full_time_position),'FREQUENCY'] +=1
    return train_groupby.sort_values(['FREQUENCY','COUNT'],ascending=[False,False]).head(1).SOC_NAME.values[0]


# In[ ]:


start_time = time.time() 
train.loc[train['SOC_NAME'].isnull(),'SOC_NAME'] = train.apply(lambda x : match_soc_name(x['EMPLOYER_NAME'],x['JOB_TITLE'],x['FULL_TIME_POSITION']) if pd.isnull(x['SOC_NAME']) else "",axis=1)
end_time = time.time()


# In[ ]:


print (end_time-start_time)


# In[ ]:


save_file=file_path+'h1b_train_fill_soc_na.csv'
train.to_csv(save_file,index=False)


# In[ ]:


train.head(50)


# In[ ]:


type(train[train['SOC_NAME'].isna()].SOC_NAME)


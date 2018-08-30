
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time 
from multiprocessing import Pool , cpu_count
num_cores = cpu_count()


# In[ ]:


file_path='./datas/'
file_name=file_path+"h1b_train.csv"
train_data=pd.read_csv(file_name)
train_data=train_data.loc[(train_data['CASE_STATUS']=='CERTIFIED') | (train_data['CASE_STATUS']=='DENIED')]
train_data["ID"]=train_data.index

# In[ ]:


def init_train_groupby () :
    train_groupby=pd.DataFrame({'COUNT': train_data.groupby(['EMPLOYER_NAME','SOC_NAME','JOB_TITLE','FULL_TIME_POSITION']).size().sort_values(ascending=False)}).reset_index()
    train_groupby["FREQUENCY"]=0
    return train_groupby


# In[ ]:


train_data_soc_na=train_data[train_data.SOC_NAME.isna()]
train_data_soc_not_na=train_data.dropna(subset=['SOC_NAME'])
#train_data_soc_na


# In[ ]:


train_groupby=init_train_groupby()
#train_groupby


# In[ ]:


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_cores)
    pool = Pool(processes=num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# In[ ]:


cnt = 1
def match_soc_name(employer_name,job_title,full_time_position):
    """
    최빈값으로  socname 입력해주기
    EMPLOYER SAME -> 4 add
    JOB_TITLE -> 2 add
    FULL_TIME SAME -> 1 add
    가중치설정
    """
    train_groupby["FREQUENCY"]=0
    #global cnt
    #cnt +=1
    #print (cnt)
    #print (type((employer_name)))
    global cnt
    if (cnt % 1000==0):
        print(cnt)
    cnt += 1
    train_groupby.loc[(train_groupby['EMPLOYER_NAME']==employer_name),'FREQUENCY'] +=4
    train_groupby.loc[(train_groupby['JOB_TITLE']==job_title),'FREQUENCY']+=2
    train_groupby.loc[(train_groupby['FULL_TIME_POSITION']==full_time_position),'FREQUENCY'] +=1
    return train_groupby.sort_values(['FREQUENCY','COUNT'],ascending=[False,False]).head(1).SOC_NAME.values[0]


# In[ ]:


def apply_soc_name(data):
    print("working process" ,data)
    data.loc[data['SOC_NAME'].isnull(),'SOC_NAME'] = data.apply(lambda x : match_soc_name(x['EMPLOYER_NAME'],x['JOB_TITLE'],x['FULL_TIME_POSITION']) if pd.isnull(x['SOC_NAME']) else "",axis=1)
    return data


# In[ ]:


if __name__ == '__main__':
    start_time = time.time() 
    train_data_soc_fill_na=parallelize_dataframe(train_data_soc_na, apply_soc_name)
    end_time = time.time()
    task1 = end_time - start_time
    now = time.gmtime(task1)
    print(now.tm_hour, now.tm_min, now.tm_sec)

    soc_fillna_data = pd.concat([train_data_soc_not_na, train_data_soc_fill_na])

    # In[ ]:

    soc_fillna_data.sort_index(inplace=True)
    #soc_fillna_data

    # In[ ]:

    save_file = file_path + 'h1b_train_fillna_soc_data.csv'
    soc_fillna_data.to_csv(save_file, index=False)
# In[ ]:




# In[ ]:




# In[ ]:


#train_data_soc_fill_na


# In[ ]:




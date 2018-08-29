
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_state_ratio=pd.read_csv('./datas/h1b_train_ratio.csv',keep_default_na=False)
df_soccode_ratio=pd.read_csv('./datas/soccode_with_ratio.csv',keep_default_na=False)


# In[43]:


file_path='./datas/'
file_name=file_path+"h1b_train_preprocessing.csv"
df=pd.read_csv(file_name,keep_default_na=False)


# In[13]:


df_state_ratio=df_state_ratio.drop("CERTIFIED",axis=1)
df_state_ratio=df_state_ratio.drop("DENIED",axis=1)


# In[26]:


df_state_ratio=df_state_ratio.set_index('STATE')
ratio_dict=df_state_ratio.to_dict()
ratio_mapping=ratio_dict['ratio']


# In[29]:


df['STATE_RATIO']=df['STATE'].map(ratio_mapping)


# In[33]:


df_soccode_ratio=df_soccode_ratio.drop("Unnamed: 0",axis=1)
df_soccode_ratio


# In[37]:


df_soccode_ratio=df_soccode_ratio.set_index('SOC_CODE')


# In[38]:


soccode_ratio_dict=df_soccode_ratio.to_dict()
soccode_ratio_mapping=soccode_ratio_dict['ratio_soc']


# In[39]:


df['SOC_RATIO']=df['SOC_CODE'].map(soccode_ratio_mapping)


# In[44]:


df


# In[45]:


soccode_ratio_mapping


# In[51]:


df_soccode_ratio.sort_values('ratio_soc')


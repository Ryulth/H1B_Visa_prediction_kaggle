
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file_path='./datas/'
file_name=file_path+"h1b_train_preprocessing.csv"
df=pd.read_csv(file_name,keep_default_na=False)
#df.columns
print("DATA LOAD DONE")


# In[157]:


df["ID"]=df.index


# In[163]:


df["ID"]=str(df["ID"].values)+"1"


# In[164]:


df


# In[37]:


import itertools


# In[87]:


x=[]

num=0o0
list_=np.array([0o0,0o400,0o200,0o100,0o040,0o020,0o010,0o04,0o02,0o01])
for i in range (0,len(list_)) :
   num=0o0
   num+=list_[i]
   x.append((np.array(list(itertools.combinations(list_,i)))))
   print((np.array(list(itertools.combinations(list_,i)))))
   print(oct(int(num)))
   #print (oct(int(0o400/0o010)))

x


# In[65]:


oct(sum(x[1][3]))


# In[89]:


x=np.array(x)


# In[92]:


len(x)


# In[91]:


x.flatten()


# In[113]:


temp10=[]
for i in range(0,len(x)) :
    for j in range(0,len(x[i])):
        temp10.append(oct(sum(x[i][j])))
temp9=np.array(temp10)


# In[117]:


np.unique(temp9)


# In[111]:


temp2=pd.read_csv('./datas/h1b_train_ratio.csv',keep_default_na=False)
temp3=pd.read_csv('./datas/soccode_with_ratio.csv',keep_default_na=False)


# In[112]:


temp2


# In[127]:


temp3.sort_values('ratio_soc')


# In[118]:


tx=[0o0400,0o020,0o010]


# In[126]:


(tx.sort(reverse=True))[0]


# In[124]:


tx


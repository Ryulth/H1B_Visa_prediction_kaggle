
# coding: utf-8

# ## BOAZ Mini Project
# 
# ### h1b data [h1b_train, h1b_dev, h1b_test]
# 
# - Use h1b __train data__ for train your model.
# - Tune your model with __dev data__.
# - Finally check your best performed model score with __test data__.
# - __*You should clean your train dev test data first!!*__
# - Score function : Accuracy
# <br>
# <br>
# - __Your goal is to predict CASE_STATUS, using other features.__
# - You can choose the features you want to use in your project.

# ### Attribute Information
# 
# - __CASE_STATUS__
#     - The CASE_STATUS field denotes the status of the application after LCA processing. Certified applications are filed with USCIS for H-1B approval.
# <br>
# <br>
# - EMPLOYER_NAME
#     - Name of the employer submitting labor condition application.
# <br>
# <br>
# - SOC_NAME
#     - Occupational name associated with the SOC_CODE. SOC_CODE is the occupational code associated with the job being requested for temporary labor condition, as classified by the Standard Occupational Classification (SOC) System.
# <br>
# <br>
# - JOB_TITLE
#     - Title of the job.
# <br>
# <br>
# - FULL_TIME_POSITION
#     - Y = Full Time Position; N = Part Time Position.
# <br>
# <br>
# - PREVAILING_WAGE
#     - Prevailing Wage for the job being requested for temporary labor condition. The wage is listed at annual scale in USD. The prevailing wage for a job position is defined as the average wage paid to similarly employed workers in the requested occupation in the area of intended employment. The prevailing wage is based on the employer’s minimum requirements for the position.
# <br>
# <br>
# - YEAR
#     - Year in which the H-1B visa petition was filed.
# <br>
# <br>

# ### In this assignment, you will design, implement, and evaluate the appropriate models for given data.
# 0. Preprocess; normalization, feature selection, etc.
# 1. Model selection; characteristics of datasets need to be comprehended.
# 2. Evaluation; This step should be done properly to prevent overfitting problem."
# 3. Enhancement; parameter tuning and feature selection, etc.

# In[1]:


import pandas as pd
file_path="./datas/"


# In[2]:


file_name=file_path+"h1b_train.csv"
train=pd.read_csv(file_name)


# In[25]:


temp=train['lan']==train[]


# In[27]:


train.sort_values(by=["PREVAILING_WAGE"])


# In[4]:


label=pd.read_csv("h1b_test_no_ylabel.csv")


# In[18]:


label.sort_values(by=["Unnamed: 0"]).tail(10)


# In[6]:


file_name_dev=file_path+"h1b_dev.csv"
dev=pd.read_csv(file_name_dev)


# In[17]:


dev.sort_values(by=["Unnamed: 0"]).tail(10)


# In[30]:


import re


# In[34]:


train[train.EMPLOYER_NAME==str(re.compile('GOOGLE',re.I))]


# In[53]:


train[train.EMPLOYER_NAME.str.contains('google',case=False)==True]


# In[54]:


train=train.loc[(train_data['CASE_STATUS']=='CERTIFIED') | (train_data['CASE_STATUS']=='DENIED')


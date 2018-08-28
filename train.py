
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re


# In[2]:


import osㅇㅇ
print (os.getcwd()) #현재 디렉토리의


# In[ ]:


file_path='./datas/'
file_name=file_path+'h1b_train.csv'
train=pd.read_csv(file_name)
train=train.loc[(train['CASE_STATUS']=='CERTIFIED') | (train['CASE_STATUS']=='DENIED')]


# In[ ]:


train


# In[ ]:


count_nan = len(train) - train.count()


# In[ ]:


train[train.JOB_TITLE.str.contains('TECHNICAL CONSULTANT',case=False)==True]


# In[ ]:


train_soc_na=train[train.SOC_NAME.isna()]


# In[ ]:


train_soc_na


# In[ ]:


train_temp=train[(train['JOB_TITLE']== 'TECHNOLOGY LEAD - US') 
                 &(train['SOC_NAME']=='Computer Systems Analysts')
                 &(train['EMPLOYER_NAME']=='INFOSYS LIMITED') ]


# In[ ]:


train_temp


# In[ ]:


train.JOB_TITLE.str.replace(' ','').str.lower()


# In[ ]:


train_groupby=pd.DataFrame({'COUNT': train.groupby(['EMPLOYER_NAME','SOC_NAME','JOB_TITLE','FULL_TIME_POSITION']).size().sort_values(ascending=False)}).reset_index()


# In[ ]:


temp.loc[(temp['FULL_TIME_POSITION']== 'Y') 
                 &(temp['JOB_TITLE']=='TECHNOLOGY LEAD - US')
                 &(temp['EMPLOYER_NAME']=='INFOSYS LIMITED') ,'temp'] +=3


# In[ ]:


temp['temp']=0


# In[ ]:


train_groupby["FREQUENCY"]=0


# In[ ]:


train_groupby


# In[ ]:


train_groupby.loc[(train_groupby['EMPLOYER_NAME']=='INFOSYS LIMITED'),'FREQUENCY'] +=4
train_groupby.loc[(train_groupby['JOB_TITLE']=='TECHNOLOGY LEAD'),'FREQUENCY']+=2
train_groupby.loc[(train_groupby['FULL_TIME_POSITION']=='Y'),'FREQUENCY'] +=1
a=train_groupby.sort_values(['FREQUENCY','COUNT'],ascending=[False,False]).head(1).SOC_NAME


# In[ ]:


train_groupby.sort_values(['FREQUENCY','COUNT'],ascending=[False,False]).head(1).SOC_NAME.values[0]


# In[ ]:


train_groupby.sort_values(['FREQUENCY','COUNT'],ascending=[False,False])


# In[ ]:


bin(7)


# In[ ]:



if not temp[(temp['FULL_TIME_POSITION']== 'Y') 
                 &(temp['JOB_TITLE']=='TECHNOLOGY LEAD - US')
                 &(temp['EMPLOYER_NAME']=='INFOSYS LIMITED') ].SOC_NAME.empty :
    print (temp[(temp['FULL_TIME_POSITION']== 'Y') 
                 &(temp['JOB_TITLE']=='TECHNOLOGY LEAD - US')
                 &(temp['EMPLOYER_NAME']=='INFOSYS LIMITED') ].SOC_NAME[0])


# In[ ]:


train.JOB_TITLE.str.replace(' ','').str.lower().value_counts()


# In[ ]:


df = pd.DataFrame(np.random.randint(0, 3, (10, 4)), columns=list('abcd'))
df


# In[ ]:


df.apply(pd.Series.value_counts)


# In[ ]:


def set_soc_code():
    """
    MAKE JOB OCT CODE 팔진수 코드를 부여해 보기 쉽게 한다  
    0o0: OTHERS 
    0o400 : COMPUTER 
    0o200 : MEDICAL
    0o100 : SCIENCES
    0o040 : EDUCATION
    0o020 : MANAGE
    0o010 : BUSSINESS
    0o004 : TECHNOLOGY
    0o002 : FINANCIAL
    0o001 : LIVESTOCK
    """
    code_computer=np.array([0o400,'com','software','programmer','project',
                   'developer','database','consultant', 'manager'])
    code_medical=np.array([0o200,'medical','doctor','physic','dentist',
                          'surgeon','nurse','psych','health'])
    code_sci=np.array([0o100,'chemist','physicist','bio','scientist',
              'clinical','math','statistic','predictive','stats'])
    code_edu=np.array([0o40,'teach','linguist','professor','school','principal'])
    code_manage=np.array([0o20,'relation','manage','operation','chief','plan','executive'])
    code_bussiness=np.array([0o10,'advertis','marketing','business','research','promotion'])
    code_tech=np.array([0o4,'techno','engineer','surveyor','carto',
             'architect','drafter','information','security'])
    code_finacial=np.array([0o2,'accountant','finan'])
    code_livestock=np.array([0o1,'water','butcher'])
    code_list=[code_computer,code_medical,code_sci,code_edu,code_manage,
               code_bussiness,code_tech,code_finacial,code_livestock]
    soc_code_array= np.array(code_list)
    return soc_code_array
soc_code_array=set_soc_code()


# In[ ]:


def match_soc_code(soc_name_input) :
    soc_code_num=0o0 # 나중에 반환될값
    soc_name_series=pd.Series(str(soc_name_input).split(' ')) # 글자를 띄어쓰기 별로 구분해서 Series 화 시킨다
    soc_find=lambda x: soc_name_series[soc_name_series.str.contains('|'.join(x),case=False)].size>=1 # 람다식 함수 생성  or 를 join 으로 다 붙여준후에 contains 해주고 그게 크기를 구하는 함수
    soc_result=np.array(list(filter(soc_find,soc_code_array)))#필터를 통해 아까 람다식이 참인 것들만 반환해 준다
    soc_code_list=[int(i[0])  for i in soc_result  ]#팔진수를 인티저 형으로 받아서 더해준다
    soc_code_num=sum(soc_code_list)
    if (soc_code_num>=1) : # 카테고리화 되지 못한 항목은 0으로 유지 선택된 것은 맨앞에 1을 붙여주어 구별하기 편하게 한다.
        soc_code_num+=0o1000 
    return oct(soc_code_num) # 카테고리화 된 번호 반환


# In[ ]:


train["SOC_NAME"].head(1000).map(lambda x : match_soc_code(x))


# In[ ]:


train["SOC_NAME"].head(10).map(lambda x : match_soc_code(x))


# In[ ]:


train["SOC_NAME"].head(100)


# In[ ]:


temp=[['ddd', 'aikkt'],['og','adt'],['et','f']]
searchfor =  np.array(temp)
s = pd.Series(['cat','hat','dog','fog','pet'])
#s[s.str.contains('|'.join(searchfor),case=False)]


# In[ ]:


do=lambda x: s[s.str.contains('|'.join(x),case=False)].size>=1

a=list(map(do,searchfor))
b=list(filter(do,searchfor))
c= [i[0] for i in b ]
c


# In[ ]:


str="COMPUTER"
l = ['COMPUTER','SOFEWARE','PROGRAMMER',
            'PROJECT','DEVELOPER','DATA','CONSULTANT',
            'MANAGER']
def find_char(str1,list_):
    x=[i  for i in list_ if i in str1]
    return x
find_char(str,l)


# In[ ]:


print ( bin (10))


# In[ ]:


oct(0o10+0o20)


# In[ ]:


a=0o400
a


# In[ ]:


bin(a)


# In[ ]:


oct(a)


# In[ ]:


df1 = pd.DataFrame(np.random.randint(1, 5, (10,2)), columns=['a','b'])
df1


# In[ ]:


df1.sort_values(['a', 'b'], ascending=[False, False])


# In[ ]:


train


# In[ ]:


df_split=np.array_split(train,4)


# In[ ]:


df_split[0]


# In[7]:


df = pd.DataFrame({'A': [1, 2, 1, 2, 1, 2, 3,3,3,3]})
df.mode()


# In[3]:


df



# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time 
from multiprocessing import Pool , cpu_count
num_cores = cpu_count()


# In[2]:


file_path='./datas/'
file_name=file_path+"h1b_train_fillna_soc_data.csv"
train_data=pd.read_csv(file_name)
#train_data=train_data.loc[(train_data['CASE_STATUS']=='CERTIFIED') | (train_data['CASE_STATUS']=='DENIED')]


# In[3]:


def set_soc_code():
    """
    MAKE JOB OCT CODE 팔진수 코드를 부여해 보기 쉽게 한다  ->나중에 2진수로 변환시 보기 매우 편리
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
                   'developer','database','consultant', 'manager','graphic',
                           'multimedia','animators','program'])
    code_medical=np.array([0o200,'medical','doctor','physic','dentist',
                          'surgeon','nurse','psych','health','internists',
                          'pharmacists','practitioners','therapist',
                          'pediatricians','pathologist','veterinarians',
                          'Chiropractor','Rehabilitation','Gynecologist',
                          'Obstetrician','Anesthesiologist','OPTOMETRIST'])
    code_tech=np.array([0o100,'techno','engineer','surveyor','carto','architect',
                        'drafter','information','security','designers','machine',
                       'tech','repairer','interpreters','EDITOR','Mechanic'])
    code_edu=np.array([0o40,'teach','linguist','professor','school','principal','edu',
                      'instructional'])
    code_manage=np.array([0o20,'relation','manage','operation','chief','plan',
                          'executive','resources','coache','training','lawyer',
                         'legal','law','curator','Librarian','Recruiter'])
    code_bussiness=np.array([0o10,'advertis','marketing','business','research',
                             'promotion','logisticians','economist','auditor',
                            'sale','retail'])
    code_sci=np.array([0o4,'chemist','physicist','bio','scientist','Epidemiologist',
              'clinical','math','statistic','predictive','stats','Astronomer'])
    code_finacial=np.array([0o2,'accountant','finan','cost','estimators',
                           'credit','actuaries','budget','accounting','Insurance '])
    code_livestock=np.array([0o1,'water','butcher','fashion','scout',
                            'art','interior','MILLWRIGHTS','gem','bartender',
                            'shoe','directors','correspondents','model','food',
                             'ship','author','REPORTER','chef','cook','Dietitian',
                            'Athletic','Fitness'])
    code_list=[code_computer,code_medical,code_sci,code_edu,code_manage,
               code_bussiness,code_tech,code_finacial,code_livestock]
    soc_code_array= np.array(code_list)
    return soc_code_array


# In[4]:


soc_code_array=set_soc_code()


# In[5]:


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_cores)
    pool = Pool(processes=num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# In[ ]:

cnt=1
def match_soc_code(soc_name_input) :
    soc_code_num=0o0 # 나중에 반환될값
    soc_name_series=pd.Series(str(soc_name_input).split(' ')) # 글자를 띄어쓰기 별로 구분해서 Series 화 시킨다
    soc_find=lambda x: soc_name_series[soc_name_series.str.contains('|'.join(x),case=False)].size>=1 # 람다식 함수 생성  or 를 join 으로 다 붙여준후에 contains 해주고 그게 크기를 구하는 함수
    soc_result=np.array(list(filter(soc_find,soc_code_array)))#필터를 통해 아까 람다식이 참인 것들만 반환해 준다
    soc_code_list=[int(i[0])  for i in soc_result  ]#팔진수를 인티저 형으로 받아서 더해준다
    #soc_code_num=sum(soc_code_list)
    global cnt

    if(cnt%1000==0) :
        print (cnt)
    cnt += 1
    if (len(soc_code_list)>=1) : # 카테고리화 되지 못한 항목은 0으로 유지 선택된 것은 맨앞에 1을 붙여주어 구별하기 편하게 한다.
        soc_code_num+=0o1000
        soc_code_list.sort(reverse=True)
        soc_code_num += soc_code_list[0]
    return oct(soc_code_num) # 카테고리화 된 번호 반환


# In[ ]:


num=1
def map_soc_code(data) :
    global num
    print("working process" ,num)
    num +=1

    data["SOC_CODE"]=data["SOC_NAME"].map(lambda x : match_soc_code(x))
    return data


# In[ ]:


if __name__ == '__main__':
    start_time = time.time() 
    train_data_match_soc_code=parallelize_dataframe(train_data, map_soc_code)
    end_time = time.time()
    task1=end_time-start_time
    now = time.gmtime(task1)
    print(now.tm_hour, now.tm_min, now.tm_sec)
    save_file = file_path + 'h1b_train_match_soc_code.csv'
    train_data_match_soc_code.to_csv(save_file, index=False)

# In[ ]:





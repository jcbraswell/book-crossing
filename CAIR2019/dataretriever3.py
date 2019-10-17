#!/usr/bin/env python
# coding: utf-8

# General
from __future__ import print_function, division
import sys

# Database
import cx_Oracle
from sqlalchemy import create_engine
from getpass import getpass

# Tools
import pandas as pd
import seaborn as sns
import random
import string
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas_profiling

pd.options.display.max_columns = None

# Analytics
#import pymc3 as pm
from scipy.stats import beta

#IMBALANCED DATA
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

#SciKitLearn Models
from sklearn.linear_model import LogisticRegression, ElasticNetCV, SGDClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier #(wait for scikit release 18.0)
from sklearn.neighbors import KNeighborsClassifier

#XgBoost Model ###################################################################################
# import os
# mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'
# os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
##################################################################################################

#MODEL SELECTION, #EVALUATION METRICS
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn import preprocessing


# # Academic Data: Performance & Demographics

# #### Grades Data
# First we obtain the students grades along with a few variables. The goal is to aggregate the grade records to a semester summary.

# In[2]:


username = input('Enter username: ')


# In[3]:


password = getpass(prompt='Enter password: ')


# In[4]:


service_name = 'iraarch'
host = 'ira-oradb01.its.csulb.edu'
port = '1521'
grades_query = 'grd.sql'
dem_query = 'dae.sql'

def db_query(username, password, service_name, host, port, query):

    dsn = cx_Oracle.makedsn(host, port, service_name=service_name)

    cstr = 'oracle://{user}:{password}@{dsn}'.format(
        user=username,
        password=password,
        dsn=dsn
    )

    engine =  create_engine(
        cstr,
        convert_unicode=False,
        pool_recycle=10,
        pool_size=50,
    )

    with open(query, 'r') as f:
        data=f.read()#.replace('\n', '')
        
    return (data, engine)


# In[5]:


grades, engine = db_query(username, password, service_name, host, port, grades_query)
grd = pd.read_sql(grades, engine)

demo, engine = db_query(username, password, service_name, host, port, dem_query)
dem = pd.read_sql(demo, engine)


print(grd.shape)
print(dem.shape)


# In[6]:


grd.info(memory_usage='deep')


# In[7]:


dem.info(memory_usage='deep')


# In[8]:


#Change 'UNKNOWN' to more unique string to avoid having columns with same name after one-hot-encode

dem['first_generation'] = dem['first_generation'].apply(lambda x: 'First Generation Unknown' if x == 'UNKNOWN' else x)
dem['ethnicity'] = dem['ethnicity'].apply(lambda x: 'ETHNICITY UNKNOWN' if x == 'UNKNOWN' else x)


# ##### Create Training and Test/Validation Sets of Students at this stage to avoid leakeage 

# In[9]:


students = pd.DataFrame(grd['emplid'].unique(), columns=['emplid'])

print('there are {} students'.format(students.shape[0]))


# ##### Create the Grades Trainning and Validation Student Set

# In[10]:


students_train, students_dev = train_test_split(students, test_size=0.10, random_state=42)

students_train = pd.DataFrame(students_train)

students_dev = pd.DataFrame(students_dev)


# In[11]:


students_train.columns = ['EMPLID']
students_dev.columns = ['EMPLID']


# In[12]:


grd[grd['emplid']=='011155428'].sort_values(by=['term_code'])


# # Preprocessing: One-Hot-Encode Letter Grades

# In[13]:


grd.columns = map(str.upper, grd.columns)


# In[14]:


grd = pd.concat([grd,pd.get_dummies(grd['OFFICIAL_GRADE'], drop_first=True)], axis=1)

grd.shape


# ##### Create Variables to Calculate GPA

# In[15]:


grd['GRADE_POINTS_IN_GPA'] = grd['GRADE_POINTS'] * grd['OFFICIAL_GRADE'].apply(
    lambda x: None if x in ['AU','CR','NC','RD','RP','W','WE'] else 1
)

grd['UNITS_IN_GPA'] = grd['UNITS_TAKEN'] * grd['OFFICIAL_GRADE'].apply(
    lambda x: None if x in ['AU','CR','NC','RD','RP','W','WE'] else 1
)

grd['UNITS_FOR_CREDIT'] = grd['UNITS_TAKEN'] * grd['OFFICIAL_GRADE'].apply(
    lambda x: None if x in ['AU','NC','RD','RP','W','WE'] else 1
)

#######################################################################################################################

grd['BCMP_GRADE_POINTS_IN_GPA'] = grd['BCMP'] * grd['GRADE_POINTS'] * grd['OFFICIAL_GRADE'].apply(
    lambda x: None if x in ['AU','CR','NC','RD','RP','W','WE'] else 1
)

grd['BCMP_UNITS_IN_GPA'] = grd['BCMP_UNITS_TAKEN'] * grd['OFFICIAL_GRADE'].apply(
    lambda x: None if x in ['AU','CR','NC','RD','RP','W','WE'] else 1
)

grd['BCMP_UNITS_FOR_CREDIT'] = grd['BCMP'] * grd['UNITS_TAKEN'] * grd['OFFICIAL_GRADE'].apply(
    lambda x: None if x in ['AU','NC','RD','RP','W','WE'] else 1
)


# In[16]:


grd['SUMMER'] = (grd['CLASS_TERM'].apply(lambda x: str(x)[-1]) == '3')* 1 * grd['UNITS_FOR_CREDIT']

grd['WINTER'] = (grd['CLASS_TERM'].apply(lambda x: str(x)[-1]) == '1')* 1 * grd['UNITS_FOR_CREDIT']


# ##### Reduce the dataframe to variables of current interest

# In[17]:


grd = grd.sort_values(by=['EMPLID','TERM_CODE']).copy()[['COHORT', 
                 'EMPLID', 
                 'TERM_CODE',
                 'EOT_ACAD_PLAN_CD',
                 'GRADE_POINTS_IN_GPA',
                 'UNITS_TAKEN',
                 'UNITS_IN_GPA',
                 'UNITS_FOR_CREDIT',
                 'BCMP',
                 'BCMP_GRADE_POINTS_IN_GPA',
                 'BCMP_UNITS_TAKEN',
                 'BCMP_UNITS_IN_GPA',
                 'BCMP_UNITS_FOR_CREDIT',
                 'A', 
                 'AU', 
                 'B',
                 'C', 
                 'CR', 
                 'D', 
                 'F', 
                 'I', 
                 'NC', 
                 'RP', 
                 'W', 
                 'WE', 
                 'WU',
                 'SUMMER',
                 'WINTER']]


# ##### Aggregate and Reduce from Course Dimension to Term Dimension: Create a cummulative sum of Grade Points and GPA Units:

# In[18]:


aggregations = { 'GRADE_POINTS_IN_GPA':'sum',
                 'UNITS_TAKEN':'sum',
                 'UNITS_IN_GPA':'sum',
                 'UNITS_FOR_CREDIT':'sum',
                 'BCMP':'sum',
                 'BCMP_GRADE_POINTS_IN_GPA':'sum',
                 'BCMP_UNITS_TAKEN':'sum',
                 'BCMP_UNITS_IN_GPA':'sum',
                 'BCMP_UNITS_FOR_CREDIT':'sum',
                 'A':'sum', 
                 'AU':'sum', 
                 'B':'sum',
                 'C':'sum', 
                 'CR':'sum', 
                 'D':'sum', 
                 'F':'sum', 
                 'I':'sum', 
                 'NC':'sum', 
                 'RP':'sum', 
                 'W':'sum', 
                 'WE':'sum', 
                 'WU':'sum',
               'SUMMER':'sum',
               'WINTER':'sum'}

grouped_agg = grd.groupby(['COHORT','EMPLID','TERM_CODE','EOT_ACAD_PLAN_CD']).agg(aggregations).reset_index()


# In[19]:


grouped_agg.head()


# In[20]:


grouped_cumsum = grd[['EMPLID',
                             'TERM_CODE',
                             'EOT_ACAD_PLAN_CD',
                             'COHORT',
                             'GRADE_POINTS_IN_GPA',
                             'UNITS_TAKEN',
                             'UNITS_IN_GPA',
                             'UNITS_FOR_CREDIT',
                             'BCMP',
                             'BCMP_GRADE_POINTS_IN_GPA',
                             'BCMP_UNITS_TAKEN',
                             'BCMP_UNITS_IN_GPA',
                             'BCMP_UNITS_FOR_CREDIT',
                             'SUMMER',
                             'WINTER']].groupby(['EMPLID',
                                                 'TERM_CODE',
                                                 'COHORT',
                                                 'EOT_ACAD_PLAN_CD',]).sum().groupby(level=[0]).cumsum().reset_index()

grouped_cumsum = grouped_cumsum.add_prefix('CUM_')

grd = pd.concat([grouped_agg,grouped_cumsum],axis=1)

grd.drop(['CUM_EMPLID','CUM_TERM_CODE','CUM_COHORT','CUM_EOT_ACAD_PLAN_CD'],axis=1,inplace=True)


# #### Create Term and Cummulative GPA

# In[21]:


grd['TERM_GPA'] = grd['GRADE_POINTS_IN_GPA'] / grd['UNITS_IN_GPA']

grd['CUM_GPA'] = grd['CUM_GRADE_POINTS_IN_GPA'] / grd['CUM_UNITS_IN_GPA']

grd['BCMP_TERM_GPA'] = grd['BCMP_GRADE_POINTS_IN_GPA'] / grd['BCMP_UNITS_IN_GPA']

grd['BCMP_CUM_GPA'] = grd['CUM_BCMP_GRADE_POINTS_IN_GPA'] / grd['CUM_BCMP_UNITS_IN_GPA']


# ##### Check for NaN values since division by 0 is possible:

# In[22]:


grd.isnull().sum()


# ###### Need to deal with NaN values - use imputation

# ##### impute missing Term and Cum GPA
# 
# In this instance it is reasonable to set the Term GPA and CumGPA to zero since NaNs result from Units in GPA and Cum Units in GPA being zero. This means either the student had no Units in GPA for a given term or the Cum Units in GPA was zero since the student failed to pass units in the initial term

# In[23]:


grd.fillna(0, inplace=True)


# ##### Example of students with CumGPA = 0 

# In[24]:


grd[grd['CUM_GPA'] == 0].head()


# In[25]:


emplid = '011375323'

grd[grd['EMPLID'] == emplid]


# ##### Typically students with CumGPA = 0 fail or withdraw

# In[26]:


grd[grd['EMPLID'] == emplid].iloc[0]


# # The Load Index

# $L_s = \dfrac{\sum_{i=1}^{k_s} u_i}{U_s}$
# 
# $u_i: units\ \ earned \ \ by \ \ taking \ \ class \ \ i.$
# 
# $k_s: number \ \ of \ \ classes \ \ taken \ \ in \ \ semester \ \ s.$
# 
# $U_s: number \ \ of \ \ units \ \ prescribed \ \ to \ \ be \ \ earned \ \ by \ \ semester \ \ s.$
# 
# $The \ \ pattern \ \  of \ \ the \ \ difference \ \  d \ \ 0,7,8,9,10,17,18,19,20,... \ \ $
# 
# $Pattern \ \ when \ \ d \ \ mod \ \ 10 \ \ is \ \ 0$    $$N = 2\dfrac{d}{10} + 1 = \dfrac{d}{5} + 1$$
# 
# Pattern when d mod 10 is 8    $$N = 2\left(\dfrac{d - 8}{10} + 1\right) = \dfrac{d + 2}{5}$$

# In[27]:


grd['TERM_DIFF'] = pd.to_numeric(grd['TERM_CODE']) - pd.to_numeric(grd['COHORT'])

grd['N'] = grd['TERM_DIFF'].apply(lambda x: int(x/5+1) if x%10 == 0 else int((x + 2)/5) )

grd['PRESCRIBED_UNITS'] = grd['N'] * 15


# In[28]:


grd.head()


# In[29]:


grd[['EMPLID','TERM_CODE','N']].head(19)


# In[30]:


grd['LOAD_INDEX'] = grd['CUM_UNITS_FOR_CREDIT'] / grd['PRESCRIBED_UNITS']

grd['COMPLETION_RATE'] = grd['UNITS_FOR_CREDIT'] / grd['UNITS_TAKEN']


# ## Create The Exclusive Load Index 

# In[31]:


grd['UNITS_FOR_CREDIT_EXCLUDE'] = (grd['TERM_DIFF'] >= 0) * grd['UNITS_FOR_CREDIT']

grouped_cumsum = grd[['EMPLID',
                      'TERM_CODE',
                      'COHORT',
                      'UNITS_FOR_CREDIT_EXCLUDE']].groupby(['EMPLID',
                                                                 'TERM_CODE',
                                                                 'COHORT']).sum().groupby(level=[0]).cumsum().reset_index()

grouped_cumsum = grouped_cumsum.add_prefix('CUM_')

grd = pd.concat([grd,grouped_cumsum],axis=1)

grd['N_EXCLUDE'] = grd['TERM_DIFF'].apply(lambda x: 0 if x < 0 else(int(x/5+1) if int(repr(x)[-1]) == 0 else int((x + 2)/5) ))

grd['PRESCRIBED_UNITS_EXCLUDE'] = grd['N_EXCLUDE'] * 15

grd['LOAD_INDEX_EXCLUDE'] = grd['CUM_UNITS_FOR_CREDIT_EXCLUDE'] / grd['PRESCRIBED_UNITS_EXCLUDE']


# In[32]:


grd.head()


# ## Create the Only Prior Load Index Indicator

# In[33]:


grd['LOAD_INDEX_ONLY']=grd['LOAD_INDEX']-grd['LOAD_INDEX_EXCLUDE']

grouped_diff = grd[['EMPLID', 'LOAD_INDEX_ONLY']].groupby(['EMPLID']).transform(max).reset_index()

grd = grd.drop(columns='LOAD_INDEX_ONLY')

grd = pd.concat([grd,grouped_diff],axis=1)


# ## Create DFW Variables

# In[34]:


grd['DFW'] = grd['D'] + grd['F'] + grd['I'] + grd['NC'] + grd['W'] + grd['WE'] + grd['WU']

grd['DFW_RATE'] = grd['DFW']/grd['UNITS_TAKEN']


# ## Select Varaibles

# In[35]:


grd.drop(labels=['GRADE_POINTS_IN_GPA','UNITS_IN_GPA','BCMP_GRADE_POINTS_IN_GPA',
                'BCMP_UNITS_IN_GPA','CUM_GRADE_POINTS_IN_GPA', 
                'CUM_UNITS_IN_GPA', 'CUM_BCMP_GRADE_POINTS_IN_GPA', 'TERM_DIFF', 
                 'CUM_EMPLID', 'CUM_TERM_CODE', 'CUM_COHORT',
                 'CUM_UNITS_FOR_CREDIT_EXCLUDE', 'N_EXCLUDE', 'PRESCRIBED_UNITS_EXCLUDE', 'index'],axis=1, inplace=True)


# ## Exclude Rows Before $T_0$

# In[36]:


grd = grd[grd['COHORT'] <= grd['TERM_CODE']]


# ##### The Completion Rate calculation generated a few NaNs
# 
# Setting these NaNs to zero is appropriate since they result from dividing by zero (no units taken)

# In[37]:


grd.fillna(0, inplace=True)


# ## Demographic Data

# In[38]:


dem.columns = map(str.upper, dem.columns)
dem.columns


# ### One-Hot-Encode Demographics

# In[39]:


dem = pd.concat([dem,
                pd.get_dummies(dem['GENDER'], drop_first=True, prefix='GENDR'),
                pd.get_dummies(dem['ETHNICITY'], drop_first=False),
                pd.get_dummies(dem['FIRST_GENERATION'], drop_first=False),
                pd.get_dummies(dem['DEP_FAMILY_SIZE'], drop_first=False, prefix='DEP_FAM'),
                pd.get_dummies(dem['MINORITY'], drop_first=False, prefix='URM'), 
                pd.get_dummies(dem['APPLICANT_FAMILY_SIZE'], drop_first=False, prefix='APP_FAM'),
                pd.get_dummies(dem['APPLICANT_INCOME'], drop_first=False, prefix='INCM'),
                pd.get_dummies(dem['PELL_ELIGIBILITY'], drop_first=False, prefix='PELL')], axis=1)


# In[40]:


dem.columns = map(str.upper, dem.columns)
dem.columns


# In[41]:


dem.drop(labels=['GENDER', 'ETHNICITY', 'FIRST_GENERATION',
       'DEP_FAMILY_SIZE', 'MINORITY', 'APPLICANT_FAMILY_SIZE',
       'APPLICANT_INCOME', 'PELL_ELIGIBILITY'], axis=1, inplace=True)


# ### Create Time to Graduation Response Variables

# In[42]:


dem['DEM_N'] = dem['DEM_DIFF_INDX'].apply(lambda x: x if (x >= 0) == False 
                             else int(x/5+1) if (x%10 == 0 or x%10 == 7) 
                             else int((x + 2)/5) )

dem['YRS_TO_GRAD'] = dem['DEM_N'] * 0.5


# In[43]:


dem[['DEM_DIFF_INDX','YRS_TO_GRAD']].head(12)


# In[44]:


dem.columns


# In[45]:


dem = pd.concat([dem,pd.get_dummies(dem['YRS_TO_GRAD'], drop_first=False, prefix='GRAD_IN')], axis=1)


# In[46]:


dem.drop(labels=['DEM_DIFF_INDX','DEM_N','DAE_EMPLID','PELLTOT_EMPLID','ESA_EMPLID',
                 'YRS_TO_GRAD'], axis=1, inplace=True)


# In[47]:


dem.columns


# In[48]:


dem.columns


# ### Join the Demographic Data with the CSULB Academic Performance Data

# In[49]:


supreme = pd.merge(dem, grd, on='EMPLID', how='left')


# In[50]:


#pandas_profiling.ProfileReport(supreme)


# In[51]:


#pandas_profiling.ProfileReport(supreme).get_rejected_variables()


# In[52]:


supreme.drop(labels=[ 'DFW',
                      'DEM_COHORT',
                      'AP',
                      'BCMP_UNITS_FOR_CREDIT',
                      'BCMP_UNITS_TAKEN',
                      'CUM_BCMP_UNITS_FOR_CREDIT',
                      'CUM_BCMP_UNITS_IN_GPA',
                      'CUM_BCMP_UNITS_TAKEN',
                      'CUM_UNITS_FOR_CREDIT',
                      'INCM_NO RESPONSE',
                      'UNITS_FOR_CREDIT',
                      'URM_UNKNOWN',
                      'URM_VISA NON U.S.'], axis=1, inplace=True)


# In[53]:


pd.options.display.max_seq_items = supreme.columns.shape[0]

supreme.columns


# ### ACT, SAT and HS GPA Scores Preprocessing
# 
# Students may have ACT or SAT or both scores. The idea is to create a feature that would capture test performance in a general sense. The approach use here is to create three features that capture performance in Math, Reading and Composite performance. To this end we scale and center both ACT and SAT test scores in math, reading and composite and, in the event a student has taken both,choose the maximum normalized score.
# 
# ##### Need to impute missing values of ACT and SAT scores:
# 
# Before this preprocessing step is undertaken it is necessary to split the student data into trainning and development sets in order to avoid "leakeage" from trainning into development since the imputing calculations and methods use the entire dataset.

# In[54]:


supreme['N'].head()


# In[55]:


supreme_train = pd.merge(students_train, supreme, on='EMPLID', how='inner')
print(supreme_train.shape)

supreme_dev = pd.merge(students_dev, supreme, on='EMPLID', how='inner')
print(supreme_dev.shape)


# In[56]:


supreme_train.isnull().sum().head(55)


# In[57]:


supreme_train.iloc[:,44:57].columns


# In[58]:


supreme_train[['ACT_COMP', 'ACT_READ', 'ACT_MATH', 'ACT_ENG', 'ACT_SCI', 'SAT_READ',
       'SAT_MATH', 'SAT_COMP', 'GPA_HS']].head()


# In[59]:


supreme_train['N'].head()


# ##### Scale the scores and choose the max

# In[60]:


supreme_train[['ACT_COMP', 'ACT_READ', 'ACT_MATH', 
         'ACT_ENG', 'ACT_SCI', 'SAT_READ',
         'SAT_MATH', 'SAT_COMP']] = preprocessing.scale(supreme_train[['ACT_COMP', 'ACT_READ', 'ACT_MATH', 
                                                                         'ACT_ENG', 'ACT_SCI', 'SAT_READ', 
                                                                         'SAT_MATH', 'SAT_COMP']])


# In[61]:


supreme_dev[['ACT_COMP', 'ACT_READ', 'ACT_MATH', 
         'ACT_ENG', 'ACT_SCI', 'SAT_READ',
         'SAT_MATH', 'SAT_COMP']] = preprocessing.scale(supreme_dev[['ACT_COMP', 'ACT_READ', 'ACT_MATH', 
                                                                         'ACT_ENG', 'ACT_SCI', 'SAT_READ', 
                                                                         'SAT_MATH', 'SAT_COMP']])


# In[62]:


print(
    
    supreme_train[['ACT_COMP', 'ACT_READ', 'ACT_MATH', 'ACT_ENG', 'ACT_SCI', 'SAT_READ',
       'SAT_MATH', 'SAT_COMP']].mean()
)

print(
    
    supreme_train[['ACT_COMP', 'ACT_READ', 'ACT_MATH', 'ACT_ENG', 'ACT_SCI', 'SAT_READ',
       'SAT_MATH', 'SAT_COMP']].std()
)


# In[63]:


supreme_train['T_COMP'] = supreme_train[['ACT_COMP','SAT_COMP']].apply(lambda x: x.max(), axis=1)
supreme_train['T_READ'] = supreme_train[['ACT_READ','SAT_READ']].apply(lambda x: x.max(), axis=1)
supreme_train['T_MATH'] = supreme_train[['ACT_MATH','SAT_MATH']].apply(lambda x: x.max(), axis=1)


# In[64]:


supreme_dev['T_COMP'] = supreme_dev[['ACT_COMP','SAT_COMP']].apply(lambda x: x.max(), axis=1)
supreme_dev['T_READ'] = supreme_dev[['ACT_READ','SAT_READ']].apply(lambda x: x.max(), axis=1)
supreme_dev['T_MATH'] = supreme_dev[['ACT_MATH','SAT_MATH']].apply(lambda x: x.max(), axis=1)


# In[65]:


supreme_train[['T_COMP','T_READ','T_MATH']].isnull().sum()


# ##### For now impute values by using the mean

# In[66]:


imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')


# In[67]:


test_scores = supreme_train[['T_COMP','T_READ','T_MATH','GPA_HS']]

imp_mean.fit(test_scores)  

supreme_train[['T_COMP','T_READ','T_MATH','GPA_HS']] = imp_mean.transform(test_scores)


# In[68]:


test_scores = supreme_dev[['T_COMP','T_READ','T_MATH','GPA_HS']]

imp_mean.fit(test_scores)  

supreme_dev[['T_COMP','T_READ','T_MATH','GPA_HS']] = imp_mean.transform(test_scores)


# ##### Drop unecessary features

# In[69]:


supreme_train['N'].head()


# In[70]:


supreme_train.columns[30:50]


# In[71]:


supreme_train.drop(['ACT_COMP', 'ACT_READ', 'ACT_MATH', 'ACT_ENG', 'ACT_SCI', 'SAT_READ',
       'SAT_MATH', 'SAT_COMP'], axis=1, inplace=True)

supreme_dev.drop(['ACT_COMP', 'ACT_READ', 'ACT_MATH', 'ACT_ENG', 'ACT_SCI', 'SAT_READ',
       'SAT_MATH', 'SAT_COMP'], axis=1, inplace=True)


# ##### The AP feature is null - impute with a constant for now

# In[72]:


supreme_train['AP'] = 0

supreme_dev['AP'] = 0


# In[73]:


supreme_train.isnull().sum().head(55)


# ### Create the Major Plan Change Index

# ##### Academic plans are 10 characters in length

# In[74]:


supreme_train['EOT_ACAD_PLAN_CD'].apply(lambda x: len(x)).describe()


# ##### The first 4 characters representthe department/subject. The 5th and 6th character encode the plan type and allows to separate pre-majors from majors. A change from pre-major to major in the same department/subject is not considered a major change.

# In[75]:


print (
    supreme_train['EOT_ACAD_PLAN_CD'].apply(lambda x: x[4:6]).head()
)

print (
    supreme_train['EOT_ACAD_PLAN_CD'].apply(lambda x: x[:4]).head()
)


# ##### Create codes

# In[76]:


supreme_train['SUBJECT'] = supreme_train['EOT_ACAD_PLAN_CD'].apply(lambda x: x[:4])
supreme_train['PLAN_TYPE'] = supreme_train['EOT_ACAD_PLAN_CD'].apply(lambda x: x[4:6])


# In[77]:


supreme_dev['SUBJECT'] = supreme_dev['EOT_ACAD_PLAN_CD'].apply(lambda x: x[:4])
supreme_dev['PLAN_TYPE'] = supreme_dev['EOT_ACAD_PLAN_CD'].apply(lambda x: x[4:6])


# In[78]:


supreme_train[['EMPLID','SUBJECT','PLAN_TYPE']].head()


# ##### Create Major Change indicator

# In[79]:


supreme_train['MAJOR_CHANGE_INDICATOR'] = ( supreme_train[['EMPLID','SUBJECT','PLAN_TYPE']] == supreme_train[['EMPLID','SUBJECT','PLAN_TYPE']].shift() ).apply(
    
    lambda x: 0 if x[0] == False 
           
           or ( x[0] == True and x[1] == True and x[2] == True ) 
           
           or ( x[0] == True and x[1] == True and x[2] == False )
           
           else 1 if ( x[0] == True and x[1] == False and x[2] == True )
           
           or ( x[0] == True and x[1] == False and x[2] == False )
           
           else None,
           
           axis=1)


# In[80]:


supreme_dev['MAJOR_CHANGE_INDICATOR'] = ( supreme_dev[['EMPLID','SUBJECT','PLAN_TYPE']] == supreme_dev[['EMPLID','SUBJECT','PLAN_TYPE']].shift() ).apply(
    
    lambda x: 0 if x[0] == False 
           
           or ( x[0] == True and x[1] == True and x[2] == True ) 
           
           or ( x[0] == True and x[1] == True and x[2] == False )
           
           else 1 if ( x[0] == True and x[1] == False and x[2] == True )
           
           or ( x[0] == True and x[1] == False and x[2] == False )
           
           else None,
           
           axis=1)


# 
# ##### Create a cumulative Major Change Counter

# In[81]:


grouped_cumsum = supreme_train[['EMPLID',
                             'TERM_CODE',
                             'EOT_ACAD_PLAN_CD',
                             'COHORT',
                             'MAJOR_CHANGE_INDICATOR'
                             ]].groupby(['EMPLID',
                                                 'TERM_CODE',
                                                 'EOT_ACAD_PLAN_CD',
                                                 'COHORT',
                                                 ]).sum().groupby(level=[0]).cumsum().reset_index()

grouped_cumsum = grouped_cumsum.add_prefix('CUM_')

supreme_train = pd.concat([supreme_train,grouped_cumsum['CUM_MAJOR_CHANGE_INDICATOR']],axis=1)

supreme_train.rename(columns={'CUM_MAJOR_CHANGE_INDICATOR':'MAJOR_CHANGE_CNT','N':'SEMESTER_INDEX'}, inplace=True)


# In[82]:


grouped_cumsum = supreme_dev[['EMPLID',
                             'TERM_CODE',
                             'EOT_ACAD_PLAN_CD',
                             'COHORT',
                             'MAJOR_CHANGE_INDICATOR'
                             ]].groupby(['EMPLID',
                                                 'TERM_CODE',
                                                 'EOT_ACAD_PLAN_CD',
                                                 'COHORT',
                                                 ]).sum().groupby(level=[0]).cumsum().reset_index()

grouped_cumsum = grouped_cumsum.add_prefix('CUM_')

supreme_dev = pd.concat([supreme_dev,grouped_cumsum['CUM_MAJOR_CHANGE_INDICATOR']],axis=1)

supreme_dev.rename(columns={'CUM_MAJOR_CHANGE_INDICATOR':'MAJOR_CHANGE_CNT','N':'SEMESTER_INDEX'}, inplace=True)


# ##### Rearrange features

# In[83]:


supreme_train.columns


# In[84]:


id_var = ['COHORT', 'EMPLID']

perf_var = ['TERM_CODE', 'SEMESTER_INDEX', 'EOT_ACAD_PLAN_CD', 'MAJOR_CHANGE_INDICATOR','MAJOR_CHANGE_CNT',
            'UNITS_TAKEN','BCMP', 'A', 'AU', 'B' ,'C','CR', 'D', 'F', 'I', 'NC', 'RP', 'W', 'WE', 'WU', 'SUMMER', 
            'CUM_BCMP', 'CUM_SUMMER', 'CUM_WINTER', 'TERM_GPA','CUM_GPA', 'BCMP_TERM_GPA', 'BCMP_CUM_GPA',
            'COMPLETION_RATE', 'LOAD_INDEX_EXCLUDE', 'LOAD_INDEX_ONLY','DFW_RATE','T_COMP', 'T_READ','T_MATH']

dem_var = ['GPA_HS', 'GENDR_M', 'AFRICAN AMERICAN','ASIAN AMERICAN', 'CAUCASIAN', 'LATINO/LATINA', 
           'NATIVE AMERICAN','PACIFIC ISLANDER', 'TWO OR MORE RACES, INCLUDING MINORITY',
           'TWO OR MORE RACES, NON-MINORITIES', 'ETHNICITY UNKNOWN', 'VISA NON U.S.','CONTINUING GENERATION STUDENT', 
           'FIRST GENERATION STUDENT', 'FIRST GENERATION UNKNOWN','DEP_FAM_1', 'DEP_FAM_10', 'DEP_FAM_12', 'DEP_FAM_15', 'DEP_FAM_2',
           'DEP_FAM_25', 'DEP_FAM_3', 'DEP_FAM_4', 'DEP_FAM_5', 'DEP_FAM_6','DEP_FAM_7', 'DEP_FAM_8', 'DEP_FAM_9', 
           'DEP_FAM_NA', 'URM_MINORITY','URM_NON-MINORITY', 'APP_FAM_1',
           'APP_FAM_2', 'APP_FAM_3', 'APP_FAM_6', 'APP_FAM_NA','INCM_$12,000 TO $23,999', 'INCM_$6,000 TO $11,999',
           'INCM_$60,000 OR MORE', 'INCM_LESS THEN $6000', 'PELL_NON TRADITIONAL', 
           'PELL_TRADITIONAL']

resp_var = ['GRAD_IN_2.5','GRAD_IN_3.0', 'GRAD_IN_3.5', 'GRAD_IN_4.0', 'GRAD_IN_4.5','GRAD_IN_5.0', 'GRAD_IN_5.5', 
          'GRAD_IN_6.0']


# eyError: "[, 'CUM_UNITS_FOR_CREDIT'] not in index"

# In[ ]:





# In[85]:


supreme_train = supreme_train[id_var + perf_var + dem_var + resp_var]
supreme_dev = supreme_dev[id_var + perf_var + dem_var + resp_var]


# In[86]:


for feature in supreme_train.columns: print(feature)


# In[87]:


supreme_train[['SUMMER','CUM_SUMMER','CUM_WINTER']].head(9)


# In[88]:


supreme_train[supreme_train['EMPLID'] == '011193375']


# ### Descripte Stats

# In[89]:


#pandas_profiling.ProfileReport(supreme_train)


# In[90]:


#pandas_profiling.ProfileReport(supreme_train).get_rejected_variables()


# In[ ]:





# In[91]:


#pandas_profiling.ProfileReport(supreme_dev)


# In[92]:


#pandas_profiling.ProfileReport(supreme_dev).get_rejected_variables()


# In[93]:


# import seaborn as sns
# sns.set()

# # Load the example iris dataset
# #planets = sns.load_dataset("planets")

# cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
# ax = sns.scatterplot(x="COMPLETION_RATE", y="LOAD_INDEX_EXCLUDE",
#                      hue="TERM_GPA", size="CUM_UNITS_TAKEN",
#                      palette=cmap, sizes=(10, 200),
#                      data=supreme)

# # import seaborn as sns
# # sns.set()

# # # Load the example iris dataset
# # planets = sns.load_dataset("planets")

# # cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
# # ax = sns.scatterplot(x="distance", y="orbital_period",
# #                      hue="year", size="mass",
# #                      palette=cmap, sizes=(10, 200),
# #                      data=planets)


# In[94]:


# sns.set(style="white", palette="muted", color_codes=True)

# # Set up the matplotlib figure
# f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=False)
# sns.despine(left=True)

# # Plot a simple histogram with binsize determined automatically
# sns.distplot(supreme_train['LOAD_INDEX'], kde=False, color="b", ax=axes[0,0])

# sns.distplot(supreme_train['LOAD_INDEX_EXCLUDE'], kde=False, color="g", ax=axes[1,0])

# sns.distplot(supreme_train['GPA_HS'], kde=False, color="r", ax=axes[0,1])

# sns.distplot(supreme_train['CUM_GPA'], kde=False, color="m", ax=axes[1,1])


# In[95]:


supreme_train['GPA_HS'].describe()


# In[96]:


pd.to_numeric(supreme['TERM_CODE']).describe()


# In[ ]:





# In[97]:


supreme.head()


# ### Define Sequential Data Chunks

# In[98]:


supreme_train['TERM_CODE'] = pd.to_numeric(supreme_train['TERM_CODE'])
supreme_dev['TERM_CODE'] = pd.to_numeric(supreme_dev['TERM_CODE'])


# In[99]:


supreme_train[supreme_train['EMPLID'] == '010841881'][['EMPLID', 'SEMESTER_INDEX', 'TERM_GPA']].head()


# In[100]:


supreme_train['SEMESTER_INDEX'].unique()


# In[101]:


seq = {'TRAIN0':supreme_train[id_var + dem_var + resp_var]}
seq['DEV0'] = supreme_dev[id_var + dem_var + resp_var]


# In[102]:


for s in supreme_train['SEMESTER_INDEX'].unique():
    seq['TRAIN{}'.format(s)] = supreme_train[supreme_train['SEMESTER_INDEX'] <= s]
    seq['DEV{}'.format(s)] = supreme_dev[supreme_dev['SEMESTER_INDEX'] <= s]


# In[103]:


seq['TRAIN0'].head()


# In[104]:


print(seq['TRAIN0']['GRAD_IN_4.0'].mean())


# In[105]:


# supreme_dev.to_csv('supreme_dev.csv')

# supreme_train.to_csv('supreme_train.csv')


# # Modelling

# In[106]:


def model_tuner(X_train,y_train,X_dev,y_dev,model,grid = None):
    
    if grid == None:
        clf = model
    else:
        clf = GridSearchCV(model, grid, cv=10, n_jobs = -1)
    
    clf_fit = clf.fit(X_train,y_train)
    
    if grid != None: 
        best_par = clf.best_params_
    
    y_dev_pred = clf.predict(X_dev)
    y_train_pred = clf.predict(X_train)
    p_pred = clf.predict(X_dev)
    cm = confusion_matrix(y_dev,y_dev_pred)
    dev_accuracy = accuracy_score(y_dev,y_dev_pred)
    train_accuracy = accuracy_score(y_train,y_train_pred)
    report = classification_report(y_dev,y_dev_pred)
    
    if grid != None: 
        print ('\nthe optimal parameters are: {}'.format(best_par))
    
    print ('\naccuracy on the dev set is: {}'.format(dev_accuracy))
    print ('\naccuracy on the train set is: {}'.format(train_accuracy))
    print ('\nconfusion matrix:\n\n{}'.format(cm))
    print ('\nclassification report:\n\n{}'.format(report))
    
    if grid != None:
        results_dict = {'best model':clf_fit,'best parameters':best_par, 'predicted dev values':y_dev_pred, 
                        'predicted training values':y_train_pred,'predicted probabilities':p_pred,
                        'confusion matrix':cm,'dev accuracy':dev_accuracy,
                        'training accuracy':train_accuracy,'classification report':report}
    else:
        results_dict = {'best model':clf_fit, 'predicted dev values':y_dev_pred, 'predicted training values':y_train_pred, 
                        'predicted probabilities':p_pred,'confusion matrix':cm,'dev accuracy':dev_accuracy,
                        'training accuracy':train_accuracy,'classification report':report}
    return results_dict


# In[107]:


var = [
    
       'MAJOR_CHANGE_CNT', 'SUMMER','TERM_GPA','BCMP_TERM_GPA','COMPLETION_RATE', 'LOAD_INDEX_EXCLUDE', 
       'LOAD_INDEX_ONLY', 'DFW_RATE', 'GPA_HS','GENDR_M', 'AFRICAN AMERICAN',
       'ASIAN AMERICAN', 'CAUCASIAN', 'LATINO/LATINA', 'NATIVE AMERICAN', 'PACIFIC ISLANDER', 
       'TWO OR MORE RACES, INCLUDING MINORITY',
       'TWO OR MORE RACES, NON-MINORITIES', 'ETHNICITY UNKNOWN', 'VISA NON U.S.',
       'CONTINUING GENERATION STUDENT', 'FIRST GENERATION STUDENT', 'FIRST GENERATION UNKNOWN',
       'DEP_FAM_1', 'DEP_FAM_10', 'DEP_FAM_12', 'DEP_FAM_15', 'DEP_FAM_2',
       'DEP_FAM_25', 'DEP_FAM_3', 'DEP_FAM_4', 'DEP_FAM_5', 'DEP_FAM_6',
       'DEP_FAM_7', 'DEP_FAM_8', 'DEP_FAM_9', 'DEP_FAM_NA', 'URM_MINORITY',
       'URM_NON-MINORITY', 'APP_FAM_1',
       'APP_FAM_2', 'APP_FAM_3', 'APP_FAM_6', 'APP_FAM_NA',
       'INCM_$12,000 TO $23,999', 'INCM_$6,000 TO $11,999',
       'INCM_$60,000 OR MORE', 'INCM_LESS THEN $6000', 
       'PELL_NON TRADITIONAL', 'PELL_TRADITIONAL', 'T_COMP', 'T_READ',
       'T_MATH'
    
]


var2 = [
    
       'MAJOR_CHANGE_CNT', 'A', 'AU', 'B',
       'C', 'CR', 'D', 'F', 'I', 'NC', 'RP', 'W', 'WE', 'WU', 'SUMMER',
       'WINTER', 'CUM_SUMMER', 'CUM_WINTER', 'CUM_GPA', 'BCMP_CUM_GPA', 'COMPLETION_RATE', 
       'LOAD_INDEX_EXCLUDE', 'LOAD_INDEX_ONLY', 'DFW_RATE', 'GPA_HS'
    
]

var3 = [
    
       'MAJOR_CHANGE_CNT', 'SUMMER',
       'WINTER', 'CUM_SUMMER', 'CUM_WINTER', 'CUM_GPA', 'BCMP_CUM_GPA', 
       'LOAD_INDEX_EXCLUDE', 'LOAD_INDEX_ONLY', 'DFW_RATE', 'GPA_HS'
    
]


# In[108]:


def seq_data(t, time_to_grad, features):
    train_t = 'TRAIN{}'.format(str(t))
    dev_t = 'DEV{}'.format(str(t))
    response = 'GRAD_IN_{}.0'.format(time_to_grad)

    y_train = seq[train_t][response]
    y_dev = seq[dev_t][response]
    X_train = seq[train_t][features]
    X_dev = seq[dev_t][features]

    X_train, y_train = SMOTE().fit_sample(X_train, y_train)
    X_dev, y_dev = SMOTE().fit_sample(X_dev, y_dev)
    
    return (X_train,y_train,X_dev,y_dev)


# In[109]:


random_state = 42


# In[110]:


accuracy_matrix = pd.DataFrame()


# ##### Logistic at $t_0$

# In[111]:


X_train,y_train,X_dev,y_dev = seq_data(t=0, time_to_grad=4, features=dem_var)


# In[112]:


log_mod = LogisticRegression(solver='lbfgs', max_iter=10000, random_state = random_state)


# In[113]:


def run_model(model,grid = None,label=None):
    start = time.time()
    results_log = model_tuner(X_train,y_train,X_dev,y_dev,model,grid)
    end = time.time()
    runtime = end - start
    print (runtime/60)
    
    return {'model':label,'dev accuracy':[results_log['dev accuracy']], 'training accuracy':[results_log['training accuracy']]}
    
t0_log_accuracy = run_model(model=log_mod,label='logistic @ t0')


# In[114]:


accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t0_log_accuracy), sort=False)


# In[115]:


accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Logistic at $t_1$

# In[116]:


X_train,y_train,X_dev,y_dev = seq_data(t=1, time_to_grad=4, features=var)

t1_log_accuracy = run_model(model=log_mod,label='logistic @ t1')


# In[117]:


accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t1_log_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Logistic at $t_2$

# In[118]:


X_train,y_train,X_dev,y_dev = seq_data(t=2, time_to_grad=4, features=var)
t2_log_accuracy = run_model(model=log_mod,label='logistic @ t2')


# In[119]:


accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t2_log_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Logistic at $t_3$

# In[120]:


X_train,y_train,X_dev,y_dev = seq_data(t=3, time_to_grad=4, features=var)
t3_log_accuracy = run_model(model=log_mod,label='logistic @ t3')


# In[121]:


accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t3_log_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Logistic at $t_4$

# In[122]:


X_train,y_train,X_dev,y_dev = seq_data(t=4, time_to_grad=4, features=var)
t4_log_accuracy = run_model(model=log_mod,label='logistic @ t4')


# In[123]:


accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t4_log_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# In[124]:


ax = plt.gca()

accuracy_matrix.plot(kind='line',x='model',y='dev accuracy',ax=ax)
accuracy_matrix.plot(kind='line',x='model',y='training accuracy', color='red', ax=ax)

plt.show()


# ### Decision Trees

# ##### Decision Tree @ $t_0$

# In[125]:


X_train,y_train,X_dev,y_dev = seq_data(t=0, time_to_grad=4, features=dem_var)


# In[126]:


parameters_dt = {'max_features':list(range(3,23,6)), 'max_depth':list(np.arange(3,9,2))}

dt_mod = DecisionTreeClassifier(random_state = random_state)


# In[127]:


t0_dt_accuracy = run_model(model=dt_mod,grid=parameters_dt,label='decision tree @ t0')


# In[128]:


accuracy_matrix = pd.DataFrame()
accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t0_dt_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Decision Tree @ $t_1$

# In[129]:


X_train,y_train,X_dev,y_dev = seq_data(t=1, time_to_grad=4, features=var)


# In[130]:


t1_dt_accuracy = run_model(model=dt_mod,grid=parameters_dt,label='decision tree @ t1')


# In[131]:


accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t1_dt_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Decision Tree @ $t_2$

# In[132]:


X_train,y_train,X_dev,y_dev = seq_data(t=2, time_to_grad=4, features=var)

t2_dt_accuracy = run_model(model=dt_mod,grid=parameters_dt,label='decision tree @ t2')

accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t2_dt_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Decision Tree @ $t_3$

# In[133]:


X_train,y_train,X_dev,y_dev = seq_data(t=3, time_to_grad=4, features=var)

t3_dt_accuracy = run_model(model=dt_mod,grid=parameters_dt,label='decision tree @ t3')

accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t3_dt_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Decision Tree @ $t_4$

# In[134]:


X_train,y_train,X_dev,y_dev = seq_data(t=4, time_to_grad=4, features=var)

t4_dt_accuracy = run_model(model=dt_mod,grid=parameters_dt,label='decision tree @ t4')

accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t4_dt_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# In[135]:


ax = plt.gca()

accuracy_matrix.plot(kind='line',x='model',y='dev accuracy',ax=ax)
accuracy_matrix.plot(kind='line',x='model',y='training accuracy', color='red', ax=ax)

plt.show()


# ### Gradient Tree Boosting with gridsearch

# ##### Gradient Tree @ $t_0$

# In[136]:


# X_train,y_train,X_dev,y_dev = seq_data(t=0, time_to_grad=4, features=dem_var)

# parameters_gtb = {'learning_rate':np.arange(0.00075,1.075,0.1075), 'n_estimators':np.arange(50,650,150),
#               'max_depth':np.arange(3,9,2)}

# gtb_mod = GradientBoostingClassifier(random_state = random_state)

# t0_gtb_accuracy = run_model(model=gtb_mod,grid=parameters_gtb,label='gradient tree boost @ t0')

# accuracy_matrix = pd.DataFrame()
# accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t0_gtb_accuracy), sort=False)
# accuracy_matrix.reset_index(inplace=True,drop=True)
# accuracy_matrix


# In[ ]:





# In[137]:


# X_train,y_train,X_dev,y_dev = seq_data(t=1, time_to_grad=4, features=var)

# t1_gtb_accuracy = run_model(model=gtb_mod,grid=parameters_gtb,label='gradient tree boost @ t1')

# accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t1_gtb_accuracy), sort=False)
# accuracy_matrix.reset_index(inplace=True,drop=True)
# accuracy_matrix


# In[ ]:





# In[138]:


# X_train,y_train,X_dev,y_dev = seq_data(t=2, time_to_grad=4, features=var)

# t2_gtb_accuracy = run_model(model=gtb_mod,grid=parameters_gtb,label='gradient tree boost @ t2')

# accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t2_gtb_accuracy), sort=False)
# accuracy_matrix.reset_index(inplace=True,drop=True)
# accuracy_matrix


# In[ ]:





# In[139]:


# X_train,y_train,X_dev,y_dev = seq_data(t=3, time_to_grad=4, features=var)

# t3_gtb_accuracy = run_model(model=gtb_mod,grid=parameters_gtb,label='gradient tree boost @ t3')

# accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t3_gtb_accuracy), sort=False)
# accuracy_matrix.reset_index(inplace=True,drop=True)
# accuracy_matrix


# In[ ]:





# In[140]:


# X_train,y_train,X_dev,y_dev = seq_data(t=4, time_to_grad=4, features=var)

# t4_gtb_accuracy = run_model(model=gtb_mod,grid=parameters_gtb,label='gradient tree boost @ t4')

# accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t4_gtb_accuracy), sort=False)
# accuracy_matrix.reset_index(inplace=True,drop=True)
# accuracy_matrix


# In[ ]:





# ### Random Forest 

# ##### Random Forest @ $t_0$

# In[141]:


X_train,y_train,X_dev,y_dev = seq_data(t=0, time_to_grad=4, features=dem_var)

parameters_rf = {'n_estimators':list(range(100,600,200)),'max_features':list(range(3,23,6))}
rf = RandomForestClassifier(random_state = random_state)

t0_rf_accuracy = run_model(model=rf,grid=parameters_rf,label='random forest @ t0')

accuracy_matrix = pd.DataFrame()
accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t0_rf_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Random Forest @ $t_1$

# In[142]:


X_train,y_train,X_dev,y_dev = seq_data(t=1, time_to_grad=4, features=var)

t1_rf_accuracy = run_model(model=rf,grid=parameters_rf, label='random forest @ t1')

accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t1_rf_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Random Forest @ $t_2$

# In[143]:


X_train,y_train,X_dev,y_dev = seq_data(t=2, time_to_grad=4, features=var)

t2_rf_accuracy = run_model(model=rf,grid=parameters_rf, label='random forest @ t2')

accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t2_rf_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# ##### Random Forest @ $t_3$

# In[144]:


X_train,y_train,X_dev,y_dev = seq_data(t=3, time_to_grad=4, features=var)

t3_rf_accuracy = run_model(model=rf,grid=parameters_rf, label='random forest @ t3')

accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t3_rf_accuracy), sort=False)
accuracy_matrix.reset_index(inplace=True,drop=True)
accuracy_matrix


# In[145]:


ax = plt.gca()

accuracy_matrix.plot(kind='line',x='model',y='dev accuracy',ax=ax)
accuracy_matrix.plot(kind='line',x='model',y='training accuracy', color='red', ax=ax)

plt.show()


# ### XGBoost

# ##### XGBoost @ $t_0$

# In[146]:


X_train,y_train,X_dev,y_dev = seq_data(t=0, time_to_grad=4, features=dem_var)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_dev, label=y_dev)

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
evallist = [(dtest, 'eval'), (dtrain, 'train')]


num_round = 10000
bst = xgb.train(param, dtrain, num_round, evallist)


# In[147]:


y_pred = bst.predict(dtest)
y_pred = (y_pred > 0.5)*1

print(y_train.mean())
print(confusion_matrix(y_dev, y_pred))
print(classification_report(y_dev, y_pred))


# ##### XGBoost @ $t_1$

# In[148]:


X_train,y_train,X_dev,y_dev = seq_data(t=1, time_to_grad=4, features=dem_var)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_dev, label=y_dev)

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
evallist = [(dtest, 'eval'), (dtrain, 'train')]


num_round = 10000
bst = xgb.train(param, dtrain, num_round, evallist)


# In[149]:


y_pred = bst.predict(dtest)
y_pred = (y_pred > 0.5)*1

print(y_train.mean())
print(confusion_matrix(y_dev, y_pred))
print(classification_report(y_dev, y_pred))


# In[ ]:


y_dev_pred = bst.predict(dtest)
y_dev_pred = (y_pred > 0.5)*1


# In[158]:


dev_accuracy = accuracy_score(y_dev,y_dev_pred)
train_accuracy = accuracy_score(y_train,y_train_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[150]:


# parameters_rf = {'n_estimators':list(range(100,600,200)),'max_features':list(range(3,23,6))}
# rf = RandomForestClassifier(random_state = random_state)

# t0_rf_accuracy = run_model(model=rf,grid=parameters_rf,label='random forest @ t0')

# accuracy_matrix = pd.DataFrame()
# accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t0_rf_accuracy), sort=False)
# accuracy_matrix.reset_index(inplace=True,drop=True)
# accuracy_matrix


# #### SGD Classifier (with Elastic Net)

# In[151]:


# X_train,y_train,X_dev,y_dev = seq_data(t=0, time_to_grad=4, features=dem_var)


# parameters_en = {'l1_ratio':np.arange(0.001,1,0.1),'alpha':np.logspace(-3,1)}
# sgd_en = SGDClassifier(loss = 'log', penalty = 'elasticnet', random_state = random_state, max_iter=1000, tol=1e-3)

# t0_sgd_en_accuracy = run_model(model=sgd_en,grid=parameters_en,label='elastic net @ t0')

# accuracy_matrix = pd.DataFrame()
# accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t0_sgd_en_accuracy), sort=False)
# accuracy_matrix.reset_index(inplace=True,drop=True)
# accuracy_matrix


# In[ ]:





# In[152]:


# X_train,y_train,X_dev,y_dev = seq_data(t=1, time_to_grad=4, features=var)


# t1_sgd_en_accuracy = run_model(model=sgd_en,grid=parameters_en,label='elastic net @ t1')

# accuracy_matrix = accuracy_matrix.append(pd.DataFrame(t1_sgd_en_accuracy), sort=False)
# accuracy_matrix.reset_index(inplace=True,drop=True)
# accuracy_matrix


# In[ ]:





# In[ ]:





# ### Elastic Net

# In[153]:


# en_mod = ElasticNetCV(l1_ratio=[.025,.05,.1,.5,.7,.9,.95,.975,.99,.995,.9925,1],eps=1e-3,normalize=False,cv=50,n_jobs=-1)

# # parameters_en = {'l1_ratio':np.arange(0.01,1,0.1),'alpha':np.logspace(-3,1)}
# # sgd_en = SGDClassifier(loss = 'log', penalty = 'elasticnet', random_state = random_state, max_iter=100)


# ##### ELastic Net at $t_0$

# In[154]:


# X_train,y_train,X_dev,y_dev = seq_data(t=0, time_to_grad=4, features=dem_var)
# # run_model(model=sgd_en, grid=parameters_en)


# In[155]:


# en_fit = en_mod.fit(X_train,y_train)


# In[156]:


# def en_run_model(X_train,y_train,X_dev,y_dev):
    
#     en_fit = en_mod.fit(X_train,y_train)
#     y_dev_pred = (en_fit.predict(X_dev) > .5)*1
#     y_train_pred = (en_fit.predict(X_train) > .5)*1
#     p_pred = en_fit.predict(X_dev)
#     cm = confusion_matrix(y_dev,y_dev_pred)
#     dev_accuracy = accuracy_score(y_dev,y_dev_pred)
#     train_accuracy = accuracy_score(y_train,y_train_pred)
#     report = classification_report(y_dev,y_dev_pred)

#     print ('\naccuracy on the dev set is: {}'.format(dev_accuracy))
#     print ('\naccuracy on the train set is: {}'.format(train_accuracy))
#     print ('\nconfusion matrix:\n\n{}'.format(cm))
#     print ('\nclassification report:\n\n{}'.format(report))
    
#     return (dev_accuracy, train_accuracy)


# In[157]:


en_run_model(X_train,y_train,X_dev,y_dev)


# In[ ]:





# In[ ]:





# In[ ]:





# ##### Elastic Net at $t_1$

# In[ ]:


X_train,y_train,X_dev,y_dev = seq_data(t=1, time_to_grad=4, features=var)


# In[ ]:


en_run_model(X_train,y_train,X_dev,y_dev)


# ##### Elastic Net at $t_2$

# In[ ]:


# X_train,y_train,X_dev,y_dev = seq_data(t=2, time_to_grad=4, features=var)


# In[ ]:


# en_run_model(X_train,y_train,X_dev,y_dev)


# ##### Elastic Net at $t_3$

# In[ ]:


# X_train,y_train,X_dev,y_dev = seq_data(t=3, time_to_grad=4, features=var)


# In[ ]:


# en_run_model(X_train,y_train,X_dev,y_dev)


# In[ ]:


# coef_df = pd.DataFrame(np.abs(en_mod.coef_))
# #coef_df.index = X_dev.columns
# coef_df.sort_values(by=0,ascending=False)


# In[ ]:


# supreme.shape


# In[ ]:


# var[5]


# In[ ]:


# en_mod.coef_[5]


# In[ ]:


# var


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





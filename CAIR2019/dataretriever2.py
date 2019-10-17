#!/usr/bin/env python
# coding: utf-8

# # Summary

# The object of this project is to explore the power of a predictive system based on the notion that at the end of each semester we can diagnose the chances of a student graduating in 4 years based on all available information up to that point. 
# 
# One main objective is to identify an information saturation point upon which actionable intervention is implemented. The hypothesis is that there exists a local maxima saturation point which provides the most appropriate intervention point.

# ### Imports

# In[1]:


#BASIC MODULES
from __future__ import print_function, division

import pandas as pd
from pandas import DataFrame, Series

import seaborn as sns

from builtins import range
import matplotlib.pyplot as plt

import numpy as np

import random
import string


# In[2]:


#XgBoost Model ###################################################################################
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
##################################################################################################


# In[3]:


#SciKitLearn Models
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#MODEL SELECTION
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_score, GridSearchCV
#EVALUATION METRICS
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#IMBALANCED DATA
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler


# In[4]:


# # TensorFlow
import tensorflow as tf


# ##### Custom IR&A Library to Feature Engineer FTF Data 

# In[5]:


import ftfdata as ftf


# In[6]:


#import importlib
#importlib.reload(ftf) 


# ### Data Generation

# In[7]:


dfGen = ftf.DataGenerator('japitz','IRATableau132','iraarch','data_query.txt')


# In[8]:


df = dfGen.genDataFrame()
df.head()


# In[9]:


df.iloc[:,30:].head()


# ##### Select Fields of Interest:

# In[10]:


df.columns


# In[11]:


fields = ['CLASS_TERM',
          'EMPLID',
          'ERSS_COHORT_YEAR',
          'SAT_COMP_SCORE',
          'SAT_CRIT_READING_SCORE', 
          'SAT_MATH_SCORE', 
          'SAT_WRITING_SCORE',
          'ACT_COMPOSITE_SCORE', 
          'ACT_ENGLISH_SCORE',
          'ACT_MATHEMATICS_SCORE',
          'ACT_READING_SCORE',
          'ACT_SCIENCE_REASONING',
          'ACT_WRITING',
          'ENRL_GRADE_POINTS_IN_GPA',
          'ENRL_OFFICIAL_GRADE',
          'ENRL_UNITS_TAKEN',
          'ENRL_UNITS_IN_GPA',
          'ENRL_UNITS_FOR_CREDIT',
          'GRADUATE_WITHIN_4YR',
          'GRADUATE_WITHIN_6YR',
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
          'WU']


# ##### Define Aggregations to Create a cummulative sum of Grade Points  GPA Units and Other Variables:

# In[12]:


aggregations = { 'SAT_COMP_SCORE':'max',
                 'SAT_CRIT_READING_SCORE':'max', 
                 'SAT_MATH_SCORE':'max', 
                 'SAT_WRITING_SCORE':'max',
                 'ACT_COMPOSITE_SCORE':'max', 
                 'ACT_ENGLISH_SCORE':'max',
                 'ACT_MATHEMATICS_SCORE':'max',
                 'ACT_READING_SCORE':'max',
                 'ACT_SCIENCE_REASONING':'max',
                 'ACT_WRITING':'max',
                 'ENRL_GRADE_POINTS_IN_GPA':'sum',
                 'ENRL_OFFICIAL_GRADE':'sum',
                 'ENRL_UNITS_TAKEN':'sum',
                 'ENRL_UNITS_IN_GPA':'sum',
                 'ENRL_UNITS_FOR_CREDIT':'sum',
                 'GRADUATE_WITHIN_4YR':'max',
                 'GRADUATE_WITHIN_6YR':'max',
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
                 'WU':'sum'}


# ### Create Specific Dataset for Analysis

# In[13]:


dfTran = ftf.DataTrans(df,fields,aggregations)

df_student = dfTran.transformer()


# ### Inspect Correlations

# In[14]:


dfTran.corr_mat(df_student)


# ##### Example of student with high load index due to accumulating credits prior to begining of program

# ### To include:
#  T0: HS GPA and ACT/SAT score and ACT/SAT flag (standardize??? alternatively bin it???)
#  
#  T1: T0 + major at EOT1 + Biology Chemestry Mathematics and Physiscs (BCMP) and the Non-BCMP GPA (semester and cum) and all other variables included so far.
#  
#  T2: T1 + major at EOT2 + new performance variables updated to T2
#  
#  ### Simple Things to Do:
#  Example: MATH as major at T1
#           FIN as major at T2,
#           How many course taken with MATH prefix and FIN
#           If ma

# In[15]:


df_student[df_student['EMPLID'] == '008549240']


# ##### Obsevations with a load index > 2.9

# In[16]:


df_student[['EMPLID','GRADUATE_WITHIN_4YR', 'CLASS_TERM','ENRL_UNITS_FOR_CREDIT','COMPLETION_RATE','TERM_GPA','CUM_GPA','CUM_ENRL_UNITS_FOR_CREDIT', 'PRESCRIBED_UNITS','N','LOAD_INDEX']][df_student['LOAD_INDEX']>2.9].shape


# In[17]:


df_student.dtypes


# In[18]:


df_student['ACT_WRITING'] = df_student['ACT_WRITING'].astype('int64')


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


x = df_student['LOAD_INDEX'].replace(np.inf,np.nan).dropna()


# In[21]:


ax=sns.distplot(x)


# In[22]:


x.describe()


# ##### Create Pseudo IDs

# In[23]:


uemplids = Series(list(set(df_student['EMPLID'])),name='EMPLID')


# In[24]:


pseudoids = Series([''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(32)) for i in range(len(uemplids))],name='PSEUDOID')


# In[25]:


idmap = pd.concat([uemplids,pseudoids], axis=1)
idmap.head()


# In[26]:


df_anom = df_student.join(idmap.set_index('EMPLID'),on='EMPLID').drop('EMPLID',axis=1)
df_anom.head()


# In[27]:


#df_anom.to_csv('dataset_0.csv')


# ### Extract Data for Semester T1

# In[28]:


df_student_t1 = df_student[df_student['CLASS_TERM'] == '2112'].copy()
df_student_t1.head()


# In[29]:


df_student_t1.columns


# In[30]:


#df_student_t1[['EMPLID','A','B']]


# In[31]:


df_student_t1['GRADUATE_WITHIN_4YR'].mean()


# In[ ]:





# ### Extract Data for Semester T2 and T3

# In[32]:


df_student_t2 = df_student[df_student['CLASS_TERM'] == '2114'].copy()
print(df_student_t2['GRADUATE_WITHIN_4YR'].mean())

df_student_t3 = df_student[df_student['CLASS_TERM'] == '2122'].copy()
print(df_student_t3['GRADUATE_WITHIN_4YR'].mean())

df_student_t4 = df_student[df_student['CLASS_TERM'] == '2124'].copy()
print(df_student_t4['GRADUATE_WITHIN_4YR'].mean())


# In[33]:


df_student_t4.columns


# In[34]:


df_student_t4[df_student_t4['EMPLID']=='004926946']


# ### Further Prepare Data for Analysis in SciKitLearn

# In[35]:


data = df_student_t4.copy().replace([np.inf,-np.inf], np.nan).dropna()


# In[36]:


y = np.ravel(data[['GRADUATE_WITHIN_4YR']])
X = data[['SAT_COMP_SCORE',
          'SAT_CRIT_READING_SCORE', 
          'SAT_MATH_SCORE', 
          'SAT_WRITING_SCORE',
          'ACT_COMPOSITE_SCORE', 
          'ACT_ENGLISH_SCORE',
          'ACT_MATHEMATICS_SCORE',
          'ACT_READING_SCORE',
          'ACT_SCIENCE_REASONING',
          'ACT_WRITING',
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
          'CUM_ENRL_GRADE_POINTS_IN_GPA',
          'CUM_ENRL_UNITS_TAKEN', 'CUM_ENRL_UNITS_IN_GPA',
          'CUM_ENRL_UNITS_FOR_CREDIT', 
          'TERM_GPA', 
          'CUM_GPA', 
          'TERM_DIFF',
          'PRESCRIBED_UNITS', 
          'LOAD_INDEX', 
          'COMPLETION_RATE']]


# ### Logistic Regression (Quick Example)

# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

log_mod = LogisticRegression(random_state = 42)

print(type(X_train),type(X_test))


# In[38]:


y_train.mean()


# In[39]:


log_fit = log_mod.fit(X_train,y_train)

y_pred = log_mod.predict(X_test)


# In[40]:


# majority class
print(1-y_test.mean())
print(log_mod.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(log_mod.score(X_test, y_test) - 1 + y_test.mean())


# ##### SMOTE  - Resample Data to Eliminate Imbalance

# In[41]:


X_trainb, y_trainb = SMOTE().fit_sample(X_train, y_train)

# log_fit = log_mod.fit(X_train,y_train)

# y_pred = log_mod.predict(X_test)

print(type(X_train))


# In[42]:


# print(y_resampled.mean())
# print(log_mod.score(X_test, y_test))
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(log_mod.score(X_test, y_test) - 1 + y_test.mean())


# ##### ADASYN  - Resample Data to Eliminate Imbalance

# In[43]:


# X_resampled, y_resampled = ADASYN().fit_sample(X_train, y_train)

# log_fit = log_mod.fit(X_resampled,y_resampled)

# y_pred = log_mod.predict(X_test)


# In[44]:


# print(y_resampled.mean())
# print(log_mod.score(X_test, y_test))
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(log_mod.score(X_test, y_test) - 1 + y_test.mean())


# ##### Random Oversample  - Resample Data to Eliminate Imbalance

# In[45]:


# X_resampled, y_resampled = RandomOverSampler().fit_sample(X_train, y_train)

# log_fit = log_mod.fit(X_resampled,y_resampled)

# y_pred = log_mod.predict(X_test)


# In[46]:


# print(y_resampled.mean())
# print(log_mod.score(X_test, y_test))
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(log_mod.score(X_test, y_test) - 1 + y_test.mean())


# In[ ]:





# # Model Building

# ### Elastic Net

# In[47]:


en_mod = ElasticNetCV(l1_ratio=[.1,.5,.7,.9,.95,.975,.99,.995,1],eps=1e-3,normalize=True,cv=100,n_jobs=-1)


# In[48]:


en_fit = en_mod.fit(X_trainb,y_trainb)


# In[49]:


y_pred = (en_mod.predict(X_test) > 0.6)*1


# In[50]:


confusion = confusion_matrix(y_test, y_pred)
print(y_train.mean())
print(en_mod.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(en_mod.score(X_test, y_test) - 1 + y_test.mean())


# In[51]:


en_mod.alpha_


# In[52]:


en_mod.l1_ratio_


# In[53]:


en_mod.coef_[en_mod.coef_.argmax()]


# In[54]:


en_mod.coef_.max()


# In[55]:


# en_mod.coef_.sort()
# en_mod.coef_[-1]


# In[56]:


X.columns[en_mod.coef_.argmax()]


# In[57]:


en_mod.coef_.argmax()


# In[58]:


coef_df = pd.DataFrame(en_mod.coef_)
coef_df.index = X.columns
coef_df.sort_values(by=0,ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# #### Random Forest

# In[59]:


rf_mod = RandomForestClassifier(n_estimators=10)
rf_fit = rf_mod.fit(X_trainb,y_trainb)
y_pred = rf_mod.predict(X_test)


# #### Prediction

# In[60]:


y_pred = rf_mod.predict(X_test)


# #### Performance Evaluation

# In[61]:


confusion = confusion_matrix(y_test, y_pred)
print(y_train.mean())
print(rf_mod.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(rf_mod.score(X_test, y_test) - 1 + y_test.mean())


# In[62]:


print(cross_val_score(rf_mod, X_test, y_test, cv=10).mean())


# Positive and Negative Predictive Values

# In[63]:


TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]
FN = confusion[1][0]

print(TP/(TP + FP))
print(TN/(TN + FN))


# ##### XGBoost

# In[64]:


type(X_test)


# In[65]:


type(X_train)


# In[66]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# In[67]:


param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
evallist = [(dtest, 'eval'), (dtrain, 'train')]


# In[68]:


num_round = 10000
bst = xgb.train(param, dtrain, num_round, evallist)


# In[69]:


y_pred = bst.predict(dtest)
y_pred = (y_pred > 0.5)*1


# In[70]:


print(y_train.mean())
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ##### GradientBoosting

# In[71]:


gb_mod = GradientBoostingClassifier(n_estimators=100)
gb_fit = gb_mod.fit(X_trainb,y_trainb)
y_pred = gb_mod.predict(X_test)


# In[72]:


#print(y_resampled.mean())
print(gb_mod.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(gb_mod.score(X_test, y_test) - 1 + y_test.mean())


# In[ ]:





# ##### Naive Prediction

# In[73]:


y_pred = np.zeros(y_pred.shape)


# In[74]:


print(1 - y_test.mean())
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(- y_pred.mean() + y_test.mean())


# In[75]:


accuracy_score(y_test,y_pred)


# In[ ]:





# #### Model Tunning and Implementation

# In[76]:


df_student.head()


# In[78]:


df_student.to_csv('S:\\student_prediction\\sample_data_00.csv')


# In[77]:


pwd


# In[ ]:





# In[ ]:





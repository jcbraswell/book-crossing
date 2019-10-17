# General
from __future__ import print_function, division
import sys

# Database
import cx_Oracle
from sqlalchemy import create_engine
from getpass import getpass

# Tools
import pandas as pd
import string
from builtins import range
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

class DataGenerator(object):
    """Docstring"""
    def __init__(self, service_name='iraarch',host='ira-oradb01.its.csulb.edu',port='1521', grades_query='grd.sql', dem_query='dae.sql'):
        super(DataGenerator, self).__init__()
        self.service_name = service_name
        self.host = host
        self.port = port
        self.grades_query = grades_query
        self.dem_query = dem_query

    def genDataFrame(self):
        """Docstring"""

        username = input('Enter username: ')
        password = getpass(prompt='Enter password: ')

        # Function creates db conection and queries db to generate data objects
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

        # Generate the grades and demographics dataframes
        grades, engine = db_query(username, password,self.service_name, self.host, self.port, self.grades_query)
        grd = pd.read_sql(grades, engine)
        
        demo, engine = db_query(username, password, self.service_name, self.host,self.port, self.dem_query)
        dem = pd.read_sql(demo, engine)

        ##################################################################################
        #                                                                                #
        # ETL PROCESSING, TRANSFORMATIONS AND STUDENT SPLITS FOR TRAINING AND VALIDATION #
        #                                                                                #
        ##################################################################################

        #Change 'UNKNOWN' to more unique string to avoid having columns with same name after one-hot-encode
        dem['first_generation'] = dem['first_generation'].apply(lambda x: 'First Generation Unknown' if x == 'UNKNOWN' else x)
        dem['ethnicity'] = dem['ethnicity'].apply(lambda x: 'ETHNICITY UNKNOWN' if x == 'UNKNOWN' else x)

        # Create Training and Test/Validation Sets of Students at this stage to avoid leakeage
        students = pd.DataFrame(grd['emplid'].unique(), columns=['emplid'])

        # Create the Grades Trainning and Validation Student Set
        students_train, students_dev = train_test_split(students, test_size=0.10, random_state=42)
        students_train = pd.DataFrame(students_train)
        students_dev = pd.DataFrame(students_dev)
        students_train.columns = ['EMPLID']
        students_dev.columns = ['EMPLID']

        ### Preprocessing: One-Hot-Encode Letter Grades ###
        grd.columns = map(str.upper, grd.columns)
        grd = pd.concat([grd,pd.get_dummies(grd['OFFICIAL_GRADE'], drop_first=True)], axis=1)

        ### Create Variables to Calculate GPA ###
        grd['GRADE_POINTS_IN_GPA'] = grd['GRADE_POINTS'] * grd['OFFICIAL_GRADE'].apply(lambda x: None if x in ['AU','CR','NC','RD','RP','W','WE'] else 1)

        grd['UNITS_IN_GPA'] = grd['UNITS_TAKEN'] * grd['OFFICIAL_GRADE'].apply(lambda x: None if x in ['AU','CR','NC','RD','RP','W','WE'] else 1)
        
        grd['UNITS_FOR_CREDIT'] = grd['UNITS_TAKEN'] * grd['OFFICIAL_GRADE'].apply(lambda x: None if x in ['AU','NC','RD','RP','W','WE'] else 1)

        grd['BCMP_GRADE_POINTS_IN_GPA'] = grd['BCMP'] * grd['GRADE_POINTS'] * grd['OFFICIAL_GRADE'].apply(lambda x: None if x in ['AU','CR','NC','RD','RP','W','WE'] else 1)

        grd['BCMP_UNITS_IN_GPA'] = grd['BCMP_UNITS_TAKEN'] * grd['OFFICIAL_GRADE'].apply(lambda x: None if x in ['AU','CR','NC','RD','RP','W','WE'] else 1)

        grd['BCMP_UNITS_FOR_CREDIT'] = grd['BCMP'] * grd['UNITS_TAKEN'] * grd['OFFICIAL_GRADE'].apply(lambda x: None if x in ['AU','NC','RD','RP','W','WE'] else 1)

        grd['SUMMER'] = (grd['CLASS_TERM'].apply(lambda x: str(x)[-1]) == '3')* 1 * grd['UNITS_FOR_CREDIT']

        grd['WINTER'] = (grd['CLASS_TERM'].apply(lambda x: str(x)[-1]) == '1')* 1 * grd['UNITS_FOR_CREDIT']

        ### Reduce the dataframe to variables of current interest ###

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
        ### Aggregate and Reduce from Course Dimension to Term Dimension: Create a cummulative sum of Grade Points and GPA Units: ###
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

        ### Create Term and Cummulative GPA ###

        grd['TERM_GPA'] = grd['GRADE_POINTS_IN_GPA'] / grd['UNITS_IN_GPA']
        grd['CUM_GPA'] = grd['CUM_GRADE_POINTS_IN_GPA'] / grd['CUM_UNITS_IN_GPA']
        grd['BCMP_TERM_GPA'] = grd['BCMP_GRADE_POINTS_IN_GPA'] / grd['BCMP_UNITS_IN_GPA']
        grd['BCMP_CUM_GPA'] = grd['CUM_BCMP_GRADE_POINTS_IN_GPA'] / grd['CUM_BCMP_UNITS_IN_GPA']

        ### impute missing Term and Cum GPA ###

        grd.fillna(0, inplace=True)

        ### The Full Load Index ###
        
        grd['TERM_DIFF'] = pd.to_numeric(grd['TERM_CODE']) - pd.to_numeric(grd['COHORT'])
        grd['N'] = grd['TERM_DIFF'].apply(lambda x: int(x/5+1) if x%10 == 0 else int((x + 2)/5) )
        grd['PRESCRIBED_UNITS'] = grd['N'] * 15
        grd['LOAD_INDEX'] = grd['CUM_UNITS_FOR_CREDIT'] / grd['PRESCRIBED_UNITS']

        ### The Load In dex Exclusive of Pre-degree units
        
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

        ### Create the Load Index with only pre-degree units ###

        grd['LOAD_INDEX_ONLY']=grd['LOAD_INDEX']-grd['LOAD_INDEX_EXCLUDE']
        grouped_diff = grd[['EMPLID', 'LOAD_INDEX_ONLY']].groupby(['EMPLID']).transform(max).reset_index()
        grd = grd.drop(columns='LOAD_INDEX_ONLY')
        grd = pd.concat([grd,grouped_diff],axis=1)

        
        ### Completion Rate
        
        grd['COMPLETION_RATE'] = grd['UNITS_FOR_CREDIT'] / grd['UNITS_TAKEN']
        # Fill NAs - completion rate
        grd.fillna(0, inplace=True)
        

        ### DFW Variables

        grd['DFW'] = grd['D'] + grd['F'] + grd['I'] + grd['NC'] + grd['W'] + grd['WE'] + grd['WU']
        grd['DFW_RATE'] = grd['DFW']/grd['UNITS_TAKEN']

        # Drop un-needed variables and rows

        grd.drop(labels=['GRADE_POINTS_IN_GPA','UNITS_IN_GPA','BCMP_GRADE_POINTS_IN_GPA',
                  'BCMP_UNITS_IN_GPA','CUM_GRADE_POINTS_IN_GPA', 
                  'CUM_UNITS_IN_GPA', 'CUM_BCMP_GRADE_POINTS_IN_GPA', 'TERM_DIFF', 
                   'CUM_EMPLID', 'CUM_TERM_CODE', 'CUM_COHORT',
                   'CUM_UNITS_FOR_CREDIT_EXCLUDE', 'N_EXCLUDE', 'PRESCRIBED_UNITS_EXCLUDE', 'index'],axis=1, inplace=True)

        grd = grd[grd['COHORT'] <= grd['TERM_CODE']]


        ### Demographic Data ###

        dem.columns = map(str.upper, dem.columns)

        ### One-Hot-Encode Demographics ###

        dem = pd.concat([dem,
                            pd.get_dummies(dem['GENDER'], drop_first=True, prefix='GENDR'),
                            pd.get_dummies(dem['ETHNICITY'], drop_first=False),
                            pd.get_dummies(dem['FIRST_GENERATION'], drop_first=False),
                            pd.get_dummies(dem['DEP_FAMILY_SIZE'], drop_first=False, prefix='DEP_FAM'),
                            pd.get_dummies(dem['MINORITY'], drop_first=False, prefix='URM'),
                            pd.get_dummies(dem['APPLICANT_FAMILY_SIZE'], drop_first=False, prefix='APP_FAM'),
                            pd.get_dummies(dem['APPLICANT_INCOME'], drop_first=False, prefix='INCM'),
                            pd.get_dummies(dem['PELL_ELIGIBILITY'], drop_first=False, prefix='PELL')], axis=1)

        dem.columns = map(str.upper, dem.columns)

        dem.drop(labels=['GENDER', 'ETHNICITY', 'FIRST_GENERATION','DEP_FAMILY_SIZE', 'MINORITY', 'APPLICANT_FAMILY_SIZE',
                            'APPLICANT_INCOME', 'PELL_ELIGIBILITY'], axis=1, inplace=True)

        ### Create Time to Graduation Response Variables ###

        dem['DEM_N'] = dem['DEM_DIFF_INDX'].apply(lambda x: x if (x >= 0) == False else int(x/5+1) if (x%10 == 0 or x%10 == 7) else int((x + 2)/5) )
        dem['YRS_TO_GRAD'] = dem['DEM_N'] * 0.5

        dem = pd.concat([dem,pd.get_dummies(dem['YRS_TO_GRAD'], drop_first=False, prefix='GRAD_IN')], axis=1)

        dem.drop(labels=['DEM_DIFF_INDX','DEM_N','DAE_EMPLID','PELLTOT_EMPLID','ESA_EMPLID','YRS_TO_GRAD'], axis=1, inplace=True)

        ### Join the Demographic Data with the CSULB Academic Performance Data ###

        supreme = pd.merge(dem, grd, on='EMPLID', how='left')
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

        # Separate data into dev and training

        supreme_train = pd.merge(students_train, supreme, on='EMPLID', how='inner')
        supreme_dev = pd.merge(students_dev, supreme, on='EMPLID', how='inner')

        ### Normalize the Standardized Tests ###

        def norm_test(df):

            df[['ACT_COMP', 'ACT_READ', 'ACT_MATH', 'ACT_ENG', 'ACT_SCI', 'SAT_READ','SAT_MATH',
                         'SAT_COMP']] = preprocessing.scale(df[['ACT_COMP', 'ACT_READ', 'ACT_MATH', 'ACT_ENG', 'ACT_SCI', 'SAT_READ','SAT_MATH', 'SAT_COMP']])

            df['T_COMP'] = df[['ACT_COMP','SAT_COMP']].apply(lambda x: x.max(), axis=1)
            df['T_READ'] = df[['ACT_READ','SAT_READ']].apply(lambda x: x.max(), axis=1)
            df['T_MATH'] = df[['ACT_MATH','SAT_MATH']].apply(lambda x: x.max(), axis=1)

            ##### For now impute values by using the mean ###

            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

            test_scores = df[['T_COMP','T_READ','T_MATH','GPA_HS']]
            imp_mean.fit(test_scores)
            df[['T_COMP','T_READ','T_MATH','GPA_HS']] = imp_mean.transform(test_scores)

            ### Drop unecessary features ###

            df.drop(['ACT_COMP', 'ACT_READ', 'ACT_MATH', 'ACT_ENG', 'ACT_SCI', 'SAT_READ','SAT_MATH', 'SAT_COMP'], axis=1, inplace=True)

            return df

        supreme_train = norm_test(supreme_train)
        supreme_dev = norm_test(supreme_dev)


        ### The AP feature is null - impute with a constant for now ###

        supreme_train['AP'] = 0
        supreme_dev['AP'] = 0

        ### Create the Major Plan Change Index ###

        supreme_train['SUBJECT'] = supreme_train['EOT_ACAD_PLAN_CD'].apply(lambda x: x[:4])
        supreme_train['PLAN_TYPE'] = supreme_train['EOT_ACAD_PLAN_CD'].apply(lambda x: x[4:6])

        supreme_dev['SUBJECT'] = supreme_dev['EOT_ACAD_PLAN_CD'].apply(lambda x: x[:4])
        supreme_dev['PLAN_TYPE'] = supreme_dev['EOT_ACAD_PLAN_CD'].apply(lambda x: x[4:6])

        ##### Create Major Change indicator ###

        supreme_train['MAJOR_CHANGE_INDICATOR'] = ( supreme_train[['EMPLID',
                                                                              'SUBJECT',
                                                                              'PLAN_TYPE']] == supreme_train[['EMPLID',
                                                                                                                     'SUBJECT',
                                                                                                                     'PLAN_TYPE']].shift() ).apply(
                                                                                                                          lambda x: 0 if x[0] == False
                                                                                                                          or ( x[0] == True and x[1] == True and x[2] == True )
                                                                                                                          or ( x[0] == True and x[1] == True and x[2] == False )
                                                                                                                          else 1 if ( x[0] == True and x[1] == False and x[2] == True )
                                                                                                                          or ( x[0] == True and x[1] == False and x[2] == False )
                                                                                                                          else None,
                                                                                                                          axis=1)

        supreme_dev['MAJOR_CHANGE_INDICATOR'] = ( supreme_dev[['EMPLID',
                                                                         'SUBJECT',
                                                                         'PLAN_TYPE']] == supreme_dev[['EMPLID',
                                                                                                             'SUBJECT',
                                                                                                             'PLAN_TYPE']].shift() ).apply(
                                                                                                                   lambda x: 0 if x[0] == False
                                                                                                                   or ( x[0] == True and x[1] == True and x[2] == True )
                                                                                                                   or ( x[0] == True and x[1] == True and x[2] == False )
                                                                                                                   else 1 if ( x[0] == True and x[1] == False and x[2] == True )
                                                                                                                   or ( x[0] == True and x[1] == False and x[2] == False )
                                                                                                                   else None,
                                                                                                                   axis=1)

        ### Create a cumulative Major Change Counter ###

        grouped_cumsum = supreme_train[['EMPLID',
                                             'TERM_CODE',
                                             'EOT_ACAD_PLAN_CD',
                                             'COHORT',
                                             'MAJOR_CHANGE_INDICATOR']].groupby(['EMPLID',
                                                                                          'TERM_CODE',
                                                                                          'EOT_ACAD_PLAN_CD',
                                                                                          'COHORT',]).sum().groupby(level=[0]).cumsum().reset_index()

        grouped_cumsum = grouped_cumsum.add_prefix('CUM_')
        supreme_train = pd.concat([supreme_train,grouped_cumsum['CUM_MAJOR_CHANGE_INDICATOR']],axis=1)
        supreme_train.rename(columns={'CUM_MAJOR_CHANGE_INDICATOR':'MAJOR_CHANGE_CNT','N':'SEMESTER_INDEX'}, inplace=True)

        grouped_cumsum = supreme_dev[['EMPLID',
                                           'TERM_CODE',
                                           'EOT_ACAD_PLAN_CD',
                                           'COHORT',
                                           'MAJOR_CHANGE_INDICATOR']].groupby(['EMPLID',
                                                                                        'TERM_CODE',
                                                                                        'EOT_ACAD_PLAN_CD',
                                                                                        'COHORT',]).sum().groupby(level=[0]).cumsum().reset_index()
        
        grouped_cumsum = grouped_cumsum.add_prefix('CUM_')
        supreme_dev = pd.concat([supreme_dev,grouped_cumsum['CUM_MAJOR_CHANGE_INDICATOR']],axis=1)
        supreme_dev.rename(columns={'CUM_MAJOR_CHANGE_INDICATOR':'MAJOR_CHANGE_CNT','N':'SEMESTER_INDEX'}, inplace=True)

        datasets = {'train_set':supreme_train, 'dev_set':supreme_dev}

        return datasets

      

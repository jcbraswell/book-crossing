import cx_Oracle
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class DataGenerator(object):
    """docstring for DataGenerator"""
    def __init__(self, username, password, dbname, query):
        super(DataGenerator, self).__init__()
        self.username = username
        self.password = password
        self.dbname = dbname
        self.query = query
        
    def genDataFrame(self):
        connection = cx_Oracle.connect(self.username, self.password, self.dbname)
        cursor = connection.cursor()
        
        with open(self.query, 'r') as f:
            data=f.read()
            
        cursor.execute(data)
        students = cursor.fetchall()
        colum_names = [cursor.description[i][0] for i in range(len(cursor.description))]
        df = DataFrame(data = students, columns = colum_names)
        df.sort_values(by=['EMPLID','CLASS_TERM'], inplace=True)
        
        #One-Hot-Encode Grades
        df = pd.concat([df,pd.get_dummies(df['ENRL_OFFICIAL_GRADE'], drop_first=True)], axis=1)
        
        #Generate Variables to Calculate GPA
        df['ENRL_GRADE_POINTS_IN_GPA'] = df['ENRL_GRADE_POINTS'] * df['ENRL_OFFICIAL_GRADE'].apply(
            lambda x: None if x in ['AU','CR','NC','RD','RP','W','WE'] else 1)
        
        df['ENRL_UNITS_IN_GPA'] = df['ENRL_UNITS_TAKEN'] * df['ENRL_OFFICIAL_GRADE'].apply(
            lambda x: None if x in ['AU','CR','NC','RD','RP','W','WE'] else 1)
        
        df['ENRL_UNITS_FOR_CREDIT'] = df['ENRL_UNITS_TAKEN'] * df['ENRL_OFFICIAL_GRADE'].apply(
            lambda x: None if x in ['AU','NC','RD','RP','W','WE'] else 1)
        
        #Create Graduation Within 4 or 6 Years Indicator Functions
        s = df['GRADUATIONRATE_1YR'] + df['GRADUATIONRATE_2YR'] + df['GRADUATIONRATE_3YR'] + df['GRADUATIONRATE_4YR']
        v = df['GRADUATIONRATE_1YR'] + df['GRADUATIONRATE_2YR'] + df['GRADUATIONRATE_3YR'] + df['GRADUATIONRATE_4YR'] + df['GRADUATIONRATE_5YR'] + df['GRADUATIONRATE_6YR']
        df['GRADUATE_WITHIN_4YR'] = s.apply(lambda x: 0 if x == 0 else 1)
        df['GRADUATE_WITHIN_6YR'] = v.apply(lambda x: 0 if x == 0 else 1)

        return df
        
class DataTrans(object):
    """docstring for DataTrans"""
    def __init__(self, df, fields, aggregations):
        super(DataTrans, self).__init__()
        self.df = df
        self.fields = fields
        self.aggregations = aggregations
        
    def transformer(self):
        df_sub = self.df.copy()[self.fields]
        grouped_agg = df_sub.groupby(['EMPLID','CLASS_TERM','ERSS_COHORT_YEAR']).agg(self.
                                                                                     aggregations).reset_index()
        grouped_agg.drop('ENRL_OFFICIAL_GRADE', axis=1, inplace=True)
        grouped_cumsum = df_sub[['EMPLID',
                                     'CLASS_TERM',
                                     'ERSS_COHORT_YEAR',
                                     'ENRL_GRADE_POINTS_IN_GPA',
                                     'ENRL_OFFICIAL_GRADE',
                                     'ENRL_UNITS_TAKEN',
                                     'ENRL_UNITS_IN_GPA',
                                     'ENRL_UNITS_FOR_CREDIT',]].groupby(['EMPLID',
                                                                         'CLASS_TERM',
                                                                         'ERSS_COHORT_YEAR']).sum().groupby(level=[0]).cumsum().reset_index()
        grouped_cumsum = grouped_cumsum.add_prefix('CUM_')
        df_sub = pd.concat([grouped_agg,grouped_cumsum],axis=1)
        df_sub.drop(['CUM_EMPLID','CUM_CLASS_TERM','CUM_ERSS_COHORT_YEAR'],axis=1,inplace=True)
        
        df_sub['TERM_GPA'] = df_sub['ENRL_GRADE_POINTS_IN_GPA'] / df_sub['ENRL_UNITS_IN_GPA']
        df_sub['CUM_GPA'] = df_sub['CUM_ENRL_GRADE_POINTS_IN_GPA'] / df_sub['CUM_ENRL_UNITS_IN_GPA']
        
        df_sub['TERM_DIFF'] = pd.to_numeric(df_sub['CLASS_TERM']) - pd.to_numeric(df_sub['ERSS_COHORT_YEAR'])
        df_sub['N'] = df_sub['TERM_DIFF'].apply(lambda x: int(x/5+1) if int(repr(x)[-1]) == 0 else int((x + 2)/5) )
        df_sub['PRESCRIBED_UNITS'] = df_sub['N'] * 15
        
        df_sub['LOAD_INDEX'] = df_sub['CUM_ENRL_UNITS_FOR_CREDIT'] / df_sub['PRESCRIBED_UNITS']
        df_sub['COMPLETION_RATE'] = df_sub['ENRL_UNITS_FOR_CREDIT'] / df_sub['ENRL_UNITS_TAKEN']
        
        return df_sub

    def corr_mat(self, X):
        corr = X.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(8, 8))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,
                    square=True, #xticklabels=2, yticklabels=2,
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        
        plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 07:35:29 2020

@author: egeakertek
"""
import pandas as pd
import pandas_profiling
from scipy import stats # statistical library
from statsmodels.stats.weightstats import ztest # statistical library for hypothesis testing
import plotly.graph_objs as go # interactive plotting library
import ppscore as pps
import seaborn as sns

#%% Read the data

data = pd.read_csv("train.csv")


#%% EDA

# descrpition = data.describe()
# info1= data.info()

# Profiler. Amazing tool.
# report = pandas_profiling.ProfileReport(data)

# report.to_file(output_file='report.html')

# AutoViz. Awesome.

# from autoviz.AutoViz_Class import AutoViz_Class

# AV = AutoViz_Class()

# report2 = AV.AutoViz("train.csv")

#%%  Is age important?

data_survivors = data[data['Survived'] == 1]
data_nonsurvivors = data[data['Survived'] == 0]
dist_a = data_survivors['Age'].dropna()

# Second distribution for the hypothesis test: Ages of non-survivors
dist_b = data_nonsurvivors['Age'].dropna()
# Z-test: Checking if the distribution means (ages of survivors vs ages of non-survivors) are statistically different
t_stat, p_value = ztest(dist_a, dist_b)

print("----- Z Test Results -----")
print("T stat. = " + str(t_stat))
print("P value = " + str(p_value)) # P-value is less than 0.05

print("")

# T-test: Checking if the distribution means (ages of survivors vs ages of non-survivors) are statistically different
t_stat_2, p_value_2 = stats.ttest_ind(dist_a, dist_b)
print("----- T Test Results -----")
print("T stat. = " + str(t_stat_2))
print("P value = " + str(p_value_2)) # P-value is less than 0.05

#%% SEX 

# Taking the count of each Sex value inside the Survivors
data_survivors_sex = data_survivors['Sex'].value_counts()

data_survivors_sex = pd.DataFrame({'Sex':data_survivors_sex.index, 'count':data_survivors_sex.values})

# Taking the count of each Sex value inside the Survivors
data_nonsurvivors_sex = data_nonsurvivors['Sex'].value_counts()
data_nonsurvivors_sex = pd.DataFrame({'Sex':data_nonsurvivors_sex.index, 'count':data_nonsurvivors_sex.values})

# Pclass

# Taking the count of each Pclass value inside the Survivors
data_survivors_pclass = data_survivors['Pclass'].value_counts()
data_survivors_pclass = pd.DataFrame({'Pclass':data_survivors_pclass.index, 'count':data_survivors_pclass.values})

# Taking the count of each Pclass value inside the Survivors
data_nonsurvivors_pclass = data_nonsurvivors['Pclass'].value_counts()
data_nonsurvivors_pclass = pd.DataFrame({'Pclass':data_nonsurvivors_pclass.index, 'count':data_nonsurvivors_pclass.values})

#%% Is Fare imp?
# Third distribution for the hypothesis test - Fares of survivors
dist_c = data_survivors['Fare'].dropna()

# Fourth distribution for the hypothesis test - Fares of non-survivors
dist_d = data_nonsurvivors['Fare'].dropna()
# Z-test: Checking if the distribution means (fares of survivors vs fares of non-survivors) are statistically different
t_stat_3, p_value_3 = ztest(dist_c, dist_d)
print("----- Z Test Results -----")
print("T stat. = " + str(t_stat_3))
print("P value = " + str(p_value_3)) # P-value is less than 0.05

print("")

# T-test: Checking if the distribution means (fares of survivors vs fares of non-survivors) are statistically different
t_stat_4, p_value_4 = stats.ttest_ind(dist_c, dist_d)
print("----- T Test Results -----")
print("T stat. = " + str(t_stat_4))
print("P value = " + str(p_value_4)) # P-value is less than 0.05

# PP - Score 

matrix_data = pps.matrix(data)[['x', 'y', 'ppscore']].pivot(columns='Features', index='y', values='ppscore')
matrix_data = matrix_data.apply(lambda x: round(x, 2)) # Rounding matrix_data's values to 0,XX

sns.heatmap(matrix_data, vmin=0, vmax=1, cmap="Blues", linewidths=0.75, annot=True)


#%% Feature Engineering

data['AgeCat'] = ''
data['AgeCat'].loc[(data['Age'] < 18)] = 'young'
data['AgeCat'].loc[(data['Age'] >= 18) & (data['Age'] < 56)] = 'mature'
data['AgeCat'].loc[(data['Age'] >= 56)] = 'senior'


# Creating a categorical variable for Family Sizes
data['FamilySize'] = ''
data['FamilySize'].loc[(data['SibSp'] <= 2)] = 'small'
data['FamilySize'].loc[(data['SibSp'] > 2) & (data['SibSp'] <= 5 )] = 'medium'
data['FamilySize'].loc[(data['SibSp'] > 5)] = 'large'


# Creating a categorical variable to tell if the passenger is alone
data['IsAlone'] = ''
data['IsAlone'].loc[((data['SibSp'] + data['Parch']) > 0)] = 'no'
data['IsAlone'].loc[((data['SibSp'] + data['Parch']) == 0)] = 'yes'


# Creating a categorical variable to tell if the passenger is a Young/Mature/Senior male or a Young/Mature/Senior female
data['SexCat'] = ''
data['SexCat'].loc[(data['Sex'] == 'male') & (data['Age'] <= 21)] = 'youngmale'
data['SexCat'].loc[(data['Sex'] == 'male') & ((data['Age'] > 21) & (data['Age']) < 50)] = 'maturemale'
data['SexCat'].loc[(data['Sex'] == 'male') & (data['Age'] > 50)] = 'seniormale'
data['SexCat'].loc[(data['Sex'] == 'female') & (data['Age'] <= 21)] = 'youngfemale'
data['SexCat'].loc[(data['Sex'] == 'female') & ((data['Age'] > 21) & (data['Age']) < 50)] = 'maturefemale'
data['SexCat'].loc[(data['Sex'] == 'female') & (data['Age'] > 50)] = 'seniorfemale'


# Taking another look at the data
data.head(10)

#%% IMPORT MODEL LIBS.
import collections
import matplotlib.pyplot as plt
from scipy import stats
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from category_encoders import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from xgboost import XGBClassifier, plot_importance as plot_importance_xgb
from lightgbm import LGBMClassifier, plot_importance as plot_importance_lgbm
def get_feature_names(df):
    
#%%
    # Splitting the target
    target = df['Survived']

    # Dropping unused columns from the feature set
    df.drop(['PassengerId', 'Survived', 'Ticket', 'Name', 'Cabin'], axis=1, inplace=True)

    # Splitting categorical and numerical column dataframes
    categorical_df = df.select_dtypes(include=['object'])
    numeric_df = df.select_dtypes(exclude=['object'])

    # And then, storing the names of categorical and numerical columns.
    categorical_columns = list(categorical_df.columns)
    numeric_columns = list(numeric_df.columns)
    
    print("Categorical columns:\n", categorical_columns)
    print("\nNumeric columns:\n", numeric_columns)

    return target, categorical_columns, numeric_columns

target, categorical_columns, numeric_columns = get_feature_names(data)


#%% Balancing the data


def balancingClassesRus(x_train, y_train):
    
    # Using RandomUnderSampler to balance our training data points
    rus = RandomUnderSampler(random_state=7)
    features_balanced, target_balanced = rus.fit_resample(x_train, y_train)
    
    print("Count for each class value after RandomUnderSampler:", collections.Counter(target_balanced))
    
    return features_balanced, target_balanced


def balancingClassesSmoteenn(x_train, y_train):
    
    # Using SMOTEEN to balance our training data points
    smn = SMOTEENN(random_state=7)
    features_balanced, target_balanced = smn.fit_resample(x_train, y_train)
    
    print("Count for each class value after SMOTEEN:", collections.Counter(target_balanced))
    
    return features_balanced, target_balanced

def balancingClassesSmote(x_train, y_train):

    # Using SMOTE to to balance our training data points
    sm = SMOTE(random_state=7)
    features_balanced, target_balanced = sm.fit_resample(x_train, y_train)

    print("Count for each class value after SMOTE:", collections.Counter(target_balanced))

    return features_balanced, target_balanced















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

matrix_data = pps.matrix(data)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
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

    
#%% Splitting the target

def get_feature_names(df):
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

#%% Function responsible for checking our model's performance on the test data
def testSetResultsClassifier(classifier, x_test, y_test):
    predictions = classifier.predict(x_test)
    
    results = []
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    
    results.append(f1)
    results.append(precision)
    results.append(recall)
    results.append(roc_auc)
    results.append(accuracy)
    
    print("\n\n#---------------- Test set results (Best Classifier) ----------------#\n")
    print("F1 score, Precision, Recall, ROC_AUC score, Accuracy:")
    print(results)
    
    return results


#%% Creating the pipelines

# Now, we are going to create our Pipeline, fitting several different data preprocessing, feature selection 
# and modeling techniques inside a RandomSearchCV, to check which group of techniques has better performance.

# Building a Pipeline inside RandomSearchCV, responsible for finding the best model and it's parameters
def defineBestModelPipeline(df, target, categorical_columns, numeric_columns):
    
    # Splitting original data into Train and Test
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.1, random_state=42)
    y_train = y_train.to_numpy() # Transforming training targets into numpy arrays
    y_test = y_test.to_numpy() # Transforming test targets into numpy arrays
    
    
    # # If desired, we can balance training classes using one of the functions below
    # # Obtaining balanced data for modeling using Random Under Sampling
    x_train, y_train = balancingClassesRus(x_train, y_train)

    # # Obtaining balanced data for modeling using SMOTEENN
    #x_train, y_train = balancingClassesSmoteenn(x_train, y_train)

    # # Obtaining balanced data for modeling using SMOTE
    #x_train, y_train = balancingClassesSmote(x_train, y_train)

# 1st -> Numeric Transformers
    # Here, we are creating different several different data transformation pipelines 
    # to be applied in our numeric features
    numeric_transformer_1 = Pipeline(steps=[('imp', IterativeImputer(max_iter=30, random_state=42)),
                                            ('scaler', MinMaxScaler())])
    
    numeric_transformer_2 = Pipeline(steps=[('imp', IterativeImputer(max_iter=20, random_state=42)),
                                            ('scaler', StandardScaler())])
    
    numeric_transformer_3 = Pipeline(steps=[('imp', SimpleImputer(strategy='mean')),
                                            ('scaler', MinMaxScaler())])
    
    numeric_transformer_4 = Pipeline(steps=[('imp', SimpleImputer(strategy='median')),
                                            ('scaler', StandardScaler())])
    
    # 2nd -> Categorical Transformer
    # Despite my option of not doing it, you can also choose to create different 
    # data transformation pipelines for your categorical features.
    categorical_transformer = Pipeline(steps=[('frequent', SimpleImputer(strategy='most_frequent')),
                                              ('onehot', OneHotEncoder(use_cat_names=True))])
    # 3rd -> Combining both numerical and categorical pipelines
    # Here, we are creating different ColumnTransformers, each one with a different numerical transformation
    data_transformations_1 = ColumnTransformer(transformers=[('num', numeric_transformer_1, numeric_columns),
                                                             ('cat', categorical_transformer, categorical_columns)])
    
    data_transformations_2 = ColumnTransformer(transformers=[('num', numeric_transformer_2, numeric_columns),
                                                             ('cat', categorical_transformer, categorical_columns)])
    
    data_transformations_3 = ColumnTransformer(transformers=[('num', numeric_transformer_3, numeric_columns),
                                                             ('cat', categorical_transformer, categorical_columns)])
    
    data_transformations_4 = ColumnTransformer(transformers=[('num', numeric_transformer_4, numeric_columns),
                                                             ('cat', categorical_transformer, categorical_columns)])
    
    # And finally, we are going to apply these different data transformations to RandomSearchCV,
    # trying to find the best imputing strategy, the best feature engineering strategy
    # and the best model with it's respective parameters.
    # Below, we just need to initialize a Pipeline object with any transformations we want, on each of the steps.
    pipe = Pipeline(steps=[('data_transformations', data_transformations_1), # Initializing data transformation step by choosing any of the above
                           ('feature_eng', PCA()), # Initializing feature engineering step by choosing any desired method
                           ('clf', SVC())]) # Initializing modeling step of the pipeline with any model object
                           #memory='cache_folder') -> Used to optimize memory when needed

    # Now, we define the grid of parameters that RandomSearchCV will use. It will randomly chose
    # options for each step inside the dictionaries ('data transformations', 'feature_eng', 'clf'
    # and 'clf parameters'). In the end of it's iterations, RandomSearchCV will return the best options.
    params_grid = [
                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [KNeighborsClassifier()],
                     'clf__n_neighbors': stats.randint(1, 30),
                     'clf__metric': ['minkowski', 'euclidean']},
                    
                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [LogisticRegression()],
                     'clf__penalty': ['l1', 'l2'],
                     'clf__C': stats.uniform(0.01, 10)},

                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [SVC()],
                     'clf__C': stats.uniform(0.01, 1),
                     'clf__gamma': stats.uniform(0.01, 1)},
        
                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [DecisionTreeClassifier()],
                     'clf__criterion': ['gini', 'entropy'],
                     'clf__max_features': [None, "auto", "log2"],
                     'clf__max_depth': [None, stats.randint(1, 5)]},

                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [RandomForestClassifier()],
                     'clf__n_estimators': stats.randint(10, 175),
                     'clf__max_features': [None, "auto", "log2"],
                     'clf__max_depth': [None, stats.randint(1, 5)],
                     'clf__random_state': stats.randint(1, 49)},
        
                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [ExtraTreesClassifier()],
                     'clf__n_estimators': stats.randint(10, 150),
                     'clf__max_features': [None, "auto", "log2"],
                     'clf__max_depth': [None, stats.randint(1, 6)]},
        
                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [GradientBoostingClassifier()],
                     'clf__n_estimators': stats.randint(10, 100),
                     'clf__learning_rate': stats.uniform(0.01, 0.7),
                     'clf__max_depth': [None, stats.randint(1, 6)]},
        
                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [LGBMClassifier()],
                     'clf__n_estimators': stats.randint(1, 100),
                     'clf__learning_rate': stats.uniform(0.01, 0.7),
                     'clf__max_depth': [None, stats.randint(1, 6)]},
                    
                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [XGBClassifier()],
                     'clf__n_estimators': stats.randint(5, 125),
                     'clf__eta': stats.uniform(0.01, 1),
                     'clf__max_depth': [None, stats.randint(1, 6)],
                     'clf__gamma': stats.uniform(0.01, 1)},

                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [StackingClassifier(estimators=[('svc', SVC(C=1, gamma=1)),
                                                            ('rf', RandomForestClassifier(max_depth=7, max_features=None, n_estimators=60, n_jobs=-1, random_state=42)),
                                                            ('xgb', XGBClassifier(eta=0.6, gamma=0.7, max_depth=None, n_estimators=30))],
                                                final_estimator=LogisticRegression(C=1))]},
        
                    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [VotingClassifier(estimators=[('gbt', GradientBoostingClassifier(learning_rate=0.8, max_depth=None, n_estimators=30)),
                                                          ('lgbm', LGBMClassifier(n_estimators=30, learning_rate=0.6, max_depth=None)),
                                                          ('xgb', XGBClassifier(eta=0.8, gamma=0.8, max_depth=None, n_estimators=40))],
                                              voting='soft')]}
                    ]
    # Now, we fit a RandomSearchCV to search over the grid of parameters defined above
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    best_model_pipeline = RandomizedSearchCV(pipe, params_grid, n_iter=500, 
                                             scoring=metrics, refit='accuracy', 
                                             n_jobs=-1, cv=5, random_state=42)

    best_model_pipeline.fit(x_train, y_train)
    
    
    # At last, we check the final results
    print("\n\n#---------------- Best Data Pipeline found in RandomSearchCV  ----------------#\n\n", best_model_pipeline.best_estimator_[0])
    print("\n\n#---------------- Best Feature Engineering technique found in RandomSearchCV  ----------------#\n\n", best_model_pipeline.best_estimator_[1])
    print("\n\n#---------------- Best Classifier found in RandomSearchCV  ----------------#\n\n", best_model_pipeline.best_estimator_[2])
    print("\n\n#---------------- Best Estimator's average Accuracy Score on CV (validation set) ----------------#\n\n", best_model_pipeline.best_score_)
    
    return x_train, x_test, y_train, y_test, best_model_pipeline

#%% Calling the pipelines

# Calling the function above, returing train/test data and best model's pipeline
x_train, x_test, y_train, y_test, best_model_pipeline = defineBestModelPipeline(data, target, categorical_columns, numeric_columns)


# Checking best model's performance on test data
test_set_results = testSetResultsClassifier(best_model_pipeline, x_test, y_test)


#%%
df_results = pd.DataFrame(best_model_pipeline.cv_results_)

print(df_results)







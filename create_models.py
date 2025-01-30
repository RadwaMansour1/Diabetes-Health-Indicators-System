import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from tkinter import *
from tkinter import messagebox
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
import pickle


def clean(df):
    # drop duplicates
    # df.drop_duplicates(keep=False, inplace=True)
    # df.reset_index(inplace=True)
    # df.drop(['index'],axis=1, inplace=True)

    # balance data
    x = df.loc[:, 'HighBP':]
    y = df.loc[:, 'Diabetes_binary']
    undersample = RandomUnderSampler(sampling_strategy='majority')
    x1, y1 = undersample.fit_resample(x, y)
    balanced_df = pd.DataFrame.from_records(data=pd.concat((y1, x1), axis=1))
    balanced_df.columns = df.columns

    # feature selection
    cor = balanced_df.corr()
    cor_target = abs(cor["Diabetes_binary"])
    relevant_features = cor_target[cor_target >= 0.25]
    df = balanced_df[relevant_features.index]

    #normalization
    sc = StandardScaler()
    scaled_data = pd.DataFrame(sc.fit_transform(
        df.drop(['Diabetes_binary'], axis=1, inplace=False)))
    scaled_data = pd.concat([df['Diabetes_binary'], scaled_data], axis=1)
    scaled_data.columns = df.columns
    return scaled_data


# data cleaning
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

df = clean(df)

# print(df)

X = df.drop(['Diabetes_binary'], axis=1)
y = df['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)


# solvers = ['newton-cg', 'lbfgs', 'liblinear']
# penalty = ['l2']
# c_values = [100, 10, 1.0, 0.1, 0.01]
# # define grid search
# grid = dict(solver=solvers, penalty=penalty, C=c_values)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=grid,
#                            n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
# grid_result = grid_search.fit(X_train, y_train)

# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# model_LogisticRegression = LogisticRegression(C=grid_result.best_params_['C'], penalty=grid_result.best_params_['penalty'], solver=grid_result.best_params_['solver'])
# model_LogisticRegression.fit(X_train, y_train)
# filename = 'Logistic_Regression.sav'
# pickle.dump(model_LogisticRegression, open(filename, 'wb'))


# didn't work
# kernel = ['poly', 'rbf', 'sigmoid']
# C = [50, 10, 1.0, 0.1, 0.01]
# gamma = ['scale']
# # define grid search
# grid = dict(kernel=kernel,C=C,gamma=gamma)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=SVC(), param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train, y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# model_svm = SVC()
# model_svm.fit(X_train, y_train)
# print(model_svm.score(X_test, y_test))
# # save the model to disk
# filename = 'SVM.sav'
# pickle.dump(model_svm, open(filename, 'wb'))


# criterion = ['gini', 'entropy']
# max_depth = [2,4,6,8,10,12]
# # define grid search
# grid = dict(criterion=criterion, max_depth=max_depth)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train, y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# id3 = DecisionTreeClassifier(criterion=grid_result.best_params_['criterion'], max_depth=grid_result.best_params_['max_depth'])
# id3.fit(X_train, y_train)
# print(id3.score(X_test, y_test))
# # save the model to disk
# filename = 'DecisionTree.sav'
# pickle.dump(id3, open(filename, 'wb'))


# n_neighbors = range(1, 21, 2)
# weights = ['uniform', 'distance']
# metric = ['euclidean', 'manhattan', 'minkowski']
# # define grid search
# grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train, y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# KNN = KNeighborsClassifier(n_neighbors=grid_result.best_params_['n_neighbors'], weights=grid_result.best_params_['weights'], metric=grid_result.best_params_['metric'])
# KNN.fit(X_train, y_train)
# print(KNN.score(X_test, y_test))
# # save the model to disk
# filename = 'KNN.sav'
# pickle.dump(KNN, open(filename, 'wb'))


# naive
# model = GaussianNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(model.score(X_test, y_test))
# pickle.dump(model, open('naive.sav', 'wb'))


# # combine models
# # load the model from disk
# filename_1 = 'Logistic_Regression.sav'
# Logistic_Regression = pickle.load(open(filename_1, 'rb'))

# filename_2 = 'DecisionTree.sav'
# id3 = pickle.load(open(filename_2, 'rb'))

# filename_3 = 'KNN.sav'
# KNN = pickle.load(open(filename_3, 'rb'))

# filename_4 = 'SVM.sav'
# SVM = pickle.load(open(filename_4, 'rb'))

# voting_classifier_hard = VotingClassifier(
#     estimators = [('dtc',id3),
#                   ('lr', Logistic_Regression),
#                   ('KNN', KNN),
#                   ('SVM', SVM),
#                   ('np', model)],
#     voting='hard')

# voting_classifier_hard.fit(X_train, y_train)
# print(voting_classifier_hard.score(X_test, y_test))

# filename = 'HardVoting.sav'
# pickle.dump(voting_classifier_hard, open(filename, 'wb'))


# cor = balanced_df.corr()
# sns.set(rc = {'figure.figsize':(10,6)})
# sns.heatmap(cor)
# plt.show()

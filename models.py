import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sklearn as sk
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def Logistic_regression(X_test, y_test):
    model = pickle.load(open('Logistic_Regression.sav', 'rb'))
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = sk.metrics.roc_curve(y_test,  y_pred_proba)
    auc = sk.metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.legend()
    plt.savefig('ROC.png')
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    plt.figure(figsize = (5, 5))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    sns.heatmap(cm, cmap = 'Reds', annot = True, fmt = 'd', linewidths = 8, cbar = False, annot_kws = {'fontsize': 20},
                yticklabels = ['Healthy', 'Diabetic'], xticklabels = ['Predicted Healthy', 'Predicted Diabetic'])
    plt.yticks()
    plt.savefig('LogisticRegression.png')
    plt.show()


def SVM(X_test, y_test):
    model = pickle.load(open('SVM.sav', 'rb'))
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    plt.figure(figsize = (5, 5))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    sns.heatmap(cm, cmap = 'Reds', annot = True, fmt = 'd', linewidths = 8, cbar = False, annot_kws = {'fontsize': 20},
                yticklabels = ['Healthy', 'Diabetic'], xticklabels = ['Predicted Healthy', 'Predicted Diabetic'])
    plt.yticks()
    plt.savefig('SVM.png')
    plt.show()


def id3(X_test, y_test):
    model = pickle.load(open('DecisionTree.sav', 'rb'))
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    plt.figure(figsize = (5, 5))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    sns.heatmap(cm, cmap = 'Reds', annot = True, fmt = 'd', linewidths = 8, cbar = False, annot_kws = {'fontsize': 20},
                yticklabels = ['Healthy', 'Diabetic'], xticklabels = ['Predicted Healthy', 'Predicted Diabetic'])
    plt.yticks()
    plt.savefig('ID3.png')


def KNN(X_test, y_test):
    model = pickle.load(open('KNN.sav', 'rb'))
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    plt.figure(figsize = (5, 5))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    sns.heatmap(cm, cmap = 'Reds', annot = True, fmt = 'd', linewidths = 8, cbar = False, annot_kws = {'fontsize': 20},
                yticklabels = ['Healthy', 'Diabetic'], xticklabels = ['Predicted Healthy', 'Predicted Diabetic'])
    plt.yticks()
    plt.savefig('KNN_cm.png')


df = pd.read_csv("clean.csv")
X = df.drop(['Diabetes_binary'], axis=1)
y = df['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=0)


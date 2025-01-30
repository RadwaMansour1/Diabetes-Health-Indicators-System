import pandas as pd
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from myLogisticRegression import *
from myKNN import *
from myDecisionTree import *
from myNaiveBayes import *
from sklearn import preprocessing
from mySVM import *


# data cleaning
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
# print(type(df))


# drop duplicates
# df.drop_duplicates(keep=False, inplace=True)
# df.reset_index(inplace=True)

df_1 = df[df.Diabetes_binary == 1]
df_0 = df[df.Diabetes_binary == 0]

df_0_sampled = resample(
    df_0, replace=True, n_samples=len(df_1), random_state=123)

balanced_df = pd.concat([df_1, df_0_sampled])
balanced_df.reset_index(inplace=True)
balanced_df.drop(['index'], axis=1, inplace=True)

# correlation
cor = balanced_df.corr()
# sns.set(rc = {'figure.figsize':(10,6)})
# sns.heatmap(cor)
# plt.show()

# bar plot
# key = ['men', 'women']
# values = [df_1['Sex'].value_counts()[0], df_1['Sex'].value_counts()[1]]
# plt.bar(key, values, color ='maroon',width = 0.4)
# plt.show()

# feature selection
cor_target = abs(cor["Diabetes_binary"])
relevant_features = cor_target[cor_target >= 0.25]
df = balanced_df[relevant_features.index]
print(df.head())

# feature engineering
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# X_train_transformed_scaled = scaler.fit_transform(X_train_transformed)
# X_test_transformed_scaled = scaler.transform(X_test_transformed)

# from sklearn.preprocessing import PolynomialFeatures

# poly = PolynomialFeatures(degree=2).fit(X_train_transformed)
# X_train_poly = poly.transform(X_train_transformed_scaled)
# X_test_poly = poly.transform(X_test_transformed_scaled)

# data scaling

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(df)
# df = pd.DataFrame(data_scaled_minmax)
# print(df.head(10))


# Logistic_Regression = myLogisticRegression(df)
# Logistic_Regression.getInformation()


# vec = DictVectorizer()
# vec.fit_transform(df)
# print(vec.toarray())

# sns.set(rc = {'figure.figsize':(10,6)})
# sns.heatmap(df.corr(), annot=True)
# plt.show()

# svm = SVM(df)
# svm.getInformation()


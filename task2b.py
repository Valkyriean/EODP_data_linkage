# Code by Jiachen Li, 1068299
import csv
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read in both files
world = pd.read_csv("world.csv", header=0, encoding='ISO-8859-1')
life = pd.read_csv("life.csv", header=0, encoding='ISO-8859-1')
# Merge two file into on dataframe
combine = world.merge(life, left_on='Country Name', right_on='Country', how="inner")
combine.sort_values(by=['Country Name'], ascending=True)
# Extract data we are interested in
x = combine[combine.columns[3:23]]
y = combine[combine.columns[26]]

splits = range(5,95,5)
accu_list=[]
for split in splits:
    X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=split/100,test_size=1-split/100)
    X_train = X_train.replace('..', np.NaN)
    X_train = X_train.fillna(X_train.median()).astype("float64")
    X_test = X_test.replace('..', np.NaN)
    X_test = X_test.fillna(X_train.median()).astype("float64")
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)



    accu_list.append(accuracy_score(y_test, y_pred))

plt.plot(splits,accu_list)
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=200)
# impute with median
x_train = x_train.replace('..', np.NaN)
x_train = x_train.fillna(x_train.median()).astype("float64")
x_test = x_test.replace('..', np.NaN)
x_test = x_test.fillna(x_train.median()).astype("float64")

######### feature engineering #########
engineered_x_train = x_train
engineered_x_test = x_test
# Interaction term pairs feature engineering
features = x_train.shape[1]
for i in range(0, features-1):
    for j in range(i+1, features):
        key_i_train = x_train.columns[i]
        key_j_train = x_train.columns[j]
        engineered_x_train[key_i_train+' X '+key_j_train] = x_train[key_i_train].mul(x_train[key_j_train])
        key_i_test = x_test.columns[i]
        key_j_test = x_test.columns[j]
        engineered_x_test[key_i_test+' X '+key_j_test] = x_test[key_i_test].mul(x_test[key_j_test])


# Clustering labels feature engineering
kmeans = KMeans(n_clusters=2, random_state=0).fit(x_train)
# engineered_x_train["kmeans"] = kmeans.labels_
# engineered_x_test["kmeans"] = kmeans.predict(x_test)
engineered_x_train = np.c_[engineered_x_train, kmeans.labels_]
engineered_x_test = np.c_[engineered_x_test, kmeans.predict(x_test)]

# Feature selection
select = SelectKBest(chi2, k=4).fit(engineered_x_train, y_train)
engineered_x_train = select.transform(engineered_x_train)
engineered_x_test = select.transform(engineered_x_test)

# Scale
engineered_scaler = preprocessing.StandardScaler().fit(engineered_x_train)
engineered_x_train = engineered_scaler.transform(engineered_x_train)
engineered_x_test = engineered_scaler.transform(engineered_x_test)

# K Neighbors Classifier with k = 3
engineered_knn3 = KNeighborsClassifier(n_neighbors=3)
engineered_knn3.fit(engineered_x_train, y_train)
engineered_knn3_pred = engineered_knn3.predict(engineered_x_test)
engineered_knn3_score = accuracy_score(y_test, engineered_knn3_pred)

print("Accuracy of feature engineering: " + str(round(engineered_knn3_score, 3)))

######### first four features #########

first_four_x_train = x_train[x_train.columns[0:4]]
first_four_x_test = x_test[x_test.columns[0:4]]
first_four_scaler = preprocessing.StandardScaler().fit(first_four_x_train)
first_four_x_train = first_four_scaler.transform(first_four_x_train)
first_four_x_test = first_four_scaler.transform(first_four_x_test)
# K Neighbors Classifier with k = 3
first_four_knn3 = KNeighborsClassifier(n_neighbors=3)
first_four_knn3.fit(first_four_x_train, y_train)
first_four_knn3_pred = first_four_knn3.predict(first_four_x_test)
first_four_knn3_score = accuracy_score(y_test, first_four_knn3_pred)
print("Accuracy of first four features: " + str(round(first_four_knn3_score, 3)))

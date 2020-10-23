# Code by Jiachen Li, 1068299
import csv
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Read in both files
world = pd.read_csv("world.csv", header=0, encoding='ISO-8859-1')
life = pd.read_csv("life.csv", header=0, encoding='ISO-8859-1')
# Create file task2a.csv for write
task2a = open("task2a.csv", 'w', newline='')
task2a_writer = csv.writer(task2a)
task2a_writer.writerow(["feature", "median", "mean", "variance"])
# Merge two file into on dataframe
combine = world.merge(life, left_on='Country Name', right_on='Country', how="inner")
combine.sort_values(by=['Country Name'], ascending=True)
# Extract data we are interested in
x = combine[combine.columns[3:23]]
y = combine[combine.columns[26]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=200)
# impute with median
x_train = x_train.replace('..', np.NaN)
x_train = x_train.fillna(x_train.median()).astype("float64")
x_test = x_test.replace('..', np.NaN)
x_test = x_test.fillna(x_train.median()).astype("float64")
# Scale x_train
for key, value in x_train.iteritems():
    task2a_writer.writerow([key, round(value.median(), 3), round(value.mean(), 3), round(value.var(), 3)])
# Scale x_test
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# K Neighbors Classifier with k = 3
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(x_train, y_train)
knn3_pred = knn3.predict(x_test)
knn3_score = accuracy_score(y_test, knn3_pred)
# K Neighbors Classifier with k = 7
knn7 = KNeighborsClassifier(n_neighbors=7)
knn7.fit(x_train, y_train)
knn7_pred = knn7.predict(x_test)
knn7_score = accuracy_score(y_test, knn7_pred)
# Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=3, random_state=200)
clf.fit(x_train, y_train)
clf_pred = clf.predict(x_test)
clf_score = accuracy_score(y_test, clf_pred)
# Printing out results
print("Accuracy of decision tree: " + str(round(clf_score, 3)))
print("Accuracy of k-nn (k=3): " + str(round(knn3_score, 3)))
print("Accuracy of k-nn (k=7): " + str(round(knn7_score, 3)))

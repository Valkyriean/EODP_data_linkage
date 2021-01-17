import pandas as pd
import numpy as np
import csv
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

life = pd.read_csv('life.csv', encoding='ISO-8859-1')
world = pd.read_csv('world.csv', encoding='ISO-8859-1')
combine = world.merge(life, left_on='Country Name', right_on='Country', how='inner')

combine.sort_values(by=['Country Name'], ascending=True)
col = combine.columns[3:23]
data = combine[col]
classlabel = combine['Life expectancy at birth (years)']
X_train, X_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.7, test_size=0.3, random_state=200)

X_train = X_train.replace('..', np.nan)
X_train = X_train.fillna(X_train.median()).astype(float)

X_test = X_test.replace('..', np.nan)
X_test = X_test.fillna(X_test.median()).astype(float)

median = []
mean = []
variance = []
for i in col:
    median.append(np.around(X_train[i].median(), 3))
    mean.append(np.around(X_train[i].mean(), 3))
    variance.append(np.around(X_train[i].var(), 3))

out = pd.DataFrame({'feature': col, 'median': median, 'mean': mean, 'variance': variance})
out.to_csv("task2a.csv", index=False)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

dt = DecisionTreeClassifier(random_state=200, max_depth=3)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("Accuracy of decision tree: ", round(accuracy_score(y_test, y_pred), 3))

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy of k-nn (k=3): ", round(accuracy_score(y_test, y_pred), 3))

knn = neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy of k-nn (k=7): ", round(accuracy_score(y_test, y_pred), 3))


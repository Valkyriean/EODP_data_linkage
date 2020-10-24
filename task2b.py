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
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform



def VAT(R):
    """

    VAT algorithm adapted from matlab version:
    http://www.ece.mtu.edu/~thavens/code/VAT.m

    Args:
        R (n*n double): Dissimilarity data input
        R (n*D double): vector input (R is converted to sq. Euclidean distance)
    Returns:
        RV (n*n double): VAT-reordered dissimilarity data
        C (n int): Connection indexes of MST in [0,n)
        I (n int): Reordered indexes of R, the input data in [0,n)
    """

    R = np.array(R)
    N, M = R.shape
    if N != M:
        R = squareform(pdist(R))

    J = list(range(0, N))

    y = np.max(R, axis=0)
    i = np.argmax(R, axis=0)
    j = np.argmax(y)
    y = np.max(y)

    I = i[j]
    del J[I]

    y = np.min(R[I, J], axis=0)
    j = np.argmin(R[I, J], axis=0)

    I = [I, J[j]]
    J = [e for e in J if e != J[j]]

    C = [1, 1]
    for r in range(2, N - 1):
        y = np.min(R[I, :][:, J], axis=0)
        i = np.argmin(R[I, :][:, J], axis=0)
        j = np.argmin(y)
        y = np.min(y)
        I.extend([J[j]])
        J = [e for e in J if e != J[j]]
        C.extend([i[j]])

    y = np.min(R[I, :][:, J], axis=0)
    i = np.argmin(R[I, :][:, J], axis=0)

    I.extend(J)
    C.extend(i)

    RI = list(range(N))
    for idx, val in enumerate(I):
        RI[val] = idx

    RV = R[I, :][:, I]

    return RV.tolist(), C, I





# Read in both files
world = pd.read_csv("world.csv", header=0, encoding='ISO-8859-1')
life = pd.read_csv("life.csv", header=0, encoding='ISO-8859-1')
# Merge two file into on dataframe
combine = world.merge(life, left_on='Country Name', right_on='Country', how="inner")
combine.sort_values(by=['Country Name'], ascending=True)
# Extract data we are interested in
world_data = world[world.columns[3:23]]
x = combine[combine.columns[3:23]]
y = combine[combine.columns[26]]


RV, C, I = VAT(x)
x=sns.heatmap(RV,cmap='viridis',xticklabels=False,yticklabels=False)
x.set(xlabel='Objects', ylabel='Objects')
plt.show()

x.heatmap(x,cmap='viridis',xticklabels=True,yticklabels=False)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
# impute with median
x_train = x_train.replace('..', np.NaN)
x_train = x_train.fillna(x_train.median()).astype("float64")
world_data = world_data.replace('..', np.NaN)
world_data = world_data.fillna(world_data.median()).astype("float64")
x_test = x_test.replace('..', np.NaN)
x_test = x_test.fillna(x_train.median()).astype("float64")

######### Feature Engineering #########
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
kmeans = KMeans(n_clusters=2).fit(world_data)
# engineered_x_train["kmeans"] = kmeans.predict(x_train)
# engineered_x_test["kmeans"] = kmeans.predict(x_test)
engineered_x_train = np.c_[engineered_x_train, kmeans.predict(x_train[x_train.columns[0:20]])]
engineered_x_test = np.c_[engineered_x_test, kmeans.predict(x_test[x_test.columns[0:20]])]

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

######### Principal Component Analysis #########


pca_scaler = preprocessing.StandardScaler()
pca_x_train = pca_scaler.fit_transform(x_train)
pca_x_test = pca_scaler.transform(x_test)
pca = PCA(n_components=4)
pca_x_train = pca.fit_transform(pca_x_train, y_train)
pca_x_test = pca.transform(pca_x_test)

# K Neighbors Classifier with k = 3
pca_knn3 = KNeighborsClassifier(n_neighbors=3)
pca_knn3.fit(pca_x_train, y_train)
pca_knn3_pred = pca_knn3.predict(pca_x_test)
pca_knn3_score = accuracy_score(y_test, pca_knn3_pred)
print("Accuracy of PCA: " + str(round(pca_knn3_score, 3)))

######### First Four Features #########

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



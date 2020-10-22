# Code by Jiachen Li, 1068299
import csv
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=200)
# impute with median
x_train = x_train.replace('..', np.NaN).fillna(x_train.median()).astype("float64")

for key, value in x_train.iteritems():
    value = value.astype("float64")
    task2a_writer.writerow([key, round(value.median(), 3), round(value.mean(), 3), round(value.var(), 3)])
    value = preprocessing.scale(value)
    x_train[key] = value
print(x_train)


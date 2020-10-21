# Code by Jiachen Li, 1068299

import csv
import pandas as pd
from sklearn import preprocessing

import numpy as np
from sklearn.model_selection import train_test_split

world = pd.read_csv("world.csv", header=0, encoding='ISO-8859-1')
life = pd.read_csv("life.csv", header=0, encoding='ISO-8859-1')

task2a = open("task2a.csv", 'w', newline='')
task2a_writer = csv.writer(task2a)
task2a_writer.writerow(["feature", "median", "mean", "variance"])

combine = world.merge(life, left_on='Country Name', right_on='Country', how="inner")
combine_train, combine_test = train_test_split(combine, test_size=0.3, random_state=200)
combine_train = combine_train.replace('..', np.NaN)
imputed = combine_train.fillna(combine_train.median())
i = 1
for key, value in combine_train.iteritems():
    if i in range(4, 24):
        value = value.astype("float64")
        # value = value.fillna(value.median())
        task2a_writer.writerow([key, value.median(), value.mean(), value.var()])
        value = preprocessing.scale(value)
        combine_train[key] = value
    i += 1
combine_train.to_csv("combine.csv", index=False)


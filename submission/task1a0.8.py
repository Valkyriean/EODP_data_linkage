import csv
import string

import pandas as pd
import textdistance

abt = pd.read_csv("abt_small.csv", encoding='ISO-8859-1')
buy = pd.read_csv("buy_small.csv", encoding='ISO-8859-1')
task1a = open("task1a.csv", 'w', newline='')
task1a_writer = csv.writer(task1a)
task1a_writer.writerow(["idAbt", "idBuy"])
for abt_i, abt_row in abt.iterrows():
    abt_name = abt_row["name"].lower().translate(str.maketrans('', '', string.punctuation))
    for buy_i, buy_row in buy.iterrows():
        buy_name = buy_row["name"].lower().translate(str.maketrans('', '', string.punctuation))
        score = textdistance.jaccard.normalized_similarity(abt_name, buy_name)
        if score > 0.6:
            task1a_writer.writerow([abt_row['idABT'], buy_row['idBuy']])

# Code by Jiachen Li, 1068299
import csv
import string
import pandas as pd
import textdistance
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Read in files
abt = pd.read_csv("abt_small.csv", encoding='ISO-8859-1')
buy = pd.read_csv("buy_small.csv", encoding='ISO-8859-1')
# Ready for write file
task1a = open("task1a.csv", 'w', newline='')
task1a_writer = csv.writer(task1a)
task1a_writer.writerow(["idAbt", "idBuy"])
# Traverse each row in abt
for abt_i, abt_row in abt.iterrows():
    max_score = 0
    # Preprocessing the name
    abt_name = abt_row["name"].lower().translate(str.maketrans('', '', string.punctuation))
    abt_brand = abt_name.split()[0]
    abt_serial_number = abt_name.split()[-1]
    for buy_i, buy_row in buy.iterrows():
        buy_name = buy_row["name"].lower().translate(str.maketrans('', '', string.punctuation))
        buy_description = str(buy_row["description"]).lower().translate(str.maketrans('', '', string.punctuation))
        manufacturer = buy_row["manufacturer"].lower().translate(str.maketrans('', '', string.punctuation)).split()[0]
        buy_serial_number = buy_name.split()[-1]
        # Try to find serial number of abt row in name and description of buy,
        # also find brand of abt in name or manufacturer in abt
        if (abt_serial_number in buy_name or abt_serial_number in buy_description) and \
                (abt_brand in buy_name or abt_brand in manufacturer):
            task1a_writer.writerow([abt_row['idABT'], buy_row['idBuy']])
        # If exact serial number or brand is not matching, try test distance match to reduce recall
        elif fuzz.token_set_ratio(abt_name, buy_name) > 87:
            task1a_writer.writerow([abt_row['idABT'], buy_row['idBuy']])

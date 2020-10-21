# Code by Jiachen Li, 1068299

import csv
import pandas as pd

# Read and write abt csv files
abt = pd.read_csv("abt.csv", encoding='ISO-8859-1')
abt_blocks = open("abt_blocks.csv", 'w', newline='')
abt_blocks_writer = csv.writer(abt_blocks)
abt_blocks_writer.writerow(["block_key", "product_id"])

# Block rows base on brand aka the first word in name
for abt_i, abt_row in abt.iterrows():
    blocks_key = abt_row["name"].lower().split(" ", 1)[0]
    abt_blocks_writer.writerow([blocks_key, abt_row['idABT']])

# Read and write buy csv files
buy = pd.read_csv("buy.csv", encoding='ISO-8859-1')
buy_blocks = open("buy_blocks.csv", 'w', newline='')
buy_blocks_writer = csv.writer(buy_blocks)
buy_blocks_writer.writerow(["block_key", "product_id"])

# Block rows base on brand aka the first word in name
for buy_i, buy_row in buy.iterrows():
    # Use manufacturer value first is applicable
    if isinstance(buy_row["manufacturer"], str):
        blocks_key = buy_row["manufacturer"].lower().split(" ", 1)[0]
    # Use first world in name other wise
    else:
        blocks_key = buy_row["name"].lower().split(" ", 1)[0]
    buy_blocks_writer.writerow([blocks_key, buy_row['idBuy']])

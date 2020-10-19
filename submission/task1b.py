import csv
import pandas as pd

abt = pd.read_csv("abt.csv", encoding='ISO-8859-1')
abt_blocks = open("abt_blocks.csv", 'w', newline='')
abt_blocks_writer = csv.writer(abt_blocks)
abt_blocks_writer.writerow(["block_key", "product_id"])

for abt_i, abt_row in abt.iterrows():
    blocks_key = abt_row["name"].lower().split(" ", 1)[0]
    abt_blocks_writer.writerow([blocks_key, abt_row['idABT']])

buy = pd.read_csv("buy.csv", encoding='ISO-8859-1')
buy_blocks = open("buy_blocks.csv", 'w', newline='')
buy_blocks_writer = csv.writer(buy_blocks)
buy_blocks_writer.writerow(["block_key", "product_id"])

for buy_i, buy_row in buy.iterrows():
    if isinstance(buy_row["manufacturer"], str):
        blocks_key = buy_row["manufacturer"].lower().split(" ", 1)[0]
    else:
        blocks_key = buy_row["name"].lower().split(" ", 1)[0]
    buy_blocks_writer.writerow([blocks_key, buy_row['idBuy']])

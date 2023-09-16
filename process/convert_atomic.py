""" 需要生成的文件
https://recbole.io/docs/user_guide/data/atomic_files.html

# dname.inter
user_id:token	item_id:token	rating:float	timestamp:float
196	242	3	881250949

# dname.link
item_id:token	entity_id:token
476	m.08gjyx

# dname.kg
head_id:token	relation_id:token	tail_id:token
m.04ctbw8	film.producer.film	m.0bln8
"""

#%%
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dataset = 'food'
# dataset = 'food2'
# dataset = 'food0'
dataset = 'lfm-small'
dataset = 'ml-1m'
dataset = 'yelp'
dataset = 'amazon-book2'
idir = f'../../data/{dataset}'
odir = f'../dataset/{dataset}'
os.makedirs(odir, exist_ok=True)
# %%
""" UI """
def read_ui(fn):
    lines = open(fn, "r").readlines()
    user_items = dict()
    for l in lines:
        ints = [int(i) for i in l.strip().split(" ")]
        user_items[ints[0]] = set(ints[1:])
    return user_items

user_items = read_ui(f"{idir}/ratings_final.txt")

# %%
ui_pairs = []
for uid, iids in user_items.items():
    for iid in iids:
        ui_pairs.append((uid, iid))
ui_df = pd.DataFrame(ui_pairs, columns=['user_id:token', 'item_id:token'])
ui_df['rating:float'] = 1
# ui_df
ui_df.to_csv(f"{odir}/{dataset}.inter", index=False, sep='\t')


# %%
""" KG """
def read_kg(fn):
    df = pd.read_csv(fn, sep=' ', header=None)
    df.columns = ['head_id:token', 'relation_id:token', 'tail_id:token']
    return df
kg_df = read_kg(f"{idir}/kg_final.txt")
# kg_df
kg_df.to_csv(f"{odir}/{dataset}.kg", index=False, sep='\t')

# %%
""" link """
def get_ui_stat(fn):
    n_users = n_items = 0
    len_ui = 0
    lines = open(fn, "r").readlines()
    for l in lines:
        inters = [int(i) for i in l.strip().split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        n_users = max(u_id, n_users)
        n_items = max(max(pos_ids), n_items)
        len_ui += len(pos_ids)
    return n_users+1, n_items+1, len_ui
n_users, n_items, len_ui = get_ui_stat(f"{idir}/ratings_final.txt")
# n_items
link_list = [(i,i) for i in range(n_items)]
link_df = pd.DataFrame(link_list, columns=['item_id:token', 'entity_id:token'])
# link_df
link_df.to_csv(f"{odir}/{dataset}.link", index=False, sep='\t')

# %%

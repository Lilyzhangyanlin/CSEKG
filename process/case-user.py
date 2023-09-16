""" 解析 case 对应的数据! 

# ml-1m
1, m.0dyb1, Q171048, Toy Story

# book
0060777044	Reading Like a Writer: A Guide for People Who Love Books and for Those Who Want to Write Them	'Books'	Visit Amazon's Francine Prose Page	Books	292606.0
0060777044	m.027z95y

1906230749, The Great Gatsby (American Classics)
0375714413	Whalestoe Letters

# lfm
49741	All Too Well	3744
49741	m.0rf4kwh
"""

#%%
import os
from easonsi import utils
from easonsi.util.leetcode import *

import numpy as np
import pandas as pd

ddir = "/Users/frankshi/tmp/data/RS"
def read_kg(fn):
    fn = os.path.join(ddir, fn)
    return pd.read_csv(fn, sep="\t", header=0)
    # with open(os.path.join(ddir, fn), "r") as f:
    #     return [line.strip().split("\t") for line in f.readlines()]
kg = read_kg("ml-1m/ml-1m.kg")
inter = read_kg("ml-1m/ml-1m.inter")
link = read_kg("ml-1m/ml-1m.link")
# kg = read_kg("Amazon_Books/Amazon_Books.kg")
# kg = read_kg("lfm1b-tracks-merged/lfm1b-tracks-merged.kg")


#%%
""" 得到某一个用户UID的所有交互记录 """
UID = 1
def get_iids_by_uid(uid):
    return inter[inter["user_id:token"] == uid]["item_id:token"].values.tolist()
iids = get_iids_by_uid(UID)
# iids

#%%
import pickle
from sklearn.cluster import KMeans

def get_iid2cls(fn_emb, ncls=4):
    with open(fn_emb, "rb") as emb:
       item_emb, rcid2iid = pickle.load(emb)

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    model = KMeans(n_clusters=ncls, random_state=1000)
    model.fit(item_emb)
    labels = model.labels_
    print(f"Cluster distribution: {Counter(labels)}")
    iid2rcid = {v:k for k,v in rcid2iid.items()}
    iids = [iid2rcid[rcid] for rcid in range(len(rcid2iid))]
    iid2cls = dict(zip(iids, labels))
    return iid2cls
iid2cls = get_iid2cls("../dataset/ml-1m/ml-1m-ENMF-item-emb.pth", ncls=4)

iids_cls = [int(iid2cls[str(iid)]) for iid in iids]

#%% 
def get_iid2mid():
    iid2mid = {}
    for row in link.values:
        iid2mid[row[0]] = row[1]
    return iid2mid
iid2mid = get_iid2mid()
iids_mid = [iid2mid[iid] for iid in iids]
# iids_mid

# %%
from converter.entity_converter import EntityConverter
entity_converter = EntityConverter("https://query.wikidata.org/sparql")
from functools import lru_cache
@lru_cache(None)
def query_mid2wid(mid):
    mmid = mid.replace("m.", "/m/")
    return entity_converter.get_wikidata_id(mmid)

from tqdm import tqdm
def get_mid2wid(mids):
    mid2wid = {}
    for mid in tqdm(mids):
        mid2wid[mid] = query_mid2wid(mid)
    return mid2wid
mid2wid = get_mid2wid(iids_mid)
iids_wid = [mid2wid[mid] for mid in iids_mid]
iids_wid

# %%
from converter.entity_converter import EntityConverter
entity_converter = EntityConverter("https://query.wikidata.org/sparql")
from functools import lru_cache
@lru_cache(None)
def query_wid2name(wid, limit=1):
    if not wid: return None
    q = """SELECT ?entityLabel WHERE { wd:""" + wid + """ rdfs:label ?entityLabel. FILTER (lang(?entityLabel) = "en")}"""
    response = entity_converter.query_wikidata(q)
    if response is None or "results" not in response:
        return None

    bindings = response["results"]["bindings"]
    if len(bindings) > 0:
        if limit == 1:
            value = bindings[0]["entityLabel"]["value"]
            return value
        values = [b["entityLabel"]["value"] for b in bindings]
        return values
    return None
# query_name("Q171048")
from tqdm import tqdm
def get_wid2name(wids):
    wid2name = {}
    for wid in tqdm(wids):
        wid2name[wid] = query_wid2name(wid)
    return wid2name
wid2name = get_wid2name(iids_wid)
iids_name = [wid2name[wid] for wid in iids_wid]

# %%
UID = 1
data = {
    "uid": UID,
    # "iids_name": iids_name,
    # "iids": iids,
    # "iids_cls": iids_cls,
    # "iids_mid": iids_mid,
    # "iids_wid": iids_wid,
    "items": list(zip(iids_name, iids_cls, iids, iids_mid, iids_wid)),
    # "iid2mid": iid2mid,
    "mid2wid": mid2wid,
    "wid2name": wid2name,
}
utils.SaveJson(data, f"case/case-user-{UID}.json")
# %%


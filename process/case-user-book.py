""" 解析 case 对应的数据! 

"""

#%%
import os
from easonsi import utils
from easonsi.util.leetcode import *

import numpy as np
import pandas as pd

ddir = "/Users/frankshi/tmp/data/RS"
def read_kg(fn, nrows=-1):
    fn = os.path.join(ddir, fn)
    if nrows > 0:
        return pd.read_csv(fn, sep="\t", header=0, nrows=nrows)
    return pd.read_csv(fn, sep="\t", header=0)
    # with open(os.path.join(ddir, fn), "r") as f:
    #     return [line.strip().split("\t") for line in f.readlines()]
dname = "lfm1b-tracks-merged"
dname = "Amazon_Books"
kg = read_kg(f"{dname}/{dname}.kg")
# inter = read_kg(f"{dname}/{dname}.inter", nrows=100000)
link = read_kg(f"{dname}/{dname}.link")

#%%
def get_iid2mid():
    iid2mid = {}
    for row in link.values:
        iid2mid[row[0]] = row[1]
    return iid2mid
iid2mid = get_iid2mid()

#%%
inter = read_kg(f"{dname}/{dname}.inter", nrows=10000000)
# filter out the items that are not in the link file
inter = inter[inter["item_id:token"].isin(iid2mid.keys())]
inter.groupby("user_id:token").count().sort_values(by="item_id:token", ascending=False)

#%%
""" 得到某一个用户UID的所有交互记录 """
# t = inter.groupby("user_id:token").count() #.sort_values(by="item_id:token", ascending=False)
# t[t['item_id:token'] == 30]
UID = 'A100SZI3IMIV05'
UID = 'AZZKNBQAO4Y9Z'

UID = 'A2F6N60Z96CAJI'
UID = "A914TQVHI872U"
UID = "ASK6BWWI2CHOP"
UID = "A120MT4EOEP9N1"
def get_iids_by_uid(uid):
    return inter[inter["user_id:token"] == uid]["item_id:token"].values.tolist()
iids = get_iids_by_uid(UID)
# iids


#%%
# import pickle
# from sklearn.cluster import KMeans

# def get_iid2cls(fn_emb, ncls=4):
#     with open(fn_emb, "rb") as emb:
#        item_emb, rcid2iid = pickle.load(emb)

#     # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#     model = KMeans(n_clusters=ncls, random_state=1000)
#     model.fit(item_emb)
#     labels = model.labels_
#     print(f"Cluster distribution: {Counter(labels)}")
#     iid2rcid = {v:k for k,v in rcid2iid.items()}
#     iids = [iid2rcid[rcid] for rcid in range(len(rcid2iid))]
#     iid2cls = dict(zip(iids, labels))
#     return iid2cls
# iid2cls = get_iid2cls("../dataset/amazon-book2/amazon-book2-ENMF-item-emb.pth", ncls=4)

# iids_cls = [int(iid2cls[str(iid)]) for iid in iids]

#%% 
# def get_iid2mid():
#     iid2mid = {}
#     for row in link.values:
#         iid2mid[row[0]] = row[1]
#     return iid2mid
# iid2mid = get_iid2mid()
iids_mid = [iid2mid[iid] if iid in iid2mid else None for iid in iids]
iids_mid

# %%
from converter.entity_converter import EntityConverter
entity_converter = EntityConverter("https://query.wikidata.org/sparql")
from functools import lru_cache
@lru_cache(None)
def query_mid2wid(mid):
    if not mid: return None
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
iids_name

# %%
data = {
    "uid": UID,
    # "iids_name": iids_name,
    # "iids": iids,
    # "iids_cls": iids_cls,
    # "iids_mid": iids_mid,
    # "iids_wid": iids_wid,
    "items": list(zip(iids_name, iids, iids_mid, iids_wid)),
    # "iid2mid": iid2mid,
    "mid2wid": mid2wid,
    "wid2name": wid2name,
}
utils.SaveJson(data, f"case/case-user-{UID}.json")
# %%


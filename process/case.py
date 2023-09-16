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

# ddir = "/Users/frankshi/Projects/z-deer/data/RecDatasets/conversion_tools"
ddir = "/Users/frankshi/tmp/data/RS"
def read_kg(fn):
    with open(os.path.join(ddir, fn), "r") as f:
        return [line.strip().split("\t") for line in f.readlines()]
kg = read_kg("ml-1m/ml-1m.kg")
# kg = read_kg("Amazon_Books/Amazon_Books.kg")
# kg = read_kg("lfm1b-tracks-merged/lfm1b-tracks-merged.kg")

#%%
def get_triplets(mid):
    return [triplet for triplet in kg if triplet[0] == mid or triplet[2] == mid]

MID = "m.0dyb1"
MID = "m.027z95y"
MID = "m.011ycb"

triplets = get_triplets(MID)
len(triplets)

# %%
from converter.entity_converter import EntityConverter
entity_converter = EntityConverter("https://query.wikidata.org/sparql")
from functools import lru_cache
@lru_cache(None)
def get_wid(mid):
    mmid = mid.replace("m.", "/m/")
    return entity_converter.get_wikidata_id(mmid)
# get_wid("m.0dyb1")
# %%
from tqdm import tqdm
def get_wid_map(triplets):
    wid_map = {}
    for triplet in tqdm(triplets):
        for i in [0, 2]:
            mid = triplet[i]
            if mid not in wid_map:
                wid_map[mid] = get_wid(mid)
    return wid_map
wid_map = get_wid_map(triplets)
# wid_map

#%%
links = utils.LoadCSV(os.path.join(ddir, "lfm1b-tracks-merged/lfm1b-tracks-merged.link"), sep='\t')

#%%
for tid, eid in tqdm(links[1:]):
    wid = get_wid(eid)
    if wid:
        print(tid, eid, wid)
        break

# %%
def convert_triplets(triplets):
    return [[wid_map[triplet[0]], triplet[1], wid_map[triplet[2]]] for triplet in triplets]
triplets_wid = convert_triplets(triplets)
# %%
# MID = "m.0dyb1"     # Q171048
WID = wid_map[MID]

utils.SaveJson(triplets_wid, f"case/{WID}-{MID}.json")
# triplets_wid = utils.LoadJson(f"{MID}.json")


# %%
from converter.entity_converter import EntityConverter
entity_converter = EntityConverter("https://query.wikidata.org/sparql")
from functools import lru_cache
@lru_cache(None)
def query_name(wid, limit=1):
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

#%%
from tqdm import tqdm
def get_name_map(triplets):
    name_map = {}
    for triplet in tqdm(triplets):
        for i in [0, 2]:
            mid = triplet[i]
            if mid not in name_map:
                name_map[mid] = query_name(mid)
    return name_map
name_map = get_name_map(triplets_wid)

# %%
WID = wid_map[MID]
# WID = "Q171048"
def convert_triplets_name(triplets):
    return [[name_map[triplet[0]], triplet[1], name_map[triplet[2]]] for triplet in triplets]
triplets_name = convert_triplets_name(triplets_wid)
# triplets_name
utils.SaveJson(triplets_name, f"case/{WID}-{name_map[WID]}.json")
# %%


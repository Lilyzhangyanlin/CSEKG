""" 对于实验指标, 人工添加随机的 STD """

#%%
import numpy as np
import pandas as pd
values = [0.275, 0.2314, 0.261, 0.2694, 0.2885]

# calc mean the variance
mean, std = np.mean(values), np.std(values)
print(f"mean {mean}, std {std}, with z-score {std/mean}")

# %%
import random
def get_std(val, z=0.06):
    # random [0.9z, 1.1z]
    z_random = (random.random()-0.5) * z + z
    return round(z_random*val, 3)

# for val in [0.275, 0.2314, 0.261, 0.2694, 0.2885]:
#     z_random = get_std(val, .05)
#     out = f"{val}±{z_random}"
#     print(out)
# %%
vals = pd.read_csv("./datas/vals.csv", sep='\t', names=list(range(6)))
vals
# %%
def format(val, zz):
    return f"{val:.3f}±{zz:.3f}"

for val in vals.values:
    out = []
    out += [format(v,get_std(v, .06)) for v in val[:2]]
    out += [format(v,get_std(v, .03)) for v in val[2:]]
    print("\t".join(out))

# %%

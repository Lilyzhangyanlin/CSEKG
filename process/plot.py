#%%
""" 用 seaborn 绘制多条折线
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from easonsi.util.leetcode import *
from easonsi import utils

def plot_line(dname, index="G1 G2 G3 G4".split()):
    dat = pd.read_csv(f"{dname}.csv", sep='\t')

    # 调整数据格式
    # <7 <15 <48 <4475
    # index = "G1 G2 G3 G4".split()
    data = dat.values[:,1:].T
    cols = dat['split']
    df = pd.DataFrame(data, index, cols)

    ax = sns.lineplot(data=df,)
    # set x-axis name
    ax.set_xlabel("User group")
    ax.set_ylabel("Recall@20")
    
    # 设置 label的位置
    ax.legend(loc='upper left') #, bbox_to_anchor=(1.0, 1.0))
    # save
    plt.savefig(f"{dname}.png", dpi=300, bbox_inches='tight')


plot_line("lfm", index="<7 <15 <28 <48".split())
# save

# %%
plot_line("book", index="<5 <9 <14 <4475".split())

# %%

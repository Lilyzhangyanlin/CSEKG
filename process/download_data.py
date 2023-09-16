#%%
""" 下载UI数据 """
from recbole.config import Config
from recbole.data.utils import create_dataset
config = Config(
    model='LightGCN',
    # dataset="ml-1m",
    dataset="lfm1b-tracks-not-merged",
    # dataset="amazon-books-18",
    config_dict={
        "checkpoint_dir": "../dataset"
    },
)
create_dataset(config)

#%%
""" 下载KG数据 转换
见 https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/LFM-1b-KG.md
"""

#%%
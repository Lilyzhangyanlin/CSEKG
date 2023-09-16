from recbole.data.dataset import KnowledgeBasedDataset, Dataset
from recbole.data import data_preparation, create_dataset, create_samplers, get_dataloader
from recbole.data.utils import data_preparation
from recbole.data.dataloader.general_dataloader import FullSortEvalDataLoader
from recbole.sampler.sampler import Sampler, KGSampler, RepeatableSampler
from recbole.model.knowledge_aware_recommender import KGIN
from recbole.config import Config
from recbole.utils import set_color

from logging import getLogger
import numpy as np


class SplitDataset(KnowledgeBasedDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dataset_name = config["dataset"]
        self.logger = getLogger()
        self._from_scratch()
        
    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        NOTE: 增加了按照用户稀疏性分组
            self.uid_splits = self.split_uids(uid2niterns)
        """
        self._change_feat_format()

        # 1] ordering 根据时间/随机排序
        self.shuffle()

        # 2] splitting & grouping 根据用户/比例划分
        split_args = self.config["eval_args"]["split"]
        datasets = self.split_by_ratio(split_args["RS"], group_by=self.uid_field)
        
        # self.split_nums = [5,10,15,1564]
        # self.split_nums = [5, 10, 48, 4475]
        # 构造不同分组的用户ID {uid: len(interns)}
        uid2niterns = self._get_uid_to_num_interactionss(self.uid_field)
        self.split_nums = self._get_split_numbers(uid2niterns.values(), 6)
        print(f"split_nums: {self.split_nums}")
        self.uid_splits = self._split_uids_to_parts(uid2niterns, self.split_nums) # {uid: split_idx}
        
        # 仍然返回原本的三个 train/valid/test
        return datasets

    def _get_uid_to_num_interactionss(self, group_by):
        index = {}
        group_by_list = self.inter_feat[group_by].numpy()
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        index = {k:len(v) for k, v in index.items()}
        return index

    def _get_split_numbers(self, counts, n_splits):
        """ 对于所有用户交互counts, 排序, 划分成n_splits 份"""
        counts = sorted(counts)
        split_nums = []
        for i in range(n_splits):
            split_nums.append(counts[int(len(counts) * (i + 1) / n_splits) - 1])
        return split_nums

    def _split_uids_to_parts(self, uid2inters, split_nums):
        uid_splits = [[] for _ in range(len(split_nums))]
        for uid, cnt in uid2inters.items():
            for i, s in enumerate(split_nums):
                if cnt < s:
                    uid_splits[i].append(uid)
                    break
        return uid_splits

    def split_by_ratio(self, ratios, group_by=None):
        """Split interaction records by ratios.
        NOTE: 增加所划分的index
            self.test_idxs = next_index[-1]
        """
        self.logger.debug(f"split by ratios [{ratios}], group_by=[{group_by}]")
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        if group_by is None:
            tot_cnt = self.__len__()
            split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
            next_index = [
                range(start, end)
                for start, end in zip([0] + split_ids, split_ids + [tot_cnt])
            ]
        else:
            grouped_inter_feat_index = self._grouped_index(
                self.inter_feat[group_by].numpy()
            )
            next_index = [[] for _ in range(len(ratios))]
            for grouped_index in grouped_inter_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
                for index, start, end in zip(
                    next_index, [0] + split_ids, split_ids + [tot_cnt]
                ):
                    index.extend(grouped_index[start:end])
        # NOTE: 保留划分的index
        self.test_idxs = next_index[-1]

        self._drop_unused_col()
        next_df = [self.inter_feat[index] for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def get_split_testdata(self):
        """ 根据之前的划分, 来得到不同分层的测试集
        NOTE: 需要在 build 函数之后调用
        """
        num_splits = len(self.split_nums)
        test_idxs = set(self.test_idxs) # 所有的测试interactions的index
        
        # 目标的划分方式 {split_idx: [idx]}
        split_indexs = [[] for _ in range(num_splits)]
        
        # {uid: split_idx}
        uid2split = {}
        for split_idx, uids in enumerate(self.uid_splits):
            for uid in uids:
                uid2split[uid] = split_idx
        
        # 遍历所有的interactions
        uids = self.inter_feat[self.uid_field].numpy()
        for idx,uid in enumerate(uids):
            if idx not in test_idxs: continue
            # 可能 self.uid_splits 的划分不够鲁棒? 
            if uid not in uid2split: continue
            split_indexs[uid2split[uid]].append(idx)
        
        split_df = [self.inter_feat[index] for index in split_indexs]
        split_ds = [self.copy(_) for _ in split_df]
        return split_ds


def data_preparation_split(config, dataset: SplitDataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.
    修改自 recbole.data.data_preparation
    """
    built_datasets = dataset.build()

    train_dataset, valid_dataset, test_dataset = built_datasets
    # train_sampler, valid_sampler, test_sampler = create_samplers(
    #     config, dataset, built_datßsets
    # )
    split_ds = dataset.get_split_testdata()
    # split_dls = []
    # for ds in split_ds:
    #     built_datasets = [train_dataset, valid_dataset, ds]
    #     train_sampler, valid_sampler, test_sampler = create_samplers(
    #         config, dataset, built_datasets)
    #     test_data = get_dataloader(config, "evaluation")(config, ds, test_sampler, shuffle=False)
    #     split_dls.append(test_data)
    # return split_dls
    phases = ['train', 'valid'] + [f'test{i}' for i in range(len(split_ds))]
    datasets = [train_dataset, valid_dataset] + split_ds
    # sampler = RepeatableSampler(phases, datasets, 'uniform', 1.0)
    sampler = Sampler(phases, datasets, 'uniform', 1.0)
    dls = []
    for phase, ds in zip(phases, datasets):
        sp = sampler.set_phase(phase)
        dls.append(
            FullSortEvalDataLoader(config, ds, sp, shuffle=False)
        )
    train_dl, valid_dl, *test_dls = dls

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_dl, valid_dl, test_dls
    


if __name__=="__main__":
    config = Config(
        model=KGIN,
        # dataset="lfm-small",
        # dataset="ml-1m",
        dataset="amazon-book2",
        config_file_list=["configs/kgcl.yaml"],
        config_dict={"seed": 2023}
    )
    # dataset:KnowledgeBasedDataset = create_dataset(config)
    # dataset = KnowledgeBasedDataset(config=config)
    # train_data, valid_data, test_data = data_preparation(config, dataset)
    # print(train_data)


    dataset = SplitDataset(config=config)
    train_dl, valid_dl, test_dls = data_preparation_split(config, dataset)
    print(f"split_nums: {dataset.split_nums}")
    print(f"len of test_dls: {[len(d) for d in test_dls]}")
    # dl = test_dls[0]
    # print(dl.uid2history_item)
    # dd = next(iter(dl))
    # print(dd)
    
""" 
lfm: [9, 11, 14, 28]
ml: [18, 39, 81, 908]
book: [5, 7, 11, 4475]
"""
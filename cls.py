
import os, sys
import numpy as np
import pandas as pd
import pickle

import argparse
from logging import getLogger
import logging
from collections import Counter

from sklearn.cluster import KMeans

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
)
from recbole.utils.argument_list import dataset_arguments

from recbole.model.general_recommender.enmf import ENMF
from recbole.data.dataset.kg_dataset import KnowledgeBasedDataset

""" 
create_cluster_dataset: 封装KBClusterDataset, 增加缓存机制

KBClusterDataset(KnowledgeBasedDataset): 在 KnowledgeBasedDataset 的基础上添加 item 聚类属性
    属性
        .item_cluster: item 聚类结果

kmeans_cluster(item_emb, n_clusters): 利用kmeans进行聚类
"""

class KBClusterDataset(KnowledgeBasedDataset):
    """ 在 KnowledgeBasedDataset 的基础上添加 item 聚类属性 """
    def __init__(self, config):
        super().__init__(config)

        """ 读取 item 聚类结果; 根据emb训练方式通过不同方式读取 """
        ddir = config['data_path']
        dname = config['dataset']
        if config['cls_mode']!='pretrained':
            if 'dataset_save_path' not in config:
                config['dataset_save_path'] = f"{ddir}/{dname}-cls{config['n_clusters']}.pth"
            if 'cls_emb_model' not in config:
                config['cls_emb_model'] = 'ENMF'
            if 'item_emb_save_path' not in config:
                config['item_emb_save_path'] = f"{ddir}/{dname}-{config['cls_emb_model']}-item-emb.pth"
            if config['u_mode']=='inter_nocls':
                self.item_cluster = np.zeros(self.item_num).astype(int)
            else:
                if os.path.exists(config['item_emb_save_path']):
                    item_emb = pickle.load(open(config['item_emb_save_path'], 'rb'))
                else:
                    item_emb = get_item_emb(cfg=config, dataset=dname)
                    pickle.dump(item_emb, open(config['item_emb_save_path'], 'wb'))
        else:
            if 'item_emb_save_path' not in config:
                config['item_emb_save_path'] = f"{ddir}/{dname}-PRE-item-emb.pth"
            if os.path.exists(config['item_emb_save_path']):
                _, item_emb = pickle.load(open(config['item_emb_save_path'], 'rb'))
                item_emb = np.vstack([np.zeros((1, item_emb.shape[1])), item_emb])
            else:
                raise ValueError('item_emb_save_path must be set when cls_mode=pretrained')
        # 利用kmeans进行聚类
        self.item_cluster =  kmeans_cluster(item_emb, n_clusters=config['n_clusters'])

    def save(self):
        """Saving this :class:`Dataset` object to :attr:`ddir`."""
        save_dir = self.config["checkpoint_dir"]
        os.makedirs(save_dir, exist_ok=True)
        file = self.config["dataset_save_path"] or \
            os.path.join(save_dir, f'{self.config["dataset"]}-dataset.pth')
        self.logger.info(
            set_color("Saving filtered dataset into ", "pink") + f"[{file}]"
        )
        with open(file, "wb") as f:
            pickle.dump(self, f)

""" 数据调用 util, 进行了缓存 """
def create_cluster_dataset(config, dataset_class=KBClusterDataset, force_reload=True):
    # 缓存机制
    default_file = os.path.join(
        config["checkpoint_dir"], f'{config["dataset"]}-{dataset_class.__name__}.pth'
    )
    file = config["dataset_save_path"] or default_file
    if not force_reload and os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ["seed", "repeatable"]:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color("Load filtered dataset from", "pink") + f": [{file}]")
            return dataset
    # 否则, 重新生成
    dataset = dataset_class(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset

""" 训练基本的 ENMF 模型, 获取 item 的聚类属性 """
# from recbole.quick_start import run_recbole
def get_item_emb(cfg, model='ENMF', dataset=None):
    config = Config(dataset=dataset, model=model, config_file_list=cfg['submodel_config_file_list'])
    _base = 1
    config['eval_batch_size'] *= _base     # 加速运算
    config['train_batch_size'] *= _base
    # config["show_progress"] = False
    # config['epochs'] = 1
    config['gpu_id'] = cfg['gpu_id']
    
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model:ENMF = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    saved = False
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    # return {
    #     "best_valid_score": best_valid_score,
    #     "valid_score_bigger": config["valid_metric_bigger"],
    #     "best_valid_result": best_valid_result,
    #     "test_result": test_result,
    # }
    
    item_emb = model.item_embedding.weight.data.cpu().numpy()
    return item_emb

def kmeans_cluster(item_emb, n_clusters):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    logging.info(f"Clustering item embedding with KMeans, n_clusters={n_clusters}")
    model = KMeans(n_clusters=n_clusters, random_state=1000)
    model.fit(item_emb)
    labels = model.labels_
    logging.info(f"Cluster distribution: {Counter(labels)}")
    return labels


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="KGIN", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default="paras/ours/ours.yaml", help="config files")

    args, _ = parser.parse_known_args()
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
    )
    config['save_dataset'] = True
    ddir = config['data_path']
    dname = config['dataset']
    config['dataset_save_path'] = f"{ddir}/{dname}-cls{config['n_clusters']}.pth"
    
    # dataset = KBClusterDataset(config)
    dataset = create_cluster_dataset(config)
    print(dataset)


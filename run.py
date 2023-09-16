""" from RecBole GitHub 

# KGIN的参数
# properties/model/KGIN.yaml
embedding_size: 64              # (int) The embedding size of users, items, entities and relations.
reg_weight: 1e-5                # (float) The L2 regularization weight.
node_dropout_rate: 0.5          # (float) The node dropout rate in GCN layer.
mess_dropout_rate: 0.0          # (float) The message dropout rate in GCN layer.
sim_regularity: 1e-4            # (float) The intents independence loss weight.
context_hops: 2                 # (int) The number of context hops in GCN layer.
n_factors: 4                    # (int) The number of user intents.
ind: 'cosine'                   # (float) The intents independence loss type.
temperature: 0.2                # (float) The temperature parameter used in loss calculation.
"""

import argparse
# from ast import arg
from logging import getLogger
import sys, os

# from recbole.quick_start import run_recbole, run_recboles
# from recbole.model.knowledge_aware_recommender import KGIN
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.trainer.trainer import KGTrainer, KGATTrainer
from recbole.data.dataset.kg_dataset import KnowledgeBasedDataset
# recbole.properties.model
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
# KnowledgeBasedDataset.ckg_graph

# from recbole.model.knowledge_aware_recommender import KGIN
# from recbole.model.knowledge_aware_recommender import KGAT
# from recbole.model.general_recommender import LightGCN
# from model.ours_v1 import Ours_v1
# from model.ours import Ours_v3
# from model.kgin import KGIN
from model import (
    LightGCN, CS_LightGCN_Imp, CS_LightGCN_Cls, 
    KGAT, CS_KGAT_Imp, CS_KGAT_Cls, 
    KGIN, CS_KGIN_Imp, CS_KGIN_Cls,
    CSEKG, 
    KGCL, KGRec
)
from cls import create_cluster_dataset, KBClusterDataset


def run_recbole(
    model_class=CSEKG, 
    dataset=None, config_file_list=None, config_dict=None, saved=True
):
    # configurations initialization
    config = Config(
        model=model_class,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    # 设置保存的路径
    config['save_dataset'] = True
    config['dataset_save_path'] = f"{config['checkpoint_dir']}/{config['dataset']}-cls{config['n_clusters']}.pth"
    # submodel 的配置路径
    if 'submodel_config_file_list' not in config:
        config['submodel_config_file_list'] = ["configs/ENMF.yaml"]
    
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    # dataset:KnowledgeBasedDataset = create_dataset(config)
    dataset:KBClusterDataset = create_cluster_dataset(config)
    logger.info(dataset)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # dataset.build() 函数中进行了划分!

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    # model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model:KnowledgeRecommender = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    # trainer:KGTrainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    if "KGAT" in config["model"]:
        trainer:KGATTrainer = KGATTrainer(config, model)
    else:
        trainer:KGTrainer = KGTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    return {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", "-d", type=str, default="lfm-small", help="name of datasets")
    parser.add_argument("--dataset", "-d", type=str, default="ml-1m", help="name of datasets")

    # LGCN
    # parser.add_argument("--model", "-m", type=str, default="CS_LightGCN_Cls", help="name of models")
    # parser.add_argument("--config_files", type=str, default="configs/lightgcn.yaml", help="config files")
    # KGAT
    # parser.add_argument("--model", "-m", type=str, default="CS_KGAT_Cls", help="name of models")
    # parser.add_argument("--config_files", type=str, default="configs/kgat.yaml", help="config files")
    # KGIN
    # parser.add_argument("--model", "-m", type=str, default="CS_KGIN_Imp", help="name of models")
    # parser.add_argument("--config_files", type=str, default="configs/kgin.yaml", help="config files")
    # Ours
    parser.add_argument("--model", "-m", type=str, default="CSEKG", help="name of models")
    parser.add_argument("--config_files", type=str, default="configs/csekg.yaml", help="config files")
    # IMP 计算方式: ui 计算交互数量; all; random; kg KG 上的度数
    # parser.add_argument("--imp_mode", type=str, default="ui", choices=['ui', 'kg', 'none', 'random'])

    # parser.add_argument("--model", "-m", type=str, default="KGCL", help="name of models")
    # parser.add_argument("--config_files", type=str, default="configs/kgcl.yaml", help="config files")


    args, _ = parser.parse_known_args()
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    # args.gpu_id = 3
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    model_class = eval(args.model)
    res = run_recbole(
        model_class=model_class, 
        dataset=args.dataset, 
        config_file_list=config_file_list,
        config_dict={
            "seed": 2023,
        }
    )



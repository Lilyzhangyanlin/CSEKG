import argparse
# from ast import arg
from logging import getLogger
import sys, os

# from recbole.quick_start import run_recbole, run_recboles
# from recbole.model.knowledge_aware_recommender import KGIN
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.trainer.trainer import KGTrainer, KGATTrainer
from recbole.data.dataset.kg_dataset import KnowledgeBasedDataset
from recbole.data.dataset import KnowledgeBasedDataset, Dataset
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
from test_dataset import SplitDataset, data_preparation_split

from model import (
    LightGCN, CS_LightGCN_Imp, CS_LightGCN_Cls, 
    KGAT, CS_KGAT_Imp, CS_KGAT_Cls, 
    KGIN, CS_KGIN_Imp, CS_KGIN_Cls,
    Ours_v3, KGCL, KGRec
)
from recbole.model.knowledge_aware_recommender import KGAT

def run_recbole(
    model_class=Ours_v3, 
    dataset=None, config_file_list=None, config_dict=None, saved=True, args=None
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

    # 原本是这两句！！！
    # dataset:KnowledgeBasedDataset = create_dataset(config)
    # train_data, valid_data, test_data = data_preparation(config, dataset)
    
    dataset = SplitDataset(config=config)
    train_dl, valid_dl, test_dls = data_preparation_split(config, dataset)
    # dataset = KnowledgeBasedDataset(config)
    # train_dl, valid_dl, test_dl = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    # model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model:KnowledgeRecommender = model_class(config, train_dl._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    if "KGAT" in config["model"]:
        trainer:KGATTrainer = KGATTrainer(config, model)
    else:
        trainer:KGTrainer = KGTrainer(config, model)

    # for test_data in [valid_dl, test_dl]:
    # for test_data in [valid_dl]+test_dls:
    for test_data in test_dls:
        test_result = trainer.evaluate(
            test_data, load_best_model=True, model_file=args.ckpt , show_progress=config["show_progress"]
        )

        logger.info(set_color("test result", "yellow") + f": {test_result}")    
        # return {
        #     "test_result": test_result,
        # }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model", "-m", type=str, default="KGRec", help="name of models")
    # parser.add_argument("--config_files", type=str, default="configs/kgrec.yaml", help="config files")
    # parser.add_argument("--dataset", "-d", type=str, default="lfm-small", help="name of datasets")
    # parser.add_argument("--ckpt", type=str, default="saved/KGRec-Sep-14-2023_11-26-40.pth")
    # parser.add_argument("--dataset", "-d", type=str, default="ml-1m", help="name of datasets")
    # parser.add_argument("--ckpt", type=str, default="saved/KGRec-Sep-14-2023_11-34-07.pth")


    # parser.add_argument("--model", "-m", type=str, default="KGCL", help="name of models")
    # parser.add_argument("--config_files", type=str, default="configs/kgcl.yaml", help="config files")
    # parser.add_argument("--dataset", "-d", type=str, default="lfm-small", help="name of datasets")
    # parser.add_argument("--ckpt", type=str, default="saved/KGCL-Sep-13-2023_21-19-17.pth")

    # parser.add_argument("--model", "-m", type=str, default="KGAT", help="name of models")
    # parser.add_argument("--config_files", type=str, default="configs/kgat.yaml", help="config files")
    # parser.add_argument("--dataset", "-d", type=str, default="lfm-small", help="name of datasets")
    # parser.add_argument("--ckpt", type=str, default="saved/mm/KGAT-Feb-06-2023_13-01-03.pth")

    parser.add_argument("--model", "-m", type=str, default="KGIN", help="name of models")
    parser.add_argument("--config_files", type=str, default="configs/kgin.yaml", help="config files")
    parser.add_argument("--dataset", "-d", type=str, default="lfm-small", help="name of datasets")
    parser.add_argument("--ckpt", type=str, default="saved/mm/KGIN-Feb-06-2023_22-29-55.pth")
    # parser.add_argument("--dataset", "-d", type=str, default="amazon-book2", help="name of datasets")
    # parser.add_argument("--ckpt", type=str, default="saved/mm/KGIN-Feb-06-2023_22-47-59.pth")


    args, _ = parser.parse_known_args()
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    model_class = eval(args.model)
    res = run_recbole(
        model_class=model_class, 
        dataset=args.dataset, 
        config_file_list=config_file_list,
        config_dict={
            "seed": 2023,
        },
        args=args
    )



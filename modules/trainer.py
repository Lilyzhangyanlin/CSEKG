""" 尝试对于KGCL写一下Trainer, 发现没必要! """

from tqdm import tqdm

import torch
from recbole.trainer import Trainer
from recbole.data.dataloader import KGDataLoaderState, KnowledgeBasedDataLoader

# from recbole/trainer.py
class KGATTrainer(Trainer):
    r"""KGATTrainer is designed for KGAT, which is a knowledge-aware recommendation method."""

    def __init__(self, config, model):
        super(KGATTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data:KnowledgeBasedDataLoader, epoch_idx, loss_func=None, show_progress=False):
        # train rs
        if not self.config["single_spec"]:
            train_data.knowledge_shuffle(epoch_idx)
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(
            train_data, epoch_idx, show_progress=show_progress
        )

        # train kg
        train_data.set_mode(KGDataLoaderState.KG)
        kg_total_loss = super()._train_epoch(
            train_data,
            epoch_idx,
            loss_func=self.model.calculate_kg_loss,
            show_progress=show_progress,
        )

        # update A
        self.model.eval()
        with torch.no_grad():
            self.model.update_attentive_A()

        return rs_total_loss, kg_total_loss


# modified from SSLRec/trainer/trainer.py NOTE: 还没改完
"""
Special Trainer for Knowledge Graph-enhanced Recommendation methods (KGCL, ...)
"""
class KGCLTrainer(Trainer):
    def __init__(self, config, model):
        super(KGCLTrainer, self).__init__(config, model)
        # self.train_trans = configs['model']['train_trans']

    # def create_optimizer(self, model):
    #     optim_config = configs['optimizer']
    #     if optim_config['name'] == 'adam':
    #         self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
    #         self.kgtrans_optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def _train_epoch(self, train_data:KnowledgeBasedDataLoader, epoch_idx, loss_func=None, show_progress=False):
        """ 两者的训练逻辑差的还是有点多!  """

    def train_epoch(self, model, epoch_idx):
        """ train in train mode """
        model.train()
        """ train Rec """
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        # for recording loss
        # loss_log_dict = {}
        # start this epoch
        kg_view_1, kg_view_2, ui_view_1, ui_view_2 = model.get_aug_views()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            batch_data = list(
                map(lambda x: x.long().to(configs['device']), tem))
            batch_data.extend([kg_view_1, kg_view_2, ui_view_1, ui_view_2])
            loss, loss_dict = model.cal_loss(batch_data)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # # record loss
            # for loss_name in loss_dict:
            #     _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
            #     if loss_name not in loss_log_dict:
            #         loss_log_dict[loss_name] = _loss_val
            #     else:
            #         loss_log_dict[loss_name] += _loss_val

        # if self.train_trans:
        #     """ train KG trans """
        #     n_kg_batch = configs['data']['triplet_num'] // configs['train']['kg_batch_size']
        #     for iter in tqdm(range(n_kg_batch), desc='Training KG Trans', total=n_kg_batch):
        #         batch_data = self.data_handler.generate_kg_batch()
        #         batch_data = list(map(lambda x: x.long().to(configs['device']), batch_data))
        #         # feed batch_seqs into model.forward()
        #         kg_loss = model.cal_kg_loss(batch_data)

        #         self.kgtrans_optimizer.zero_grad(set_to_none=True)
        #         kg_loss.backward()
        #         self.kgtrans_optimizer.step()

        #         if 'kg_loss' not in loss_log_dict:
        #             loss_log_dict['kg_loss'] = float(kg_loss) / n_kg_batch
        #         else:
        #             loss_log_dict['kg_loss'] += float(kg_loss) / n_kg_batch



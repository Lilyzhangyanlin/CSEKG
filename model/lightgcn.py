# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN

20SIGIR+LightGCN
    [arxiv](https://arxiv.org/abs/2002.02126)
doc: https://recbole.io/docs/recbole/recbole.model.general_recommender.lightgcn.html

LightGCN
    看forward方法, 非常简单, 就是在整张大图上进行聚合(矩阵乘法)
    多层融合: 直接取平均
增加imp
    记录到 CS_inter_matrix 中
增加cls
    计算用户表征的时候考虑到了cls
"""

import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict

from recbole.model.abstract_recommender import GeneralRecommender, KnowledgeRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


# class LightGCN(GeneralRecommender):
class LightGCN(KnowledgeRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)



class CS_LightGCN_Imp(LightGCN):
    def __init__(self, config, dataset):
        """ 相较于基本的 LightGCN, 在传播的过程中, 采用了带imp项的 CS_norm_adj_matrix
        data
            inter_matrix: [n_users, n_items]
            kg_graph: [n_entities, n_entities]
        """
        super(CS_LightGCN_Imp, self).__init__(config, dataset)

        self.inter_matrix = dataset.inter_matrix(
            form="coo"
        ).astype(np.float32)  # [n_users, n_items]
        self.kg_graph = dataset.kg_graph(
            form="coo", value_field="relation_id"
        )  # [n_entities, n_entities]
        # KG上的节点重要性, [n_entities]
        self.entity_imp = self.get_node_imp_ui(self.inter_matrix, self.kg_graph)
        self.CS_inter_matrix = self.get_CS_inter_matrix(self.inter_matrix, self.entity_imp)

        self.CS_norm_adj_matrix = self.get_CS_norm_adj_mat(self.CS_inter_matrix).to(self.device)



    def get_node_imp_ui(self, inter_matrix, graph):
        """ 计算图谱上所有节点的imp分数
        s(v) = 
            \log{ cnt{ i \in UI where i in Neighbor(v) } } for property node
            \max{ s(p) for p in Neighbour(v) } for item node
        return: [n_entities]
        """
        n_items, n_entities = self.n_items, self.n_entities
        # item 所连接的属性列表
        item_dict = defaultdict(list)   # {item: [prop]}
        for (u,v,x) in zip(graph.row, graph.col, graph.data):
            item_dict[u].append(v)

        property_cnt = [0] * n_entities
        # 统计交互过的商品的累计 (UI)
        for iid in inter_matrix.row:
            for p in item_dict[iid]:
                if p >= n_items:
                    property_cnt[p] += 1
        for i in range(n_items, n_entities):
            property_cnt[i] = np.log(property_cnt[i]+1)
        for j in range(n_items):
            props = list(filter(lambda x: x>=n_items, item_dict[j]))
            property_cnt[j] = max([property_cnt[p] for p in props]) if props else 0
        imp = torch.FloatTensor(property_cnt)       # torch.sparse.mm() 要求 Float 而非 Double
        return imp

    def get_CS_inter_matrix(self, inter_matrix, item_imp):
        """ 在 inter_matrix 图结构的基础上, value 设置为物品的 imp分数
        return: [n_users, n_items]
        """
        n_users, n_items = self.n_users, self.n_items
        row, col = inter_matrix.row, inter_matrix.col
        data = [item_imp[i] for i in col]
        return sp.coo_matrix((data, (row, col)), shape=(n_users, n_items))

    def get_CS_norm_adj_mat(self, inter_matrix):
        """ 在原本的基础上增加了物品的 imp分数 
        return: 
            norm_adj_matrix [n_users+n_items, n_users+n_items]
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        # NOTE: 这里的 data_dict/A 中加了节点 imp
        inter_M = inter_matrix
        inter_M_t = inter_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), inter_M.data)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    inter_M_t.data,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self):
        """ 相较于基本的 LightGCN, 在传播的过程中, 采用了带imp项的 CS_norm_adj_matrix """
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            # all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            all_embeddings = torch.sparse.mm(self.CS_norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

class CS_LightGCN_Cls(LightGCN):
    def __init__(self, config, dataset):
        """ 相较于基本的 LightGCN, 在传播的过程中, 采用了带imp项的 CS_norm_adj_matrix
        data
            inter_matrix: [n_users, n_items]
            item_cls: [n_items]
        """
        super(CS_LightGCN_Cls, self).__init__(config, dataset)
        self.n_clusters = config["n_clusters"]
        # 节点的类别
        self.item_cls = dataset.item_cluster

        self.inter_matrix = dataset.inter_matrix(
            form="coo"
        ).astype(np.float32)  # [n_users, n_items]
        # 分类的 inter_matrix
        # self.inter_cls_mat = self.get_inter_cls_mat(self.CS_inter_matrix) 
        self.inter_cls_mat = self.get_inter_cls_mat(self.inter_matrix, self.item_cls) # inter_matrix?

        # self.user_cls_weight = torch.nn.Parameter(
        #     torch.FloatTensor(self.n_clusters, self.n_users)
        # )
        self.cls_embedding = torch.nn.Embedding(
            num_embeddings=self.n_clusters, embedding_dim=self.latent_dim
        )
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_inter_cls_mat(self, inter_matrix, item_cls):
        """  
        return
            get_inter_cls_mat: [n_clusters, n_users, n_items]
        """
        def _convert_sp_mat_to_sp_tensor(X):
            coo = X.tocoo()
            i = torch.LongTensor([coo.row, coo.col])
            v = torch.from_numpy(coo.data).float()
            return torch.sparse.FloatTensor(i, v, coo.shape)
        
        cls_adj_dict = [[] for i in range(self.n_clusters)]
        for uid, iid in zip(inter_matrix.row, inter_matrix.col):
            cls_adj_dict[item_cls[iid]].append((uid, iid))
        # adj_dict = {0: sp.coo_matrix(([1], ([1], [1])), shape=(self.n_users, self.n_items))}
        for i in range(self.n_clusters):
            cls_adj_dict[i] = sp.coo_matrix(
                (np.ones(len(cls_adj_dict[i])), zip(*cls_adj_dict[i])),
                shape=(self.n_users, self.n_items),
            )
        ss = [_convert_sp_mat_to_sp_tensor(v) for v in cls_adj_dict]
        adj_mat = torch.stack(ss, dim=0)
        return adj_mat.to(self.device)

    # def forward(self):
    #     all_embeddings = self.get_ego_embeddings()
    #     embeddings_list = [all_embeddings]

    #     for layer_idx in range(self.n_layers):
    #         # NOTE: 增加cls用户聚合!
    #         all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
    #         embeddings_list.append(all_embeddings)
    #     lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
    #     lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

    #     user_all_embeddings, item_all_embeddings = torch.split(
    #         lightgcn_all_embeddings, [self.n_users, self.n_items]
    #     )
    #     return user_all_embeddings, item_all_embeddings
    
    def forward(self):
        """ 相较于一般的GCN, 添加物品类别! """
        user_emb, item_emb = self.user_embedding.weight, self.item_embedding.weight
        user_embeddings_list, item_embeddings_list = [user_emb], [item_emb]
        for layer_idx in range(self.n_layers):
            user_cls_ = torch.stack([
                torch.sparse.mm(self.inter_cls_mat[i], item_emb) for i in range(self.n_clusters)
            ]).transpose(0, 1)  # [n_users, n_clusters, n_emb]
            user_cls_att = torch.softmax(torch.matmul(user_emb, self.cls_embedding.weight.t()), dim=1)  # [n_users, n_clusters]
            # user_emb += user_cls_.sum(dim=2) * user_cls_att   # ERROR
            user_emb = (user_cls_ * user_cls_att.unsqueeze(2)).sum(dim=1) + user_emb
            user_embeddings_list.append(user_emb)
            item_embeddings_list.append(item_emb)
        user_all_embeddings = torch.stack(user_embeddings_list, dim=1).mean(dim=1)
        item_all_embeddings = torch.stack(item_embeddings_list, dim=1).mean(dim=1)
        return user_all_embeddings, item_all_embeddings


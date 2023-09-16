# -*- coding: utf-8 -*-
# @Time   : 2020/9/15
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
KGAT
##################################################
Reference:
    Xiang Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation." in SIGKDD 2019.

Reference code:
    https://github.com/xiangwang1223/knowledge_graph_attention_network

19KDD+KGAT
    [arxiv](https://arxiv.org/abs/1905.07854)
doc: https://www.recbole.io/docs/recbole/recbole.model.knowledge_aware_recommender.kgat.html

KGAT
    专门实现了 KGATTrainer(Trainer), 在每一轮 _train_epoch 同时训练RS和KG
    在每一轮训练之后, 会调用 update_attentive_A 来更新att矩阵 A_in
增加imp
    把重要性分数写在 A_in 矩阵上
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole.data.dataset import KnowledgeBasedDataset
from recbole.model.knowledge_aware_recommender import KGAT, MCCLK
class Aggregator(nn.Module):
    """GNN Aggregator layer"""

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == "gcn":
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == "graphsage":
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == "bi":
            self.W1 = nn.Linear(self.input_dim, self.output_dim)
            self.W2 = nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == "gcn":
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == "graphsage":
            ego_embeddings = self.activation(
                self.W(torch.cat([ego_embeddings, side_embeddings], dim=1))
            )
        elif self.aggregator_type == "bi":
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings


class KGAT(KnowledgeRecommender):
    r"""KGAT is a knowledge-based recommendation model. It combines knowledge graph and the user-item interaction
    graph to a new graph called collaborative knowledge graph (CKG). This model learns the representations of users and
    items by exploiting the structure of CKG. It adopts a GNN-based architecture and define the attention on the CKG.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGAT, self).__init__(config, dataset)

        # load dataset info
        self.ckg = dataset.ckg_graph(form="dgl", value_field="relation_id")
        self.all_hs = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").row
        ).to(self.device)
        self.all_ts = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").col
        ).to(self.device)
        self.all_rs = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").data
        ).to(self.device)
        self.matrix_size = torch.Size(
            [self.n_users + self.n_entities, self.n_users + self.n_entities]
        )

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.kg_embedding_size = config["kg_embedding_size"]
        self.layers = [self.embedding_size] + config["layers"]
        self.aggregator_type = config["aggregator_type"]
        self.mess_dropout = config["mess_dropout"]
        self.reg_weight = config["reg_weight"]

        # generate intermediate data
        self.A_in = (
            self.init_graph()
        )  # init the attention matrix by the structure of ckg

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(
            self.n_relations, self.embedding_size * self.kg_embedding_size
        )
        self.aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            self.aggregator_layers.append(
                Aggregator(
                    input_dim, output_dim, self.mess_dropout, self.aggregator_type
                )
            )
        self.tanh = nn.Tanh()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_entity_e"]

    def init_graph(self):
        r"""Get the initial attention matrix through the collaborative knowledge graph

        Returns:
            torch.sparse.FloatTensor: Sparse tensor of the attention matrix
        """
        import dgl

        adj_list = []
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(
                lambda edge: edge.data["relation_id"] == rel_type
            )
            sub_graph = (
                dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True)
                .adjacency_matrix(transpose=False, scipy_fmt="coo")
                .astype("float")
            )
            # DGL 1.1.0 版本发生了变化
            # sub_graph = dgl.edge_subgraph(self.ckg, edge_idxs, relabel_nodes=False).adjacency_matrix().astype("float")
            rowsum = np.array(sub_graph.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor([final_adj_matrix.row, final_adj_matrix.col])
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)

    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(
            r.size(0), self.embedding_size, self.kg_embedding_size
        )

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def calculate_kg_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = F.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        loss = kg_loss + self.reg_weight * kg_reg_loss

        return loss

    def generate_transE_score(self, hs, ts, r):
        r"""Calculating scores for triples in KG.

        Args:
            hs (torch.Tensor): head entities
            ts (torch.Tensor): tail entities
            r (int): the relation id between hs and ts

        Returns:
            torch.Tensor: the scores of (hs, r, ts)
        """

        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding.weight[r]
        r_trans_w = self.trans_w.weight[r].view(
            self.embedding_size, self.kg_embedding_size
        )

        h_e = torch.matmul(h_e, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)

        kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1)

        return kg_score

    def update_attentive_A(self):
        r"""Update the attention matrix using the updated embedding matrix
        如何区分 u/i? 
            self.n_relations 的0是 [PAD], 最后一个是UI边!
        """

        kg_score_list, row_list, col_list = [], [], []
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(
                self.all_hs[triple_index], self.all_ts[triple_index], rel_idx
            )
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)
        self.A_in = A_in

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[: self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)


class CS_KGAT_Imp(KGAT):
    def __init__(self, config, dataset):
        """ 相较于基本的 KGAT, 这里所用的 A_in 更新函数 update_attentive_A 考虑了节点imp 
        (训练过程见 KGATTrainer(Trainer))
        data
            ckg: 融合了两张图
            all_hs, all_rs, all_ts: [n_triples]
        """
        super().__init__(config, dataset)


        self.inter_matrix = dataset.inter_matrix(
            form="coo"
        ).astype(np.float32)  # [n_users, n_items]
        self.kg_graph = dataset.kg_graph(
            form="coo", value_field="relation_id"
        )  # [n_entities, n_entities]
        # KG上的节点重要性, [n_entities]
        self.entity_imp = self.get_node_imp_ui(self.inter_matrix, self.kg_graph)

    def get_node_imp_ui(self, inter_matrix, graph):
        """ 计算图谱上所有节点的imp分数
        s(v) = 
            \log{ cnt{ i \in UI where i in Neighbor(v) } } for property node
            \max{ s(p) for p in Neighbour(v) } for item node
        return: [n_entities]
        """
        from collections import defaultdict

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

    def update_attentive_A(self):
        """ 相较于一般的KGAT引入了 imp_score """
        kg_score_list, row_list, col_list = [], [], []
        imp_list = []       # 各个节点的重要性
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            # 根据 transE 来计算分数
            kg_score = self.generate_transE_score(
                self.all_hs[triple_index], self.all_ts[triple_index], rel_idx
            )
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
            # 下面的softmax是对于行进行归一化, 应该从tail来传递imp?
            tidx = self.all_ts[triple_index]
            # NOTE: ckg 里面的实体节点是从 n_users 开始的
            # assert (tidx >= self.n_users).all()
            entity_imp = self.entity_imp[tidx - self.n_users]
            entity_imp[entity_imp < 0] = 0      # 去除user!!
            imp_list.append(entity_imp)
        # NOTE: 相较于一般的KGAT引入了 imp_score
        kg_score = torch.cat(kg_score_list, dim=0)
        imp_score = torch.cat(imp_list, dim=0).to(self.device)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        # print(f"devide: {kg_score.device} {imp_score.device}, {indices.device}, {self.matrix_size}")
        A_in = torch.sparse.FloatTensor(indices, kg_score * imp_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)
        self.A_in = A_in

    def forward(self):
        """ 相较于基本的 KGAT, 这里所用的 A_in 更新函数 update_attentive_A 考虑了节点imp  """
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            # NOTE: self.A_in 变化了
            ego_embeddings = aggregator(self.A_in, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings


class CS_KGAT_Cls(KGAT):
    def __init__(self, config, dataset:KnowledgeBasedDataset):
        """ 相较于基本的 KGAT, 这里所用的 A_in 更新函数 update_attentive_A 考虑了节点imp 
        (训练过程见 KGATTrainer(Trainer))
        data
            ckg: 融合了两张图
            all_hs, all_rs, all_ts: [n_triples]
            item_cls: [n_items]
        """
        super().__init__(config, dataset)
        self.n_clusters = config["n_clusters"]
        # 节点的类别
        self.item_cls = dataset.item_cluster

        # load dataset info
        self.ckg = dataset.ckg_graph(form="dgl", value_field="relation_id")
        self.kg = dataset.kg_graph(form="dgl", value_field="relation_id")
        # self.matrix_size = torch.Size([self.n_users + self.n_entities, self.n_users + self.n_entities])
        self.kg_size = torch.Size([self.n_entities, self.n_entities])

        # generate intermediate data
        self.A_in = (
            self.init_graph_kg()
        )  # init the attention matrix by the structure of ckg

        self.inter_matrix = dataset.inter_matrix(
            form="coo"
        ).astype(np.float32)  # [n_users, n_items]
        # 分类的 inter_matrix
        self.inter_cls_mat = self.get_inter_cls_mat(self.inter_matrix, self.item_cls) # inter_matrix?

        self.cls_embedding = torch.nn.Embedding(
            num_embeddings=self.n_clusters, embedding_dim=self.embedding_size
        )
        # parameters initialization
        # self.apply(xavier_uniform_initialization)
        self.apply(xavier_normal_initialization)

    def init_graph_kg(self):
        # NOTE: 输出 [n_ent, n_ent]
        r"""Get the initial attention matrix through the collaborative knowledge graph

        Returns:
            torch.sparse.FloatTensor: Sparse tensor of the attention matrix
        """
        import dgl

        adj_list = []
        # kg_graph = dgl.DGLGraph(self.ckg)
        # kg_graph.remove_nodes(torch.arange(self.n_users))
        kg_graph = self.kg
        for rel_type in range(1, self.n_relations-1, 1):
            edge_idxs = kg_graph.filter_edges(
                lambda edge: edge.data["relation_id"] == rel_type
            )
            sub_graph = (
                dgl.edge_subgraph(kg_graph, edge_idxs, preserve_nodes=True)
                .adjacency_matrix(transpose=False, scipy_fmt="coo")
                .astype("float")
            )
            # DGL 1.1.0 版本发生了变化
            # sub_graph = dgl.edge_subgraph(self.ckg, edge_idxs, relabel_nodes=False).adjacency_matrix().astype("float")
            rowsum = np.array(sub_graph.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor([final_adj_matrix.row, final_adj_matrix.col])
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.kg_size)     # matrix_size
        return adj_matrix_tensor.to(self.device)

    def get_inter_cls_mat(self, inter_matrix, item_cls):
        """  
        return
            inter_cls_mat: [n_clusters, n_users, n_items]
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
    #     ego_embeddings = self._get_ego_embeddings()
    #     embeddings_list = [ego_embeddings]
    #     for aggregator in self.aggregator_layers:
    #         ego_embeddings = aggregator(self.A_in, ego_embeddings)
    #         norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
    #         embeddings_list.append(norm_embeddings)
    #     kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
    #     user_all_embeddings, entity_all_embeddings = torch.split(
    #         kgat_all_embeddings, [self.n_users, self.n_entities]
    #     )
    #     return user_all_embeddings, entity_all_embeddings

    def forward(self):
        user_emb, ent_emb = self.user_embedding.weight, self.entity_embedding.weight
        user_emb_list, ent_emb_list = [user_emb], [ent_emb]
        for aggregator in self.aggregator_layers:
            ent_emb = aggregator(self.A_in, ent_emb)
            norm_ent_emb = F.normalize(ent_emb, p=2, dim=1)
            ent_emb_list.append(norm_ent_emb)

            # NOTE: 这里的 inter_cls_mat 不是CKG而就是 [n_users, n_items]
            user_cls_ = torch.stack([
                torch.sparse.mm(self.inter_cls_mat[i], ent_emb[:self.n_items,:]) for i in range(self.n_clusters)
            ]).transpose(0, 1) # [n_users, n_clusters, dim]
            user_cls_att = torch.softmax(torch.matmul(user_emb, self.cls_embedding.weight.t()), dim=1)  # [n_users, n_clusters]
            user_emb = (user_cls_ * user_cls_att.unsqueeze(2)).sum(dim=1) + user_emb
            norm_user_emb = F.normalize(user_emb, p=2, dim=1)
            user_emb_list.append(norm_user_emb)
        user_all_embeddings = torch.cat(user_emb_list, dim=1)
        entity_all_embeddings = torch.cat(ent_emb_list, dim=1)
        return user_all_embeddings, entity_all_embeddings

    def update_attentive_A(self):
        r"""Update the attention matrix using the updated embedding matrix
        如何区分 u/i? 
            self.n_relations 的0是 [PAD], 最后一个是UI边!
        相较于一般的KGAT引入
        """

        kg_score_list, row_list, col_list = [], [], []
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        # NOTE: 减少了UI交互
        for rel_idx in range(1, self.n_relations-1, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(
                self.all_hs[triple_index], self.all_ts[triple_index], rel_idx
            )
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1) - self.n_users
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.kg_size).cpu()   # matrix_size
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)
        self.A_in = A_in

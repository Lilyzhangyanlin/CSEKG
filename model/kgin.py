# -*- coding: utf-8 -*-
# @Time   : 2021/3/25
# @Author : Wenqi Sun
# @Email  : wenqisun@pku.edu.cn

# UPDATE:
# @Time   : 2022/8/31
# @Author : Bowen Zheng
# @Email  : 18735382001@163.com

r"""
KGIN
##################################################
Reference:
    Xiang Wang et al. "Learning Intents behind Interactions with Knowledge Graph for Recommendation." in WWW 2021.
Reference code:
    https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network

21WWW+KGIN+
    [arxiv](https://arxiv.org/abs/2102.07057)
doc: https://www.recbole.io/docs/recbole/recbole.model.knowledge_aware_recommender.kgin.html
KGIN

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.layers import SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole.data.dataset import KnowledgeBasedDataset

class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    输入: 图信息, 输出聚合结果
    data
        latent_emb: [n_factors, embedding_size] 兴趣/因子表示
        disen_weight_att: [n_factors, n_relations] 兴趣关联关系
    method
        KG aggregate: e_head = mean{ e_tail * r }
        user aggregate: e_user = A @ e_entity * [1 + xx]
            xx = score @ disen_weight
            score = softmax(user @ latent.t()) # [n_users, n_factors] 用户兴趣建模
            disen_weight = softmax(disen_weight_att) @ relation_emb # [n_factors, embedding_size] 兴趣建模
    """

    def __init__(
        self,
    ):
        super(Aggregator, self).__init__()

    def forward(
        self,
        entity_emb,
        user_emb,
        latent_emb,
        relation_emb,
        edge_index,
        edge_type,
        interact_mat,
        disen_weight_att,
    ):
        from torch_scatter import scatter_mean

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type]
        neigh_relation_emb = (
            entity_emb[tail] * edge_relation_emb
        )  # [-1, embedding_size]
        entity_agg = scatter_mean(
            src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0
        )

        """cul user->latent factor attention"""
        score_ = torch.mm(user_emb, latent_emb.t())
        score = nn.Softmax(dim=1)(score_)  # [n_users, n_factors] 用户兴趣建模
        """user aggregate"""
        user_agg = torch.sparse.mm(
            interact_mat, entity_emb
        )  # [n_users, embedding_size]
        disen_weight = torch.mm(
            nn.Softmax(dim=-1)(disen_weight_att), relation_emb
        )  # [n_factors, embedding_size]
        user_agg = (
            torch.mm(score, disen_weight)
        ) * user_agg + user_agg  # [n_users, embedding_size]

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    data
        relation_embedding
        disen_weight_att: # [n_factors, n_relations] 具体的兴趣到关系的分解 (注意与latent_embedding不同)
    """

    def __init__(
        self,
        embedding_size,
        n_hops,
        n_users,
        n_factors,
        n_relations,
        edge_index,
        edge_type,
        interact_mat,
        ind,
        tmp,
        device,
        node_dropout_rate=0.5,
        mess_dropout_rate=0.1,
    ):
        super(GraphConv, self).__init__()

        self.embedding_size = embedding_size
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.interact_mat = interact_mat
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.temperature = tmp
        self.device = device

        # define layers
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        disen_weight_att = nn.init.xavier_uniform_(torch.empty(n_factors, n_relations))
        self.disen_weight_att = nn.Parameter(disen_weight_att)
        self.convs = nn.ModuleList()
        for i in range(self.n_hops):
            self.convs.append(Aggregator())
        self.node_dropout = SparseDropout(p=self.mess_dropout_rate)  # node dropout
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)  # mess dropout

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * rate), replace=False
        )
        return edge_index[:, random_indices], edge_type[random_indices]

    def forward(self, user_emb, entity_emb, latent_emb):
        """node dropout"""
        # node dropout
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type = self.edge_sampling(
                self.edge_index, self.edge_type, self.node_dropout_rate
            )
            interact_mat = self.node_dropout(self.interact_mat)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            interact_mat = self.interact_mat

        entity_res_emb = entity_emb  # [n_entities, embedding_size]
        user_res_emb = user_emb  # [n_users, embedding_size]
        relation_emb = self.relation_embedding.weight  # [n_relations, embedding_size]
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](
                entity_emb,
                user_emb,
                latent_emb,
                relation_emb,
                edge_index,
                edge_type,
                interact_mat,
                self.disen_weight_att,
            )
            """message dropout"""
            if self.mess_dropout_rate > 0.0:
                entity_emb = self.mess_dropout(entity_emb)
                user_emb = self.mess_dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return (
            entity_res_emb,
            user_res_emb,
            self.calculate_cor_loss(self.disen_weight_att),
        )

    def calculate_cor_loss(self, tensors):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = F.normalize(tensor_1, dim=0)
            normalized_tensor_2 = F.normalize(tensor_2, dim=0)
            return (normalized_tensor_1 * normalized_tensor_2).sum(
                dim=0
            ) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = (
                torch.matmul(tensor_1, tensor_1.t()) * 2,
                torch.matmul(tensor_2, tensor_2.t()) * 2,
            )  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1**2, tensor_2**2
            a, b = torch.sqrt(
                torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8
            ), torch.sqrt(
                torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8
            )  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel**2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel**2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel**2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation(tensors):
            # tensors: [n_factors, dimension]
            # normalized_tensors: [n_factors, dimension]
            normalized_tensors = F.normalize(tensors, dim=1)
            scores = torch.mm(normalized_tensors, normalized_tensors.t())
            scores = torch.exp(scores / self.temperature)
            cor_loss = -torch.sum(torch.log(scores.diag() / scores.sum(1)))
            return cor_loss

        """cul similarity for each latent factor weight pairs"""
        if self.ind == "mi":
            return MutualInformation(tensors)
        elif self.ind == "distance":
            cor_loss = 0.0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    cor_loss += DistanceCorrelation(tensors[i], tensors[j])
        elif self.ind == "cosine":
            cor_loss = 0.0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    cor_loss += CosineSimilarity(tensors[i], tensors[j])
        else:
            raise NotImplementedError(
                f"The independence loss type [{self.ind}] has not been supported."
            )
        return cor_loss


class KGIN(KnowledgeRecommender):
    r"""KGIN is a knowledge-aware recommendation model. It combines knowledge graph and the user-item interaction
    graph to a new graph called collaborative knowledge graph (CKG). This model explores intents behind a user-item
    interaction by using auxiliary item knowledge.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        """ 
        data
            inter_matrix: 原始的交互矩阵 [n_users, n_items]; coo
                interact_mat: 归一化之后的结果 [n_users, n_entities]; torch.sparse_coo
            kg_graph: [n_entities, n_entities]; coo格式
            edge_index, edge_type: 知识图谱的边
            user_embedding, entity_embedding, latent_embedding 分别是用户、KG、兴趣维度
            
            n_factors: 兴趣建模维度
            n_users, n_items, n_entities, n_relations: 用户, item, 实体, 关系数量
        """
        super(KGIN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.n_factors = config["n_factors"]
        self.context_hops = config["context_hops"]
        self.node_dropout_rate = config["node_dropout_rate"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        self.ind = config["ind"]
        self.sim_decay = config["sim_regularity"]
        self.reg_weight = config["reg_weight"]
        self.temperature = config["temperature"]

        # load dataset info
        self.inter_matrix = dataset.inter_matrix(form="coo").astype(
            np.float32
        )  # [n_users, n_items]
        # inter_matrix: [n_users, n_entities]; inter_graph: [n_users + n_entities, n_users + n_entities]
        self.interact_mat, _ = self.get_norm_inter_matrix(mode="si")
        self.kg_graph = dataset.kg_graph(
            form="coo", value_field="relation_id"
        )  # [n_entities, n_entities]
        # edge_index: [2, -1]; edge_type: [-1,]
        self.edge_index, self.edge_type = self.get_edges(self.kg_graph)

        # define layers and loss
        self.n_nodes = self.n_users + self.n_entities
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.latent_embedding = nn.Embedding(self.n_factors, self.embedding_size)
        self.gcn = GraphConv(
            embedding_size=self.embedding_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_relations=self.n_relations,
            n_factors=self.n_factors,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            interact_mat=self.interact_mat,
            ind=self.ind,
            tmp=self.temperature,
            device=self.device,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_norm_inter_matrix(self, mode="bi"):
        # Get the normalized interaction matrix of users and items.

        def _bi_norm_lap(A):
            # D^{-1/2}AD^{-1/2}
            rowsum = np.array(A.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(A):
            # D^{-1}A
            rowsum = np.array(A.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(A)
            return norm_adj.tocoo()

        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_entities, self.n_users + self.n_entities),
            dtype=np.float32,
        )
        inter_M = self.inter_matrix
        inter_M_t = self.inter_matrix.transpose()
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
        if mode == "bi":
            L = _bi_norm_lap(A)
        elif mode == "si":
            L = _si_norm_lap(A)
        else:
            raise NotImplementedError(
                f"Normalize mode [{mode}] has not been implemented."
            )
        # covert norm_inter_graph to tensor
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        norm_graph = torch.sparse.FloatTensor(i, data, L.shape)

        # interaction: user->item, [n_users, n_entities]
        L_ = L.tocsr()[: self.n_users, self.n_users :].tocoo()
        # covert norm_inter_matrix to tensor
        i_ = torch.LongTensor(np.array([L_.row, L_.col]))
        data_ = torch.FloatTensor(L_.data)
        norm_matrix = torch.sparse.FloatTensor(i_, data_, L_.shape)

        return norm_matrix.to(self.device), norm_graph.to(self.device)

    def get_edges(self, graph):
        index = torch.LongTensor(np.array([graph.row, graph.col]))
        type = torch.LongTensor(np.array(graph.data))
        return index.to(self.device), type.to(self.device)

    def forward(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        latent_embeddings = self.latent_embedding.weight
        # entity_gcn_emb: [n_entities, embedding_size]
        # user_gcn_emb: [n_users, embedding_size]
        # latent_gcn_emb: [n_factors, embedding_size]
        entity_gcn_emb, user_gcn_emb, cor_loss = self.gcn(
            user_embeddings, entity_embeddings, latent_embeddings
        )

        return user_gcn_emb, entity_gcn_emb, cor_loss

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.
        Args:
            interaction (Interaction): Interaction class of the batch.
        Returns:
            torch.Tensor: Training loss, shape: []
        """

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings, cor_loss = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        cor_loss = self.sim_decay * cor_loss
        loss = mf_loss + self.reg_weight * reg_loss + cor_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e, _ = self.forward()
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[: self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)


class CS_KGIN_Imp(KGIN):
    def __init__(self, config, dataset):
        """ 相较于KGIN, 修改了 interact_mat 
        data
            inter_matrix: 原始的交互矩阵 [n_users, n_items]; coo
                interact_mat: 归一化之后的结果 [n_users, n_entities]; torch.sparse_coo
            kg_graph: [n_entities, n_entities]; coo格式
            edge_index, edge_type: 知识图谱的边
            user_embedding, entity_embedding, latent_embedding 分别是用户、KG、兴趣维度
            
            n_factors: 兴趣建模维度
            n_users, n_items, n_entities, n_relations: 用户, item, 实体, 关系数量
        """
        super().__init__(config, dataset)
        # KnowledgeRecommender.__init__(self, config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.n_factors = config["n_factors"]
        self.context_hops = config["context_hops"]
        self.node_dropout_rate = config["node_dropout_rate"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        self.ind = config["ind"]
        self.sim_decay = config["sim_regularity"]
        self.reg_weight = config["reg_weight"]
        self.temperature = config["temperature"]

        # load dataset info
        # edge_index: [2, -1]; edge_type: [-1,]
        self.edge_index, self.edge_type = self.get_edges(self.kg_graph)
        # raw inter
        self.inter_matrix = dataset.inter_matrix(
            form="coo"
        ).astype(np.float32)  # [n_users, n_items]
        self.kg_graph = dataset.kg_graph(
            form="coo", value_field="relation_id"
        )  # [n_entities, n_entities]
        # KG上的节点重要性, [n_entities]
        # NOTE: 这里计算的 interact_mat 引入了imp信息
        self.entity_imp = self.get_node_imp_ui(self.inter_matrix, self.kg_graph)
        self.CS_inter_matrix = self.get_CS_inter_matrix(self.inter_matrix, self.entity_imp)
        self.interact_mat = self.get_CS_norm_adj_mat(self.CS_inter_matrix)

        # define layers and loss
        self.n_nodes = self.n_users + self.n_entities
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.latent_embedding = nn.Embedding(self.n_factors, self.embedding_size)
        self.gcn = GraphConv(
            embedding_size=self.embedding_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_relations=self.n_relations,
            n_factors=self.n_factors,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            interact_mat=self.interact_mat,
            ind=self.ind,
            tmp=self.temperature,
            device=self.device,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)

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
            norm_graph: [n_users+n_entities, n_users+n_entities]
            norm_matrix: [n_users, n_entities]
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_entities, self.n_users + self.n_entities),
            dtype=np.float32,
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

        # covert norm_inter_graph to tensor
        # i = torch.LongTensor(np.array([L.row, L.col]))
        # data = torch.FloatTensor(L.data)
        # norm_graph = torch.sparse.FloatTensor(i, data, L.shape)
        # interaction: user->item, [n_users, n_entities]
        L_ = L.tocsr()[: self.n_users, self.n_users :].tocoo()
        # covert norm_inter_matrix to tensor
        i_ = torch.LongTensor(np.array([L_.row, L_.col]))
        data_ = torch.FloatTensor(L_.data)
        norm_matrix = torch.sparse.FloatTensor(i_, data_, L_.shape)

        return norm_matrix.to(self.device)

    def forward(self):
        """ 同KGIN, gcn部分所用的 interact_mat 引入了imp """
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        latent_embeddings = self.latent_embedding.weight
        # entity_gcn_emb: [n_entities, embedding_size]
        # user_gcn_emb: [n_users, embedding_size]
        # latent_gcn_emb: [n_factors, embedding_size]
        entity_gcn_emb, user_gcn_emb, cor_loss = self.gcn(
            user_embeddings, entity_embeddings, latent_embeddings
        )
        return user_gcn_emb, entity_gcn_emb, cor_loss


class ClsAggregator(Aggregator):
    """
    Relational Path-aware Convolution Network
    输入: 图信息, 输出聚合结果
    """
    def __init__(self):
        super().__init__()

    def forward(self, entity_emb, item_emb, user_emb, cls_emb, relation_emb, latent_emb, 
                edge_index, edge_type, 
                interact_mat, inter_cls_mat, 
                disen_weight_att):
        """ 相较于 Aggregator, interact_mat 维度只用到item, 并且增加了类别建模; 在前向过程中, 同时用了intent和cls建模!
        data
            interact_mat: [n_users, n_items]
            inter_cls_mat: [n_cls, n_users, n_items]

        """
        from torch_scatter import scatter_mean

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type]
        neigh_relation_emb = (
            entity_emb[tail] * edge_relation_emb
        )  # [-1, embedding_size]
        entity_agg = scatter_mean(
            src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0
        )

        # """cul user->latent factor attention"""
        # score_ = torch.mm(user_emb, latent_emb.t())
        # score = nn.Softmax(dim=1)(score_)  # [n_users, n_factors] 用户兴趣建模
        # """user aggregate"""
        # user_agg = torch.sparse.mm(
        #     interact_mat, entity_emb
        # )  # [n_users, embedding_size]
        # disen_weight = torch.mm(
        #     nn.Softmax(dim=-1)(disen_weight_att), relation_emb
        # )  # [n_factors, embedding_size]
        # user_agg = (
        #     torch.mm(score, disen_weight)
        # ) * user_agg + user_agg  # [n_users, embedding_size]
        
        # NOTE: 同时用了intent和cls建模!
        # interact_mat 的维度从 n_entities 变成了 n_items
        user_agg = torch.sparse.mm(interact_mat, item_emb) # [n_users, embedding_size]

        user_disen_att = torch.mm(user_emb, latent_emb.t()).softmax(dim=1) # [n_users, n_factors]
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att), relation_emb) # [n_factors, embedding_size]
        user_agg = (torch.mm(user_disen_att, disen_weight)) + user_agg # [n_users, embedding_size]
        #  * user_agg + user_agg  # [n_users, embedding_size]

        user_cls_att = torch.mm(user_emb, cls_emb.t()).softmax(dim=1) # [n_users, n_cls]
        user_cls_ = torch.stack([
            torch.matmul(inter_cls_mat[i], item_emb) for i in range(inter_cls_mat.shape[0])
        ]).transpose(0, 1) # [n_users, n_cls, embedding_size]
        # norm_user_emb = F.normalize(user_emb, p=2, dim=1)
        user_agg = (user_cls_ * user_cls_att.unsqueeze(-1)).sum(dim=1) + user_agg
        
        return entity_agg, user_agg


class ClsGraphConv(GraphConv):
    """ 相较于 GraphConv, 多了 inter_cls_mat, 并调用了 ClsAggregator """
    def __init__(self, embedding_size, n_hops, n_users, n_items, n_factors, n_relations, 
                 edge_index, edge_type, 
                 interact_mat, inter_cls_mat, 
                 ind, tmp, device, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super().__init__(embedding_size, n_hops, n_users, n_factors, n_relations, edge_index, edge_type, interact_mat, ind, tmp, device, node_dropout_rate, mess_dropout_rate)
        self.n_items = n_items
        self.inter_cls_mat = inter_cls_mat

        # define layers
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        disen_weight_att = nn.init.xavier_uniform_(torch.empty(n_factors, n_relations))
        self.disen_weight_att = nn.Parameter(disen_weight_att)
        self.convs = nn.ModuleList()
        for i in range(self.n_hops):
            self.convs.append(ClsAggregator())

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def forward(self, user_emb, entity_emb, cls_emb, latent_emb):
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type = self.edge_sampling(
                self.edge_index, self.edge_type, self.node_dropout_rate
            )
            interact_mat = self.node_dropout(self.interact_mat)
            inter_cls_mat = self.node_dropout(self.inter_cls_mat)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            interact_mat = self.interact_mat
            inter_cls_mat = self.inter_cls_mat

        entity_res_emb = entity_emb             # [n_entities, embedding_size]
        user_res_emb = user_emb                 # [n_users, embedding_size]
        item_emb = entity_emb[:self.n_items]    # [n_items, embedding_size]
        relation_emb = self.relation_embedding.weight  # [n_relations, embedding_size]
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](
                entity_emb, item_emb, user_emb, cls_emb, relation_emb, latent_emb, 
                edge_index, edge_type,
                interact_mat, inter_cls_mat, 
                self.disen_weight_att,
            )
            """message dropout"""
            if self.mess_dropout_rate > 0.0:
                entity_emb = self.mess_dropout(entity_emb)
                user_emb = self.mess_dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return (
            entity_res_emb,
            user_res_emb,
            self.calculate_cor_loss(self.disen_weight_att),
        )

class CS_KGIN_Cls(KGIN):
    def __init__(self, config, dataset:KnowledgeBasedDataset):
        """ 相较于KGIN, 完全进行了魔改, 见 ClsAggregator
        data
            inter_matrix: 原始的交互矩阵 [n_users, n_items]
                inter_cls_mat: [n_clusters, n_user, n_items]
            user_embedding, entity_embedding, latent_embedding, cls_embedding 分别是用户、KG、兴趣、类别
        """
        super().__init__(config, dataset)
        self.n_clusters = config["n_clusters"]
        self.item_cls = dataset.item_cluster

        self.inter_matrix = dataset.inter_matrix(form="coo").astype(
            np.float32
        )  # [n_users, n_items]
        self.inter_cls_mat = self.get_inter_cls_mat(self.inter_matrix, self.item_cls)
        self.inter_matrix = _convert_sp_mat_to_sp_tensor(self.inter_matrix).to(self.device)

        self.n_nodes = self.n_users + self.n_entities
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.cls_embedding = nn.Embedding(self.n_clusters, self.embedding_size)
        self.latent_embedding = nn.Embedding(self.n_factors, self.embedding_size)
        self.gcn = ClsGraphConv(
            embedding_size=self.embedding_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            n_relations=self.n_relations,
            edge_index=self.edge_index,
            edge_type=self.edge_type,

            interact_mat=self.inter_matrix,     # interact_mat
            inter_cls_mat=self.inter_cls_mat,
            
            ind=self.ind,
            tmp=self.temperature,
            device=self.device,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
        )

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_inter_cls_mat(self, inter_matrix, item_cls):
        """  
        return
            inter_cls_mat: [n_clusters, n_users, n_items]
        """
        
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

    def forward(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        cls_embeddings = self.cls_embedding.weight
        latent_embeddings = self.latent_embedding.weight
        # entity_gcn_emb: [n_entities, embedding_size]
        # user_gcn_emb: [n_users, embedding_size]
        # latent_gcn_emb: [n_factors, embedding_size]
        entity_gcn_emb, user_gcn_emb, cor_loss = self.gcn(
            user_embeddings, entity_embeddings, cls_embeddings, latent_embeddings
        )
        return user_gcn_emb, entity_gcn_emb, cor_loss

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)

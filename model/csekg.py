# -*- coding: utf-8 -*-
# @Time   : 2021/3/25
# @Author : Wenqi Sun
# @Email  : wenqisun@pku.edu.cn

# UPDATE:
# @Time   : 2022/8/31
# @Author : Bowen Zheng
# @Email  : 18735382001@163.com

r"""
SAKG
相较于KGIN的改进:
    Aggregator 层中增加 plugin的建模
    GraphConv 抽象中添加了参数
    Ours_v3 主体模型部分的数据处理, 参数传递
    self.get_inter_cls_mat(self.inter_matrix): 生成类别交互矩阵
    get_edge_imp(self, inter_matrix, graph, mode='ui'): 生成边重要性
封装 GraphConv: Graph Convolutional Network
    核心: forward(self, user_emb, entity_emb, latent_emb) 在三个部分的基础上进行建模


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

from collections import *
from dgl.nn.functional import edge_softmax
import dgl

class Aggregator(nn.Module):
    def __init__(
        self,
    ):
        super(Aggregator, self).__init__()

    def forward(
        self,
        entity_emb,
        item_emb,
        user_emb,
        latent_emb,         # intent 表示 (n_factors, embedding_size)
        relation_emb,
        edge_index,
        edge_type,
        edge_imp,
        interact_mat,
        disen_weight_att,   # 兴趣-关系 (n_factors, n_relations)
        ent_rel_w,          # 实体-关系 (n_relations, hdim)
        usr_cls_w,          # 用户-类别 (n_clusters, hdim)
        inter_cls_mat,            # 用户分类别的交互记录 (n_clusters, n_users, n_items)
        # cls_interact_mat,
        e_mode='att_imp',
        u_mode='inter_cls'
        # u_mode='intent'
    ):
        from torch_scatter import scatter_mean, scatter_sum

        n_entities = entity_emb.shape[0]
        
        """KG aggregate 
        基本物品交互, 加入了边 imp 
            prop: 基本的传播, 在KGIN的基础上加了 imp
        """
        head, tail = edge_index
        if e_mode=='prop':
            # Agg(tail * r * imp) -> head
            edge_relation_emb = relation_emb[edge_type]
            # 1] 加了 edge_imp
            neigh_relation_emb = entity_emb[tail] * edge_relation_emb * edge_imp.view(-1, 1)    # [-1, embedding_size]
            entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        # elif e_mode=='prop2':
        #     edge_relation_emb = relation_emb[edge_type] # [n_edges, embedding_size]
        #     entity_relation_att = torch.mm(entity_emb, ent_rel_w.T).softmax(dim=1) # [n_entities, n_relations]
        #     mess_att = entity_relation_att[head].gather(1, edge_type.view(-1, 1)) # [n_edges, 1]
        #     message_score = edge_relation_emb * mess_att * edge_imp.view(-1, 1) # [n_edges, embedding_size]
        #     message = entity_emb[tail] * message_score # [n_entities, embedding_size]
        #     entity_agg = scatter_sum(src=message, index=head, dim_size=n_entities, dim=0) # [n_entities, embedding_size]
        elif e_mode=='att_imp':
            # Agg(tail * att * imp) -> head. 这里的att根据head和relation计算
            # 物品对于关系的注意力
            entity_relation_att = torch.mm(entity_emb, relation_emb.T).softmax(dim=1) # [n_entities, n_relations]
            mess_att = entity_relation_att[head].gather(1, edge_type.view(-1, 1)) # [n_edges, 1]
            message_score = relation_emb[edge_type] * mess_att * edge_imp.view(-1, 1) # [n_edges, embedding_size]
            message = entity_emb[tail] * message_score # [n_entities, embedding_size]
            entity_agg = scatter_sum(src=message, index=head, dim_size=n_entities, dim=0) # [n_entities, embedding_size]
        elif e_mode=='att_noimp':
            # Agg(tail * att * imp) -> head. 这里的att根据head和relation计算
            # 物品对于关系的注意力
            entity_relation_att = torch.mm(entity_emb, relation_emb.T).softmax(dim=1) # [n_entities, n_relations]
            mess_att = entity_relation_att[head].gather(1, edge_type.view(-1, 1)) # [n_edges, 1]
            message_score = relation_emb[edge_type] * mess_att  # [n_edges, embedding_size]
            message = entity_emb[tail] * message_score # [n_entities, embedding_size]
            entity_agg = scatter_sum(src=message, index=head, dim_size=n_entities, dim=0) # [n_entities, embedding_size]
        else: raise NotImplementedError

        """ user aggregate
        用户聚合方式
            intent: KGIN
            inter_cls: 利用分类交互矩阵聚合物品, 从而得到用户表示
        """
        if u_mode=='intent':
            """cul user->latent factor attention"""
            score_ = torch.mm(user_emb, latent_emb.t())
            score = nn.Softmax(dim=1)(score_)  # [n_users, n_factors] 用户兴趣分布
            """user aggregate
            一方面, 来自 interact_mat @ entity_emb —— 物品聚合
            另一方面, 来自 score @ disen_weight_att @ relation_emb  —— 用户-兴趣-关系-表示
            """
            user_agg = torch.sparse.mm(
                interact_mat, entity_emb
            )  # [n_users, embedding_size]
            disen_weight = torch.mm(
                nn.Softmax(dim=-1)(disen_weight_att), relation_emb
            )  # [n_factors, embedding_size]
            user_agg = (        # 这里得到的第二部分, 通过 x*user_agg + user_agg 的方式汇总
                torch.mm(score, disen_weight)
            ) * user_agg + user_agg  # [n_users, embedding_size]
        elif u_mode=='cls':
            """ 类似上面KGIN引入的intent中间层, 也可以直接对于用户对于聚类兴趣引入中间层 """
            pass
        elif u_mode=='inter_cls':
            """ 利用分类交互矩阵聚合物品, 从而得到用户表示 """
            user_agg = torch.sparse.mm(interact_mat, entity_emb) # [n_users, embedding_size]
            
            user_cls_att = torch.mm(user_emb, usr_cls_w.t()).softmax(dim=1) # [n_users, n_cls]
            # ent_rel_w -> relation_emb
            item_emb_2 = item_emb * relation_emb.sum(dim=0, keepdim=True) # [n_items, embedding_size]
            disen_weight = torch.stack([
                torch.matmul(inter_cls_mat[i], item_emb_2) for i in range(inter_cls_mat.shape[0])
            ]).transpose(0, 1) # [n_users, n_cls, embedding_size]
            user_agg += (disen_weight * user_cls_att.unsqueeze(-1)).sum(dim=1) # [n_users, embedding_size]
        elif u_mode=='inter_nocls':
            user_agg = torch.sparse.mm(interact_mat, entity_emb) # [n_users, embedding_size]
            
        else: raise NotImplementedError

        return entity_agg, user_agg


class GraphConv(nn.Module):
    def __init__(
        self,
        embedding_size,
        n_hops,
        n_users,
        n_items,
        n_factors,      # 用户兴趣数量
        n_relations,
        n_clusters,     # 物品类别数量
        edge_index,
        edge_type,
        edge_imp,
        interact_mat,
        inter_cls_mat,
        ind,
        tmp,
        device,
        node_dropout_rate=0.5,
        mess_dropout_rate=0.1,
        config=None
    ):
        super(GraphConv, self).__init__()

        self.embedding_size = embedding_size
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_imp = edge_imp
        self.interact_mat = interact_mat
        self.inter_cls_mat = inter_cls_mat
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.temperature = tmp
        self.device = device
        self.config=config

        # define layers
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        disen_weight_att = nn.init.xavier_uniform_(torch.empty(n_factors, n_relations))
        self.disen_weight_att = nn.Parameter(disen_weight_att)
        self.ent_rel_w = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_relations, embedding_size)))
        self.usr_cls_w = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_clusters, embedding_size)))
        self.convs = nn.ModuleList()
        for i in range(int(self.n_hops)):
            self.convs.append(Aggregator())
        self.node_dropout = SparseDropout(p=self.mess_dropout_rate)  # node dropout
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)  # mess dropout

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def edge_sampling(self, edge_index, edge_type, edge_imp, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * rate), replace=False
        )
        return edge_index[:, random_indices], edge_type[random_indices], edge_imp[random_indices]

    def forward(self, user_emb, entity_emb, latent_emb):
        """node dropout"""
        # node dropout
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type, edge_imp = self.edge_sampling(
                self.edge_index, self.edge_type, self.edge_imp, self.node_dropout_rate
            )
            interact_mat = self.node_dropout(self.interact_mat)
            inter_cls_mat = self.node_dropout(self.inter_cls_mat)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            interact_mat = self.interact_mat
            inter_cls_mat = self.inter_cls_mat

        entity_res_emb = entity_emb  # [n_entities, embedding_size]
        user_res_emb = user_emb  # [n_users, embedding_size]
        relation_emb = self.relation_embedding.weight  # [n_relations, embedding_size]
        for i in range(len(self.convs)):
            item_emb = entity_emb[:self.n_items, :]
            entity_emb, user_emb = self.convs[i](
                entity_emb,
                item_emb,
                user_emb,
                latent_emb,
                relation_emb,
                edge_index,
                edge_type,
                edge_imp,
                interact_mat,
                self.disen_weight_att,
                self.ent_rel_w,
                self.usr_cls_w,
                inter_cls_mat,
                e_mode=self.config['e_mode'],
                u_mode=self.config['u_mode'],
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


class CSEKG(KnowledgeRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, ifCase=False):
        super(CSEKG, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.n_factors = config["n_factors"]
        self.n_clusters = config["n_clusters"]
        self.context_hops = config["context_hops"]
        self.node_dropout_rate = config["node_dropout_rate"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        self.ind = config["ind"]
        self.sim_decay = config["sim_regularity"]
        self.reg_weight = config["reg_weight"]
        self.temperature = config["temperature"]

        # load dataset info
        self.dataset = dataset
        self.inter_matrix = dataset.inter_matrix(form="coo").astype(np.float32)  # [n_users, n_items]
        # NOTE: case
        if ifCase:
            # 查看用户交互记录的商品cls
            mat = self.inter_matrix.tocsr()
            UID=346
            UID=5
            iids = np.where(mat[UID,:].toarray().squeeze().astype(bool))[0]
            dataset.item_cluster[iids]
            # 查看物品所交互的用户list
            # uids = np.where(mat[:,89].toarray().squeeze())[0]

        # inter_matrix: [n_users, n_entities]; inter_graph: [n_users + n_entities, n_users + n_entities]
        self.interact_mat, _ = self.get_norm_inter_matrix(mode="si")
        # 节点的类别
        self.item_cls = dataset.item_cluster
        # self.item_cls = torch.randint(0, self.n_clusters, (self.n_items,))
        self.inter_cls_mat = self.get_inter_cls_mat(self.inter_matrix)
        self.kg_graph = dataset.kg_graph(
            form="coo", value_field="relation_id"
        )  # [n_entities, n_entities]
        # edge_index: [2, -1]; edge_type: [-1,]
        # 将 edge_soft_imp 作为参数
        
        # kg = self.kg_graph.tocsr().toarray(); # Counter(kg[89,:]); 
        # dataset.field2id_token['relation_id']
        self.edge_index, self.edge_type = self.get_edges(self.kg_graph)
        edge_imp = self.get_edge_imp(inter_matrix=self.inter_matrix, graph=self.kg_graph, mode=config['imp_mode'])
        # self.edge_imp = nn.Parameter(edge_imp)
        if config['imp_fixed']:
            self.edge_imp = torch.tensor(edge_imp, dtype=torch.float32)
        else:
            self.edge_imp = nn.Parameter(torch.tensor(edge_imp, dtype=torch.float32))
        
        # NOTE: case 
        if ifCase:
            # self.dataset.field2id_token['item_id'][89]
            # IID=300     # 银翼杀手
            IID=89      # 
            idxs = np.where(self.kg_graph.row==IID)[0]
            entity_ids = self.kg_graph.col[idxs]
            rels = self.kg_graph.data[idxs]
            edge_imp[idxs]
            # 筛选特定类别的节点重要性
            node_imp = self.get_node_imp_ui(self.inter_matrix, self.kg_graph)
            _idxs= rels==2
            node_imp[entity_ids][_idxs]
        
        # define layers and loss
        self.n_nodes = self.n_users + self.n_entities
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.latent_embedding = nn.Embedding(self.n_factors, self.embedding_size)
        self.gcn = GraphConv(
            embedding_size=self.embedding_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_items=self.n_items,
            n_relations=self.n_relations,
            n_factors=self.n_factors,
            n_clusters=self.n_clusters,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_imp=self.edge_imp,
            interact_mat=self.interact_mat,
            inter_cls_mat=self.inter_cls_mat,
            ind=self.ind,
            tmp=self.temperature,
            device=self.device,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
            config=config,
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

    def get_inter_cls_mat(self, inter_matrix):
        """  """
        cls_adj_dict = [[] for i in range(self.n_clusters)]
        for uid, iid in zip(inter_matrix.row, inter_matrix.col):
            cls_adj_dict[self.item_cls[iid]].append((uid, iid))
        # adj_dict = {0: sp.coo_matrix(([1], ([1], [1])), shape=(self.n_users, self.n_items))}
        for i in range(self.n_clusters):
            cls_adj_dict[i] = sp.coo_matrix(
                (np.ones(len(cls_adj_dict[i])), zip(*cls_adj_dict[i])),
                shape=(self.n_users, self.n_items),
            )
        ss = [self._convert_sp_mat_to_sp_tensor(v) for v in cls_adj_dict]
        adj_mat = torch.stack(ss, dim=0)
        return adj_mat.to(self.device)
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
    def get_edges(self, graph):
        """ 获取图中的边信息
        imp: 计算每条边的重要性. 计算方式为, 对于 property 统计所连边数量, 取log; 对于item选择所连props最大值. 
        """
        index = torch.LongTensor(np.array([graph.row, graph.col]))
        type = torch.LongTensor(np.array(graph.data))
        return index.to(self.device), type.to(self.device)

    def get_edge_imp(self, inter_matrix, graph, mode='ui'):
        if mode=='ui':
            node_imp = self.get_node_imp_ui(inter_matrix, graph)
        elif mode=='kg':
            node_imp = self.get_node_imp_kg(graph)
        elif mode=='same':
            node_imp = torch.ones(self.n_entities)
        elif mode=='random':
            node_imp = torch.rand(self.n_entities)
        else:
            raise NotImplementedError(f"The imp_mode [{mode}] has not been supported.")
        edge_soft_imp = self.imp_modelling(graph, node_imp)
        return edge_soft_imp.to(self.device)
    
    def get_node_imp_kg(self, graph):
        imp = torch.ones(self.n_entities)
        for u,v in zip(graph.row, graph.col):
            imp[v] += 1; imp[u] += 1
        return torch.FloatTensor(imp)
    
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

    def imp_modelling(self, graph, node_imp):
        """ 利用节点imp计算边的imp, 计算方式为对每个类型的边进行softmax """
        src,dst,etype = graph.row, graph.col, graph.data
        edge_imp = node_imp[dst]
        edge_soft_imp = torch.ones_like(edge_imp)
        kg_dgl = dgl.graph((src,dst))
        for rid in range(self.n_relations): # [PAD]=0, [UI-Relation]=25, 见 dataset.field2id_token[dataset.RELATION_ID]
            idxs = torch.tensor([i for i,x in enumerate(etype) if x==rid])
            if len(idxs)==0: continue       # [PAD]=0, [UI-Relation]=25
            edge_soft_imp_ = edge_softmax(kg_dgl, edge_imp[idxs], eids=idxs, norm_by='src')
            edge_soft_imp[idxs] = edge_soft_imp_
        return edge_soft_imp

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

"""
LightGCN 推荐模型
当前仅启用图嵌入层(LightGCN) → 推荐预测层(InnerProduct)。
DDPM 组件保留在 diffusion_layers.py，后续可重新接入。
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import coo_matrix

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class LightGCNDiffusion(GeneralRecommender):
    """纯 LightGCN 推荐模型，类名暂保留以兼容训练入口"""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.n_nodes = self.n_users + self.n_items

        # LightGCN 参数
        self.n_layers = config['n_layers']
        self.embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']

        # 嵌入矩阵 [n_users + n_items, embedding_size]
        self.node_embedding = nn.Embedding(self.n_nodes, self.embedding_size)
        nn.init.normal_(self.node_embedding.weight, std=0.1)

        # 构建归一化邻接矩阵 (仅用训练集)
        self.norm_adj = self._build_norm_adj(dataset)
        self.norm_adj = self.norm_adj.to(self.device)

        # BPR 损失
        self.bpr_loss = BPRLoss()

        self.apply(xavier_normal_initialization)

    def _build_norm_adj(self, dataset):
        """构建对称归一化邻接矩阵 A_hat = D^{-1/2} A D^{-1/2}"""
        n_users = dataset.user_num
        n_items = dataset.item_num
        n_nodes = n_users + n_items

        # 使用交互矩阵获取 (user, item) 对
        inter_matrix = dataset.inter_matrix(form='coo')
        users = inter_matrix.row
        items = inter_matrix.col

        # 构建二部图邻接矩阵
        rows = np.concatenate([users, items + n_users])
        cols = np.concatenate([items + n_users, users])
        data = np.ones(len(rows), dtype=np.float32)

        adj = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        adj = adj.tocsr()

        # 归一化 D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
        nonzero_mask = rowsum > 0
        d_inv_sqrt[nonzero_mask] = np.power(rowsum[nonzero_mask], -0.5)
        d_mat_inv_sqrt = coo_matrix(
            (d_inv_sqrt, (np.arange(n_nodes), np.arange(n_nodes))),
            shape=(n_nodes, n_nodes)
        )
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

        # 转换为稀疏张量
        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def get_ego_embeddings(self):
        """获取用户和物品的初始嵌入"""
        return self.node_embedding.weight

    # ==================== 图嵌入层 (LightGCN) ====================

    def lightgcn_forward(self):
        """LightGCN 前向传播，返回各层的嵌入和最终嵌入"""
        ego_emb = self.get_ego_embeddings()
        all_emb = [ego_emb]
        emb = ego_emb

        for k in range(self.n_layers):
            emb = torch.sparse.mm(self.norm_adj, emb)
            all_emb.append(emb)

        # 最终嵌入 = 各层均值
        final_emb = torch.stack(all_emb, dim=1).mean(dim=1)
        return all_emb, final_emb

    # ==================== 推荐预测层 ====================

    def forward(self):
        """完整前向传播: 图嵌入 → 获取最终用户/物品嵌入"""
        _, final_emb = self.lightgcn_forward()

        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]

        return user_emb, item_emb

    def get_lightgcn_embeddings(self):
        """仅获取 LightGCN 嵌入（不经过扩散增强），用于 BPR 训练"""
        _, final_emb = self.lightgcn_forward()
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        """总损失 = BPR损失 + L2正则"""
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_emb, item_emb = self.forward()

        u_emb = user_emb[user]
        pos_emb = item_emb[pos_item]
        neg_emb = item_emb[neg_item]

        # BPR 损失: 先求内积分数，再传入 BPRLoss
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        # L2 正则
        reg_loss = (1 / 2) * (
            u_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / float(len(user))

        total_loss = bpr_loss + self.reg_weight * reg_loss

        return total_loss

    def predict(self, interaction):
        """预测用户-物品评分（内积）"""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_emb, item_emb = self.forward()

        u_emb = user_emb[user]
        i_emb = item_emb[item]

        scores = torch.sum(u_emb * i_emb, dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """为给定用户预测所有物品的评分（用于评估）"""
        user = interaction[self.USER_ID]

        user_emb, item_emb = self.forward()

        u_emb = user_emb[user]
        scores = torch.matmul(u_emb, item_emb.transpose(0, 1))

        return scores.view(-1)

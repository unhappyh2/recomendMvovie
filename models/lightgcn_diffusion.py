"""
LightGCN + Diffusion 推荐模型
三层架构：图嵌入层(LightGCN) → 特征增强层(DDPM) → 推荐预测层(InnerProduct)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from models.diffusion_layers import DiffusionSchedule, DenoisingMLP


class LightGCNDiffusion(GeneralRecommender):
    """LightGCN + Diffusion 融合推荐模型"""
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

        # DDPM 参数
        self.diffusion_steps = config['diffusion_steps']
        self.diffusion_schedule = DiffusionSchedule(
            timesteps=self.diffusion_steps,
            beta_start=config['diffusion_beta_start'],
            beta_end=config['diffusion_beta_end'],
            schedule=config['diffusion_schedule'] if 'diffusion_schedule' in config else 'linear'
        )
        self.diffusion_schedule.to(self.device)

        # 去噪网络
        self.denoiser = DenoisingMLP(
            embed_dim=self.embedding_size,
            hidden_dims=config['diffusion_hidden_dims']
        )

        # BPR 损失
        self.bpr_loss = BPRLoss()

        # 可选: 记录日志
        self._first_diffusion_step = True

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
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
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

    # ==================== 特征增强层 (DDPM) ====================

    def diffusion_training_loss(self, embeddings, batch_size=None):
        """DDPM 训练损失: 在 LightGCN 嵌入上训练去噪网络"""
        n_nodes = embeddings.shape[0]
        if batch_size is None or batch_size >= n_nodes:
            sampled_emb = embeddings
        else:
            indices = torch.randperm(n_nodes, device=embeddings.device)[:batch_size]
            sampled_emb = embeddings[indices]

        # 随机采样时间步
        t = torch.randint(1, self.diffusion_steps + 1, (sampled_emb.shape[0],),
                          device=sampled_emb.device).long()

        # 生成噪声
        noise = torch.randn_like(sampled_emb)

        # 前向加噪
        x_t, _ = self.diffusion_schedule.q_sample(sampled_emb, t - 1, noise)

        # 去噪网络预测噪声
        noise_pred = self.denoiser(x_t, t)

        # MSE 损失
        loss = F.mse_loss(noise_pred, noise, reduction='mean')
        return loss

    @torch.no_grad()
    def diffusion_reverse(self, embeddings):
        """DDPM 反向去噪过程，从纯噪声逐步恢复，得到增强嵌入"""
        self.denoiser.eval()
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # 从纯噪声开始
        x = torch.randn_like(embeddings)

        for step in reversed(range(1, self.diffusion_steps + 1)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            t_idx = step - 1

            # 预测噪声
            noise_pred = self.denoiser(x, t)

            # 更新: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_cumprod_t) * noise_pred)
            alpha = self.diffusion_schedule.alphas[t_idx]
            alpha_cumprod = self.diffusion_schedule.alphas_cumprod[t_idx]
            beta = self.diffusion_schedule.betas[t_idx]

            if step > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1.0 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1.0 - alpha_cumprod)) * noise_pred
            ) + torch.sqrt(beta) * noise

        self.denoiser.train()
        return x

    @torch.no_grad()
    def diffusion_enhance(self, embeddings):
        """对 LightGCN 嵌入进行 DDPM 特征增强"""
        x_0 = embeddings
        device = embeddings.device
        batch_size = x_0.shape[0]

        # 随机选一个中间时间步加噪
        t_sample = torch.randint(1, self.diffusion_steps // 2 + 1, (1,), device=device).item()
        t = torch.full((batch_size,), t_sample, device=device, dtype=torch.long)
        t_idx = t_sample - 1

        noise = torch.randn_like(x_0)
        x_t, _ = self.diffusion_schedule.q_sample(x_0, t_idx, noise)

        # 反向去噪恢复
        self.denoiser.eval()
        x = x_t
        for step in reversed(range(1, t_sample + 1)):
            t_s = torch.full((batch_size,), step, device=device, dtype=torch.long)
            t_s_idx = step - 1

            noise_pred = self.denoiser(x, t_s)

            alpha = self.diffusion_schedule.alphas[t_s_idx]
            alpha_cumprod = self.diffusion_schedule.alphas_cumprod[t_s_idx]
            beta = self.diffusion_schedule.betas[t_s_idx]

            if step > 1:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            x = (1.0 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1.0 - alpha_cumprod)) * noise_pred
            ) + torch.sqrt(beta) * z

        self.denoiser.train()
        return x

    # ==================== 推荐预测层 ====================

    def forward(self):
        """完整前向传播: 图嵌入 → 特征增强 → 获取最终嵌入"""
        # 图嵌入层
        _, final_emb = self.lightgcn_forward()

        # 特征增强层
        enhanced_emb = self.diffusion_enhance(final_emb)

        # 分离用户和物品增强嵌入
        user_emb = enhanced_emb[:self.n_users]
        item_emb = enhanced_emb[self.n_users:]

        return user_emb, item_emb

    def get_lightgcn_embeddings(self):
        """仅获取 LightGCN 嵌入（不经过扩散增强），用于 BPR 训练"""
        _, final_emb = self.lightgcn_forward()
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        """总损失 = BPR损失 + λ * DDPM损失 + L2正则"""
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # 图嵌入 + 特征增强
        user_emb, item_emb = self.forward()

        u_emb = user_emb[user]
        pos_emb = item_emb[pos_item]
        neg_emb = item_emb[neg_item]

        # BPR 损失: 先求内积分数，再传入 BPRLoss
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        # DDPM 损失
        _, lgn_emb = self.lightgcn_forward()
        diffusion_loss = self.diffusion_training_loss(lgn_emb)

        # L2 正则
        reg_loss = (1 / 2) * (
            u_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / float(len(user))

        total_loss = bpr_loss + 0.1 * diffusion_loss + self.reg_weight * reg_loss

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

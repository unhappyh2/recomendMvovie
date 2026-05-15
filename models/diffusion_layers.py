"""
扩散模型各组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, max=0.999)
    return betas


class DiffusionSchedule:
    """扩散过程调度器，管理 alpha/beta 等噪声参数"""

    def __init__(self, timesteps, beta_start=1e-4, beta_end=0.02, schedule='linear'):
        self.timesteps = timesteps

        if schedule == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown schedule: {schedule}')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # 前向扩散 q(x_t | x_0) 的参数
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # 反向扩散 q(x_{t-1} | x_t, x_0) 的参数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def to(self, device):
        for attr_name in list(self.__dict__.keys()):
            val = getattr(self, attr_name)
            if isinstance(val, torch.Tensor):
                setattr(self, attr_name, val.to(device))
        return self

    def q_sample(self, x_start, t, noise=None):
        """前向加噪: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1-alpha_cumprod_t) * noise"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise


class TimeEmbedding(nn.Module):
    """时间步嵌入 (Sinusoidal)"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DenoisingMLP(nn.Module):
    """去噪网络 MLP，预测噪声 epsilon"""

    def __init__(self, embed_dim, hidden_dims, time_emb_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        if time_emb_dim is None:
            time_emb_dim = embed_dim

        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        layers = []
        input_dim = embed_dim + time_emb_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.SiLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        x: 噪声嵌入 [batch, embed_dim]
        t: 时间步 [batch]
        返回: 预测的噪声 [batch, embed_dim]
        """
        t_emb = self.time_mlp(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)

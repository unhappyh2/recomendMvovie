"""
DiffuRec-style recommendation model.

The class name is kept as LightGCNDiffusion so train.py and the web artifact
pipeline can stay unchanged. Internally this is a sequential diffusion model
adapted from WHUIR/DiffuRec for RecBole's GeneralRecommender interface.
"""
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.utils import InputType


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _betas_for_alpha_bar(num_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def _beta_schedule(name, steps):
    if name == 'linear':
        scale = 1000 / steps
        return np.linspace(scale * 0.0001, scale * 0.02, steps, dtype=np.float64)
    if name == 'cosine':
        return _betas_for_alpha_bar(
            steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
    if name == 'sqrt':
        return _betas_for_alpha_bar(steps, lambda t: 1 - np.sqrt(t + 0.0001))
    if name == 'trunc_lin':
        scale = 1000 / steps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, steps, dtype=np.float64)
    raise ValueError(f'Unknown diffusion schedule: {name}')


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.variance_epsilon) + self.bias


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        hidden = 0.5 * hidden * (
            1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden.pow(3)))
        )
        return self.w_2(self.dropout(hidden))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError('hidden_size must be divisible by attention heads')
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [
            layer(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (q, k, v))
        ]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            attn_mask = mask[:, None, None, :].bool()
            scores = scores.masked_fill(~attn_mask, -1e9)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        hidden = torch.matmul(probs, v)
        hidden = hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head)
        return self.output_layer(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, heads, dropout):
        super().__init__()
        self.attn_norm = LayerNorm(hidden_size)
        self.ffn_norm = LayerNorm(hidden_size)
        self.attention = MultiHeadedAttention(heads, hidden_size, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        hidden = hidden + self.dropout(self.attention(self.attn_norm(hidden), self.attn_norm(hidden), self.attn_norm(hidden), mask))
        hidden = hidden + self.dropout(self.feed_forward(self.ffn_norm(hidden)))
        return hidden


class DiffuRecApproximator(nn.Module):
    def __init__(self, hidden_size, num_blocks, heads, dropout, lambda_uncertainty):
        super().__init__()
        self.hidden_size = hidden_size
        self.lambda_uncertainty = lambda_uncertainty
        time_embed_dim = hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, hidden_size),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, heads, dropout) for _ in range(num_blocks)
        ])
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def timestep_embedding(self, timesteps):
        half = self.hidden_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.hidden_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, seq_emb, x_t, t, mask):
        time_emb = self.time_embed(self.timestep_embedding(t))
        noise_scale = torch.normal(
            mean=torch.full_like(seq_emb, self.lambda_uncertainty),
            std=torch.full_like(seq_emb, self.lambda_uncertainty),
        )
        hidden = seq_emb + noise_scale * (x_t + time_emb).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.norm(self.dropout(hidden))

        lengths = mask.long().sum(dim=1).clamp(min=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]


class DiffuRecCore(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.diffusion_steps = args.diffusion_steps
        self.rescale_timesteps = args.rescale_timesteps
        betas = _beta_schedule(args.noise_schedule, args.diffusion_steps)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.approximator = DiffuRecApproximator(
            args.hidden_size,
            args.num_blocks,
            args.attention_heads,
            args.dropout,
            args.lambda_uncertainty,
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.diffusion_steps)
        return t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean(self, x_start, x_t, t):
        return (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

    def p_sample(self, seq_emb, x_t, t, mask):
        x_start = self.approximator(seq_emb, x_t, self._scale_timesteps(t), mask)
        model_mean = self.q_posterior_mean(x_start, x_t, t)
        log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(log_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1)
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def reverse_sample(self, seq_emb, mask):
        x_t = torch.randn(seq_emb.size(0), seq_emb.size(-1), device=seq_emb.device)
        for step in reversed(range(self.diffusion_steps)):
            t = torch.full((seq_emb.size(0),), step, device=seq_emb.device, dtype=torch.long)
            x_t = self.p_sample(seq_emb, x_t, t, mask)
        return x_t

    def forward(self, seq_emb, target_emb, mask):
        t = torch.randint(0, self.diffusion_steps, (seq_emb.size(0),), device=seq_emb.device)
        x_t = self.q_sample(target_emb, t)
        return self.approximator(seq_emb, x_t, self._scale_timesteps(t), mask)


class LightGCNDiffusion(GeneralRecommender):
    """DiffuRec model adapted to the existing RecBole training pipeline."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.embedding_size = config['embedding_size']
        self.max_len = int(config['diffurec_max_len'])
        self.time_field = config['TIME_FIELD']

        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(float(config['diffurec_emb_dropout']))
        self.layer_norm = LayerNorm(self.embedding_size)

        args = SimpleNamespace(
            hidden_size=self.embedding_size,
            diffusion_steps=int(config['diffusion_steps']),
            noise_schedule=config['diffusion_schedule'],
            rescale_timesteps=bool(config['diffurec_rescale_timesteps']),
            num_blocks=int(config['diffurec_num_blocks']),
            attention_heads=int(config['diffurec_attention_heads']),
            dropout=float(config['diffurec_dropout']),
            lambda_uncertainty=float(config['diffurec_lambda_uncertainty']),
        )
        self.diffusion = DiffuRecCore(args)
        self.loss_ce = nn.CrossEntropyLoss()

        self.user_history = self._build_user_history(dataset).to(self.device)
        self.restore_user_e = None
        self.restore_item_e = None

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e', 'user_history']

    def _build_user_history(self, dataset):
        history = [[] for _ in range(dataset.user_num)]
        inter_feat = dataset.inter_feat
        user_ids = inter_feat[self.USER_ID].cpu().numpy()
        item_ids = inter_feat[self.ITEM_ID].cpu().numpy()
        if self.time_field in inter_feat.interaction:
            times = inter_feat[self.time_field].cpu().numpy()
        else:
            times = np.arange(len(user_ids))

        rows = sorted(zip(user_ids, item_ids, times), key=lambda x: (x[0], x[2]))
        for user, item, _ in rows:
            if item != 0:
                history[int(user)].append(int(item))

        padded = np.zeros((dataset.user_num, self.max_len), dtype=np.int64)
        for user, seq in enumerate(history):
            seq = seq[-self.max_len:]
            if seq:
                padded[user, -len(seq):] = seq
        return torch.LongTensor(padded)

    def _sequence_for_batch(self, users, target_items=None):
        seq = self.user_history[users].clone()
        if target_items is not None:
            for row, target in enumerate(target_items.tolist()):
                positions = (seq[row] == target).nonzero(as_tuple=False)
                if len(positions) > 0:
                    seq[row, positions[-1].item()] = 0
                    compact = seq[row][seq[row] > 0]
                    seq[row].zero_()
                    if len(compact) > 0:
                        seq[row, -len(compact):] = compact[-self.max_len:]
        return seq

    def _encode_sequence(self, seq):
        mask = (seq > 0).float()
        seq_emb = self.item_embedding(seq)
        seq_emb = self.embed_dropout(seq_emb)
        seq_emb = self.layer_norm(seq_emb)
        return seq_emb, mask

    def _user_representations(self, users, target_items=None, reverse=False):
        seq = self._sequence_for_batch(users, target_items)
        seq_emb, mask = self._encode_sequence(seq)
        if reverse:
            return self.diffusion.reverse_sample(seq_emb, mask)

        lengths = mask.long().sum(dim=1).clamp(min=1) - 1
        return seq_emb[torch.arange(seq_emb.size(0), device=seq_emb.device), lengths]

    def forward(self):
        users = torch.arange(self.n_users, device=self.device)
        user_reps = []
        for chunk in users.split(256):
            user_reps.append(self._user_representations(chunk, reverse=True))
        return torch.cat(user_reps, dim=0), self.item_embedding.weight

    def calculate_loss(self, interaction):
        self.restore_user_e, self.restore_item_e = None, None
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        seq = self._sequence_for_batch(users, pos_items)
        seq_emb, mask = self._encode_sequence(seq)
        target_emb = self.item_embedding(pos_items)
        rep_diffu = self.diffusion(seq_emb, target_emb, mask)
        scores = torch.matmul(rep_diffu, self.item_embedding.weight.t())
        return self.loss_ce(scores, pos_items)

    def predict(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        user_rep = self._user_representations(users, reverse=True)
        item_rep = self.item_embedding(items)
        return torch.sum(user_rep * item_rep, dim=1)

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        user_rep = self.restore_user_e[users]
        scores = torch.matmul(user_rep, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)

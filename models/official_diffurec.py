import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.step_sample import create_named_schedule_sampler


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == 'linear':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    if schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps, lambda t: 1 - np.sqrt(t + 0.0001)
        )
    if schedule_name == 'trunc_cos':
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    if schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64)
        return np.concatenate([first_part, second_part])
    raise NotImplementedError(f'unknown beta schedule: {schedule_name}')


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = [min(1 - alpha_bar(0), max_beta)]
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith('ddim'):
            desired_count = int(section_counts[len('ddim'):])
            for stride in range(1, num_timesteps):
                if len(range(0, num_timesteps, stride)) == desired_count:
                    return set(range(0, num_timesteps, stride))
            raise ValueError(
                f'cannot create exactly {num_timesteps} steps with an integer stride'
            )
        section_counts = [int(x) for x in section_counts.split(',')]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for idx, section_count in enumerate(section_counts):
        size = size_per + (1 if idx < extra else 0)
        if size < section_count:
            raise ValueError(
                f'cannot divide section of {size} steps into {section_count}'
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (
            1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3)))
        )
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError('hidden_size must be divisible by attention heads')
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [
            layer(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (q, k, v))
        ]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat(
                [1, 1, 1, corr.shape[-1]]
            )
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(
            hidden.transpose(1, 2).contiguous().view(
                batch_size, -1, self.num_heads * self.size_head
            )
        )
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(
            heads=attn_heads, hidden_size=hidden_size, dropout=dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            hidden_size=hidden_size, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(
            hidden,
            lambda value: self.attention.forward(value, value, value, mask=mask),
        )
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class TransformerRep(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.heads = args.attention_heads
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.hidden_size, self.heads, self.dropout)
                for _ in range(self.n_blocks)
            ]
        )

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


class DiffuXStart(nn.Module):
    def __init__(self, hidden_size, args):
        super().__init__()
        self.hidden_size = hidden_size
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, self.hidden_size),
        )
        self.att = TransformerRep(args)
        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period)
            * th.arange(start=0, end=half, dtype=th.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        x_t = x_t + emb_t
        lambda_uncertainty = th.normal(
            mean=th.full(rep_item.shape, self.lambda_uncertainty),
            std=th.full(rep_item.shape, self.lambda_uncertainty),
        ).to(x_t.device)
        rep_diffu = self.att(rep_item + lambda_uncertainty * x_t.unsqueeze(1), mask_seq)
        rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))
        out = rep_diffu[:, -1, :]
        return out, rep_diffu


class DiffuRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])
        self.noise_schedule = args.noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.num_timesteps = int(self.betas.shape[0])
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_name, self.num_timesteps
        )
        self.rescale_timesteps = args.rescale_timesteps
        self.xstart_model = DiffuXStart(self.hidden_size, args)

    def get_betas(self, noise_schedule, diffusion_steps):
        return get_named_beta_schedule(noise_schedule, diffusion_steps)

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        if mask is None:
            return x_t
        mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
        return th.where(mask == 0, x_start, x_t)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq):
        model_output, _ = self.xstart_model(rep_item, x_t, self._scale_timesteps(t), mask_seq)
        x_0 = model_output
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq):
        model_mean, model_log_variance = self.p_mean_variance(
            item_rep, noise_x_t, t, mask_seq
        )
        noise = th.randn_like(noise_x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1)))
        return model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise

    def reverse_p_sample(self, item_rep, noise_x_t, mask_seq):
        device = next(self.xstart_model.parameters()).device
        indices = list(range(self.num_timesteps))[::-1]
        for idx in indices:
            t = th.tensor([idx] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.p_sample(item_rep, noise_x_t, t, mask_seq)
        return noise_x_t

    def forward(self, item_rep, item_tag, mask_seq):
        noise = th.randn_like(item_tag)
        t, weights = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device)
        x_t = self.q_sample(item_tag, t, noise=noise)
        x_0, item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)
        return x_0, item_rep_out, weights, t


class AttDiffuseModel(nn.Module):
    def __init__(self, diffu, args):
        super().__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num + 1
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()

    def diffu_pre(self, item_rep, tag_emb, mask_seq):
        return self.diffu(item_rep, tag_emb, mask_seq)

    def reverse(self, item_rep, noise_x_t, mask_seq):
        return self.diffu.reverse_p_sample(item_rep, noise_x_t, mask_seq)

    def loss_diffu_ce(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))

    def diffu_rep_pre(self, rep_diffu):
        return torch.matmul(rep_diffu, self.item_embeddings.weight.t())

    def encode_sequence(self, sequence):
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)
        item_embeddings = self.layer_norm(item_embeddings)
        mask_seq = (sequence > 0).float()
        return item_embeddings, mask_seq

    def forward(self, sequence, tag, train_flag=True):
        item_embeddings, mask_seq = self.encode_sequence(sequence)
        if train_flag:
            tag_emb = self.item_embeddings(tag.squeeze(-1))
            rep_diffu, rep_item, weights, t = self.diffu_pre(
                item_embeddings, tag_emb, mask_seq
            )
        else:
            noise_x_t = th.randn_like(item_embeddings[:, -1, :])
            rep_diffu = self.reverse(item_embeddings, noise_x_t, mask_seq)
            rep_item, weights, t = None, None, None
        return None, rep_diffu, weights, t, None, None

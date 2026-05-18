"""
RecBole wrapper around the official DiffuRec core implementation.

The class name stays `LightGCNDiffusion` to preserve local import paths in the
project, but the internals now delegate to the vendored official DiffuRec
modules with only minimal shape and padding adaptation.
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import SequentialRecommender

from models.official_diffurec import AttDiffuseModel, DiffuRec


class LightGCNDiffusion(SequentialRecommender):
    """Official DiffuRec wrapped as a native RecBole sequential recommender."""

    def __init__(self, config, dataset_or_item_num):
        def cfg(name, default=None):
            try:
                return config[name]
            except KeyError:
                return default

        if isinstance(dataset_or_item_num, int):
            nn.Module.__init__(self)
            self.USER_ID = config['USER_ID_FIELD']
            self.ITEM_ID = config['ITEM_ID_FIELD']
            self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
            self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
            self.POS_ITEM_ID = self.ITEM_ID
            self.NEG_ITEM_ID = cfg('NEG_PREFIX', 'neg_') + self.ITEM_ID
            self.max_seq_length = int(config['MAX_ITEM_LIST_LENGTH'])
            self.n_items = int(dataset_or_item_num)
            self.device = config['device']
        else:
            super().__init__(config, dataset_or_item_num)

        self.embedding_size = int(config['embedding_size'])
        self.max_len = int(config['MAX_ITEM_LIST_LENGTH'])
        self.loss_type = cfg('loss_type', 'CE')

        args = SimpleNamespace(
            hidden_size=self.embedding_size,
            item_num=self.n_items - 1,
            emb_dropout=float(config['diffurec_emb_dropout']),
            dropout=float(config['diffurec_dropout']),
            max_len=self.max_len,
            diffusion_steps=int(config['diffusion_steps']),
            noise_schedule=config['diffusion_schedule'],
            rescale_timesteps=bool(config['diffurec_rescale_timesteps']),
            num_blocks=int(config['diffurec_num_blocks']),
            attention_heads=int(config['diffurec_attention_heads']),
            lambda_uncertainty=float(config['diffurec_lambda_uncertainty']),
            schedule_sampler_name=cfg('schedule_sampler_name', 'uniform'),
        )
        diffu_core = DiffuRec(args)
        self.model = AttDiffuseModel(diffu_core, args)
        self._reset_padding_embedding()

    def _reset_padding_embedding(self):
        with torch.no_grad():
            self.model.item_embeddings.weight[0].fill_(0)

    def _left_pad_sequence(self, item_seq, item_seq_len):
        if item_seq.size(1) == 0:
            return item_seq
        left_padded = torch.zeros_like(item_seq)
        for row_idx in range(item_seq.size(0)):
            length = int(item_seq_len[row_idx].item())
            if length <= 0:
                continue
            left_padded[row_idx, -length:] = item_seq[row_idx, :length]
        return left_padded

    def _sequence_inputs(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = self._left_pad_sequence(item_seq, item_seq_len)
        return item_seq, item_seq_len

    def calculate_loss(self, interaction):
        item_seq, _item_seq_len = self._sequence_inputs(interaction)
        pos_items = interaction[self.POS_ITEM_ID].unsqueeze(-1)
        _scores, rep_diffu, _weights, _t, _item_rep_dis, _seq_rep_dis = self.model(
            item_seq, pos_items, train_flag=True
        )
        return self.model.loss_diffu_ce(rep_diffu, pos_items)

    def forward(self, item_seq, item_seq_len):
        item_seq = self._left_pad_sequence(item_seq, item_seq_len)
        _scores, rep_diffu, _weights, _t, _item_rep_dis, _seq_rep_dis = self.model(
            item_seq,
            torch.zeros(item_seq.size(0), 1, dtype=torch.long, device=item_seq.device),
            train_flag=False,
        )
        return rep_diffu

    def predict(self, interaction):
        item_seq, item_seq_len = self._sequence_inputs(interaction)
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.model.item_embeddings(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores.masked_fill(test_item == 0, -1e9)

    def full_sort_predict(self, interaction):
        if isinstance(interaction, tuple):
            interaction = interaction[0]
        item_seq, item_seq_len = self._sequence_inputs(interaction)
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.model.item_embeddings.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        scores[:, 0] = -1e9
        return scores

    def user_representation(self, sequences):
        item_seq_len = (sequences > 0).sum(dim=1)
        return self.forward(sequences, item_seq_len)

    def export_item_embeddings(self):
        return self.model.item_embeddings.weight.detach()

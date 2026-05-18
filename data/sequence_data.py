import torch
from torch.utils.data import Dataset


def _token_to_int(token):
    if token in (None, '[PAD]'):
        return 0
    return int(token)


def _to_numpy(column):
    if hasattr(column, 'cpu'):
        return column.cpu().numpy()
    if hasattr(column, 'numpy'):
        return column.numpy()
    return column.to_numpy()


def build_user_sequences(dataset, user_field, item_field, time_field):
    inter_feat = dataset.inter_feat
    user_ids = _to_numpy(inter_feat[user_field])
    item_ids = _to_numpy(inter_feat[item_field])
    if hasattr(inter_feat, 'interaction') and time_field in inter_feat.interaction:
        timestamps = _to_numpy(inter_feat[time_field])
    elif time_field in inter_feat:
        timestamps = _to_numpy(inter_feat[time_field])
    else:
        timestamps = list(range(len(user_ids)))

    user_sequences = {}
    for user_id, item_id, timestamp in sorted(
        zip(user_ids, item_ids, timestamps), key=lambda row: (row[0], row[2])
    ):
        if item_id == 0:
            continue
        user_sequences.setdefault(int(user_id), []).append(int(item_id))
    return user_sequences


def split_user_sequences(user_sequences):
    train_sequences = {}
    val_sequences = {}
    test_sequences = {}

    for user_id, sequence in user_sequences.items():
        if len(sequence) >= 3:
            train_sequences[user_id] = sequence[:-2]
            val_sequences[user_id] = [sequence[-2]]
            test_sequences[user_id] = [sequence[-1]]
        elif len(sequence) == 2:
            train_sequences[user_id] = sequence[:-1]
            val_sequences[user_id] = [sequence[-1]]
        elif len(sequence) == 1:
            train_sequences[user_id] = sequence
    return train_sequences, val_sequences, test_sequences


class PrefixTrainDataset(Dataset):
    def __init__(self, user_sequences, max_len):
        self.max_len = max_len
        self.samples = []
        for sequence in user_sequences.values():
            for end_idx in range(1, len(sequence)):
                prefix = sequence[:end_idx]
                target = sequence[end_idx]
                self.samples.append((prefix, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        prefix, target = self.samples[index]
        prefix = prefix[-self.max_len:]
        padding_len = self.max_len - len(prefix)
        sequence = [0] * padding_len + prefix
        return torch.LongTensor(sequence), torch.LongTensor([target])


class NextItemEvalDataset(Dataset):
    def __init__(self, contexts, answers, max_len):
        self.users = sorted(answers.keys())
        self.contexts = contexts
        self.answers = answers
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user_id = self.users[index]
        sequence = self.contexts[user_id][-self.max_len:]
        padding_len = self.max_len - len(sequence)
        sequence = [0] * padding_len + sequence
        answer = self.answers[user_id][0]
        return (
            torch.LongTensor(sequence),
            torch.LongTensor([answer]),
            torch.LongTensor([user_id]),
        )


def build_raw_id_mappings(dataset, user_field, item_field):
    user_tokens = dataset.field2id_token[user_field]
    item_tokens = dataset.field2id_token[item_field]

    user_internal_to_raw = [_token_to_int(token) for token in user_tokens]
    item_internal_to_raw = [_token_to_int(token) for token in item_tokens]

    raw_user_to_internal = {
        raw_id: internal_id
        for internal_id, raw_id in enumerate(user_internal_to_raw)
        if raw_id != 0
    }
    raw_item_to_internal = {
        raw_id: internal_id
        for internal_id, raw_id in enumerate(item_internal_to_raw)
        if raw_id != 0
    }

    return {
        'user_internal_to_raw': user_internal_to_raw,
        'item_internal_to_raw': item_internal_to_raw,
        'raw_user_to_internal': raw_user_to_internal,
        'raw_item_to_internal': raw_item_to_internal,
    }

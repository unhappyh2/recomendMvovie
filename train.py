"""
训练脚本: Sequential DiffuRec 模型训练与评估
使用 RecBole 负责数据读取与 id 映射，训练/评估逻辑走自定义序列范式。
"""
import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import ensure_dir, init_logger, set_color

from data.sequence_data import (
    NextItemEvalDataset,
    PrefixTrainDataset,
    build_raw_id_mappings,
    build_user_sequences,
    split_user_sequences,
)
from models.lightgcn_diffusion import LightGCNDiffusion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_progress', action='store_true')
    return parser.parse_args()


def build_dataloaders(config, dataset):
    user_field = config['USER_ID_FIELD']
    item_field = config['ITEM_ID_FIELD']
    time_field = config['TIME_FIELD']

    user_sequences = build_user_sequences(dataset, user_field, item_field, time_field)
    train_sequences, val_answers, test_answers = split_user_sequences(user_sequences)

    train_dataset = PrefixTrainDataset(train_sequences, int(config['diffurec_max_len']))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['train_batch_size']),
        shuffle=True,
    )

    val_contexts = {user_id: train_sequences[user_id] for user_id in val_answers}
    val_dataset = NextItemEvalDataset(
        val_contexts,
        val_answers,
        int(config['diffurec_max_len']),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config['eval_batch_size']),
        shuffle=False,
    )

    test_contexts = {
        user_id: train_sequences[user_id] + val_answers[user_id]
        for user_id in test_answers
        if user_id in val_answers
    }
    filtered_test_answers = {
        user_id: test_answers[user_id]
        for user_id in test_contexts
    }
    test_dataset = NextItemEvalDataset(
        test_contexts,
        filtered_test_answers,
        int(config['diffurec_max_len']),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config['eval_batch_size']),
        shuffle=False,
    )

    return train_sequences, val_answers, filtered_test_answers, train_loader, val_loader, test_loader


def recall_ndcg(scores, labels, ks):
    max_k = max(ks)
    _, topk_indices = torch.topk(scores, k=max_k, dim=-1)
    hit = topk_indices.eq(labels.unsqueeze(1))

    metrics = {}
    for k in ks:
        hit_k = hit[:, :k]
        recall = hit_k.any(dim=1).float().mean().item()
        ndcg = 0.0
        if hit_k.any():
            positions = torch.argmax(hit_k.float(), dim=1)
            gains = torch.where(
                hit_k.any(dim=1),
                1.0 / torch.log2(positions.float() + 2.0),
                torch.zeros_like(positions, dtype=torch.float32),
            )
            ndcg = gains.mean().item()
        metrics[f'Recall@{k}'] = recall
        metrics[f'NDCG@{k}'] = ndcg
    return metrics


def mask_seen_items(scores, sequences):
    scores = scores.clone()
    for row_idx in range(sequences.size(0)):
        seen = sequences[row_idx][sequences[row_idx] > 0].unique()
        scores[row_idx, seen] = -1e9
    scores[:, 0] = -1e9
    return scores


def evaluate_model(model, data_loader, device, ks):
    if len(data_loader.dataset) == 0:
        empty_metrics = {f'Recall@{k}': 0.0 for k in ks}
        empty_metrics.update({f'NDCG@{k}': 0.0 for k in ks})
        return empty_metrics

    results = {f'Recall@{k}': [] for k in ks}
    results.update({f'NDCG@{k}': [] for k in ks})

    model.eval()
    with torch.no_grad():
        for sequences, labels, _user_ids in data_loader:
            sequences = sequences.to(device)
            labels = labels.squeeze(-1).to(device)
            scores = model.full_sort_predict(sequences)
            scores = mask_seen_items(scores, sequences)
            batch_metrics = recall_ndcg(scores, labels, ks)
            for key, value in batch_metrics.items():
                results[key].append(value)

    return {
        key: float(np.mean(values)) if values else 0.0
        for key, values in results.items()
    }


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    sample_count = 0
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.squeeze(-1).to(device)
        optimizer.zero_grad()
        loss = model.calculate_loss(sequences, labels)
        loss.backward()
        optimizer.step()

        batch_size = sequences.size(0)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
    return total_loss / max(sample_count, 1)


def build_checkpoint_payload(config, dataset, model, train_sequences, test_metrics):
    mappings = build_raw_id_mappings(
        dataset,
        config['USER_ID_FIELD'],
        config['ITEM_ID_FIELD'],
    )
    raw_train_sequences = {
        mappings['user_internal_to_raw'][user_id]: sequence
        for user_id, sequence in train_sequences.items()
        if user_id < len(mappings['user_internal_to_raw'])
        and mappings['user_internal_to_raw'][user_id] != 0
    }

    model_config = {
        'embedding_size': int(config['embedding_size']),
        'diffurec_max_len': int(config['diffurec_max_len']),
        'diffurec_num_blocks': int(config['diffurec_num_blocks']),
        'diffurec_attention_heads': int(config['diffurec_attention_heads']),
        'diffurec_dropout': float(config['diffurec_dropout']),
        'diffurec_emb_dropout': float(config['diffurec_emb_dropout']),
        'diffurec_lambda_uncertainty': float(config['diffurec_lambda_uncertainty']),
        'diffurec_rescale_timesteps': bool(config['diffurec_rescale_timesteps']),
        'diffusion_steps': int(config['diffusion_steps']),
        'diffusion_schedule': config['diffusion_schedule'],
    }

    return {
        'model_state': model.state_dict(),
        'model_config': model_config,
        'dataset_name': config['dataset'],
        'item_num': model.n_items,
        'item_embeddings': model.export_item_embeddings().cpu(),
        'user_sequences': raw_train_sequences,
        'raw_user_to_internal': mappings['raw_user_to_internal'],
        'raw_item_to_internal': mappings['raw_item_to_internal'],
        'user_internal_to_raw': mappings['user_internal_to_raw'],
        'item_internal_to_raw': mappings['item_internal_to_raw'],
        'metrics': test_metrics,
    }


def main():
    args = parse_args()

    config = Config(model='BPR', dataset='ml-100k', config_file_list=[args.config])
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.device:
        config['device'] = torch.device(args.device)

    set_seed(config['seed'])

    init_logger(config)
    logger = logging.getLogger()
    logger.info('=' * 60)
    logger.info('Sequential DiffuRec 推荐模型训练')
    logger.info('=' * 60)

    logger.info(set_color('Loading dataset...', 'green'))
    dataset = create_dataset(config)
    logger.info(f'Users: {dataset.user_num}, Items: {dataset.item_num}')
    logger.info(f'Filtered interactions: {dataset.inter_num}')

    (
        train_sequences,
        val_answers,
        test_answers,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(config, dataset)
    logger.info(f'Train users: {len(train_sequences)}')
    logger.info(f'Validation users: {len(val_answers)}')
    logger.info(f'Test users: {len(test_answers)}')
    logger.info(f'Train samples: {len(train_loader.dataset)}')

    logger.info(set_color('Building model...', 'green'))
    model = LightGCNDiffusion(config, dataset.item_num).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']))

    ks = list(config['topk'])
    best_state = None
    best_valid_score = float('-inf')
    best_valid_metrics = None

    logger.info(set_color('Start training...', 'green'))
    for epoch_idx in range(int(config['epochs'])):
        train_loss = train_one_epoch(model, train_loader, optimizer, config['device'])
        valid_metrics = evaluate_model(model, val_loader, config['device'], ks)
        valid_score = valid_metrics.get(config['valid_metric'], 0.0)

        logger.info(
            f'Epoch {epoch_idx + 1}/{config["epochs"]} '
            f'loss={train_loss:.4f} valid_{config["valid_metric"]}={valid_score:.4f}'
        )

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_valid_metrics = valid_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(f'Best valid result: {best_valid_metrics}')

    logger.info(set_color('Evaluating on test set...', 'green'))
    test_result = evaluate_model(model, test_loader, config['device'], ks)
    logger.info(f'Test result: {test_result}')

    ensure_dir(config['checkpoint_dir'])
    save_path = os.path.join(config['checkpoint_dir'], 'model_checkpoint.pt')
    checkpoint = build_checkpoint_payload(
        config,
        dataset,
        model,
        train_sequences,
        test_result,
    )
    torch.save(checkpoint, save_path)
    logger.info(f'Checkpoint saved to {save_path}')

    print('\nTest metrics:')
    for key, value in test_result.items():
        print(f'  {key}: {value:.4f}')


if __name__ == '__main__':
    main()

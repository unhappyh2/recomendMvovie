"""
训练脚本: 使用 RecBole 加载数据，按官方 DiffuRec 逻辑训练与评估
"""
import argparse
import copy
import logging
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import ensure_dir, init_logger, set_color

from data.sequence_data import (
    PrefixTrainDataset,
    NextItemEvalDataset,
    build_raw_id_mappings,
    build_user_sequences,
    split_user_sequences,
)
from models.official_diffurec import AttDiffuseModel, DiffuRec


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
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--no_progress', action='store_true')
    return parser.parse_args()


def build_model_args(config, dataset):
    return SimpleNamespace(
        hidden_size=int(config['hidden_size']),
        item_num=int(dataset.item_num - 1),
        emb_dropout=float(config['emb_dropout']),
        dropout=float(config['dropout']),
        max_len=int(config['MAX_ITEM_LIST_LENGTH']),
        diffusion_steps=int(config['diffusion_steps']),
        noise_schedule=config['noise_schedule'],
        rescale_timesteps=bool(config['rescale_timesteps']),
        num_blocks=int(config['num_blocks']),
        attention_heads=int(config['attention_heads']),
        lambda_uncertainty=float(config['lambda_uncertainty']),
        schedule_sampler_name=config['schedule_sampler_name'],
    )


def build_eval_contexts(train_sequences, val_sequences, test_sequences):
    valid_contexts = {
        user_id: list(train_sequences[user_id])
        for user_id in val_sequences
        if user_id in train_sequences
    }
    test_contexts = {}
    for user_id in test_sequences:
        base = list(train_sequences.get(user_id, []))
        base.extend(val_sequences.get(user_id, []))
        test_contexts[user_id] = base
    return valid_contexts, test_contexts


def create_loaders(config, user_sequences):
    train_sequences, val_sequences, test_sequences = split_user_sequences(user_sequences)
    max_len = int(config['MAX_ITEM_LIST_LENGTH'])

    train_dataset = PrefixTrainDataset(train_sequences, max_len)
    valid_contexts, test_contexts = build_eval_contexts(
        train_sequences, val_sequences, test_sequences
    )
    valid_dataset = NextItemEvalDataset(valid_contexts, val_sequences, max_len)
    test_dataset = NextItemEvalDataset(test_contexts, test_sequences, max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['train_batch_size']),
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config['eval_batch_size']),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config['eval_batch_size']),
        shuffle=False,
    )
    return train_loader, valid_loader, test_loader


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size(-1) + 1, device=hit.device) + 1).unsqueeze(0)
    return (hit / log2).sum(dim=-1)


def cal_hr(labels, scores, ks):
    max_k = max(ks)
    _, topk = torch.topk(scores, k=max_k, dim=-1)
    hit = labels == topk
    return [hit[:, :k].sum().item() / labels.size(0) for k in ks]


def cal_ndcg(labels, scores, ks):
    max_k = max(ks)
    _, topk = torch.topk(scores, k=max_k, dim=-1)
    hit = (labels == topk).int()
    ndcg_values = []
    for k in ks:
        max_dcg = dcg(torch.tensor([[1] + [0] * (k - 1)], device=hit.device, dtype=torch.float32))
        pred_dcg = dcg(hit[:, :k].float())
        ndcg_values.append((pred_dcg / max_dcg).mean().item())
    return ndcg_values


def evaluate(model, data_loader, device, metric_ks):
    model.eval()
    metrics_dict = {f'Recall@{k}': [] for k in metric_ks}
    metrics_dict.update({f'NDCG@{k}': [] for k in metric_ks})

    with torch.no_grad():
        for sequences, labels, _user_ids in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            _scores, rep_diffu, _weights, _t, _item_rep_dis, _seq_rep_dis = model(
                sequences, labels, train_flag=False
            )
            scores = model.diffu_rep_pre(rep_diffu)

            hr_values = cal_hr(labels, scores, metric_ks)
            ndcg_values = cal_ndcg(labels, scores, metric_ks)
            for k, hr, ndcg in zip(metric_ks, hr_values, ndcg_values):
                metrics_dict[f'Recall@{k}'].append(hr)
                metrics_dict[f'NDCG@{k}'].append(ndcg)

    return {
        metric_name: float(np.mean(values)) if values else 0.0
        for metric_name, values in metrics_dict.items()
    }


def train_one_epoch(model, data_loader, optimizer, device, logger, epoch_idx):
    model.train()
    last_loss = None

    for batch_idx, (sequences, labels) in enumerate(data_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        _scores, rep_diffu, _weights, _t, _item_rep_dis, _seq_rep_dis = model(
            sequences, labels, train_flag=True
        )
        loss = model.loss_diffu_ce(rep_diffu, labels)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        report_every = max(1, len(data_loader) // 5)
        if batch_idx % report_every == 0:
            logger.info('[%d/%d] Loss: %.4f', batch_idx, len(data_loader), last_loss)

    logger.info('Epoch %d loss: %.4f', epoch_idx, last_loss if last_loss is not None else -1.0)
    return last_loss if last_loss is not None else 0.0


def build_checkpoint_payload(config, dataset, model, test_metrics):
    mappings = build_raw_id_mappings(
        dataset,
        config['USER_ID_FIELD'],
        config['ITEM_ID_FIELD'],
    )
    user_sequences = build_user_sequences(
        dataset,
        config['USER_ID_FIELD'],
        config['ITEM_ID_FIELD'],
        config['TIME_FIELD'],
    )
    raw_user_sequences = {
        mappings['user_internal_to_raw'][user_id]: sequence
        for user_id, sequence in user_sequences.items()
        if user_id < len(mappings['user_internal_to_raw'])
        and mappings['user_internal_to_raw'][user_id] != 0
    }

    model_config = {
        'USER_ID_FIELD': config['USER_ID_FIELD'],
        'ITEM_ID_FIELD': config['ITEM_ID_FIELD'],
        'LIST_SUFFIX': config['LIST_SUFFIX'],
        'ITEM_LIST_LENGTH_FIELD': config['ITEM_LIST_LENGTH_FIELD'],
        'MAX_ITEM_LIST_LENGTH': int(config['MAX_ITEM_LIST_LENGTH']),
        'hidden_size': int(config['hidden_size']),
        'emb_dropout': float(config['emb_dropout']),
        'dropout': float(config['dropout']),
        'diffusion_steps': int(config['diffusion_steps']),
        'noise_schedule': config['noise_schedule'],
        'rescale_timesteps': bool(config['rescale_timesteps']),
        'num_blocks': int(config['num_blocks']),
        'attention_heads': int(config['attention_heads']),
        'lambda_uncertainty': float(config['lambda_uncertainty']),
        'schedule_sampler_name': config['schedule_sampler_name'],
        'device': 'cpu',
    }

    return {
        'model_state': model.state_dict(),
        'model_config': model_config,
        'dataset_name': config['dataset'],
        'item_num': model.item_num,
        'item_embeddings': model.item_embeddings.weight.detach().cpu(),
        'user_sequences': raw_user_sequences,
        'raw_user_to_internal': mappings['raw_user_to_internal'],
        'raw_item_to_internal': mappings['raw_item_to_internal'],
        'user_internal_to_raw': mappings['user_internal_to_raw'],
        'item_internal_to_raw': mappings['item_internal_to_raw'],
        'metrics': test_metrics,
    }


def main():
    args = parse_args()

    config = Config(model='SASRec', dataset='ml-100k', config_file_list=[args.config])
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.device:
        config['device'] = torch.device(args.device)
    else:
        config['device'] = torch.device(str(config['device']))

    set_seed(config['seed'])
    init_logger(config)
    logger = logging.getLogger()

    logger.info('=' * 60)
    logger.info('Official-style DiffuRec 推荐模型训练')
    logger.info('=' * 60)

    logger.info(set_color('Loading dataset...', 'green'))
    dataset = create_dataset(config)
    logger.info(f'Users: {dataset.user_num}, Items: {dataset.item_num}')
    logger.info(f'Filtered interactions: {dataset.inter_num}')

    user_sequences = build_user_sequences(
        dataset,
        config['USER_ID_FIELD'],
        config['ITEM_ID_FIELD'],
        config['TIME_FIELD'],
    )
    train_loader, valid_loader, test_loader = create_loaders(config, user_sequences)
    logger.info(
        '|Train|=%d, |Valid|=%d, |Test|=%d',
        len(train_loader.dataset),
        len(valid_loader.dataset),
        len(test_loader.dataset),
    )

    model_args = build_model_args(config, dataset)
    model = AttDiffuseModel(DiffuRec(model_args), model_args).to(config['device'])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Parameters: {trainable_params:,} trainable / {total_params:,} total')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config['decay_step']),
        gamma=float(config['gamma']),
    )

    metric_ks = list(config['topk'])
    valid_metric = str(config['valid_metric'])
    best_model = copy.deepcopy(model)
    best_valid = -1.0
    best_valid_result = {}
    bad_count = 0

    for epoch_idx in range(int(config['epochs'])):
        logger.info('Epoch: %d', epoch_idx)
        train_one_epoch(model, train_loader, optimizer, config['device'], logger, epoch_idx)
        scheduler.step()

        should_eval = epoch_idx != 0 and epoch_idx % int(config['eval_interval']) == 0
        if not should_eval:
            continue

        logger.info('Start validation at epoch %d', epoch_idx)
        valid_result = evaluate(model, valid_loader, config['device'], metric_ks)
        logger.info('Valid result: %s', valid_result)

        current = valid_result.get(valid_metric, 0.0)
        if current > best_valid:
            best_valid = current
            best_valid_result = valid_result
            best_model = copy.deepcopy(model)
            bad_count = 0
        else:
            bad_count += 1

        if bad_count >= int(config['patience']):
            logger.info('Early stop triggered at epoch %d', epoch_idx)
            break

    test_result = evaluate(best_model, test_loader, config['device'], metric_ks)
    logger.info(f'Best valid {valid_metric}: {best_valid:.4f}')
    logger.info(f'Best valid result: {best_valid_result}')
    logger.info(f'Test result: {test_result}')

    ensure_dir(config['checkpoint_dir'])
    save_path = os.path.join(config['checkpoint_dir'], 'model_checkpoint.pt')
    checkpoint = build_checkpoint_payload(config, dataset, best_model.cpu(), test_result)
    torch.save(checkpoint, save_path)
    logger.info(f'Checkpoint saved to {save_path}')

    print('\nTest metrics:')
    for key, value in test_result.items():
        print(f'  {key}: {value:.4f}')


if __name__ == '__main__':
    main()

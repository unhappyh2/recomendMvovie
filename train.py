"""
训练脚本: RecBole 原生 Trainer 驱动的 DiffuRec 训练与评估
"""
import argparse
import logging
import os
import random

import numpy as np
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import ensure_dir, init_logger, set_color

from data.sequence_data import build_raw_id_mappings, build_user_sequences
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
        'embedding_size': int(config['embedding_size']),
        'diffurec_num_blocks': int(config['diffurec_num_blocks']),
        'diffurec_attention_heads': int(config['diffurec_attention_heads']),
        'diffurec_dropout': float(config['diffurec_dropout']),
        'diffurec_emb_dropout': float(config['diffurec_emb_dropout']),
        'diffurec_lambda_uncertainty': float(config['diffurec_lambda_uncertainty']),
        'diffurec_rescale_timesteps': bool(config['diffurec_rescale_timesteps']),
        'diffusion_steps': int(config['diffusion_steps']),
        'diffusion_schedule': config['diffusion_schedule'],
        'device': 'cpu',
    }

    return {
        'model_state': model.state_dict(),
        'model_config': model_config,
        'dataset_name': config['dataset'],
        'item_num': model.n_items,
        'item_embeddings': model.export_item_embeddings().cpu(),
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

    set_seed(config['seed'])

    init_logger(config)
    logger = logging.getLogger()
    logger.info('=' * 60)
    logger.info('RecBole Trainer DiffuRec 推荐模型训练')
    logger.info('=' * 60)

    logger.info(set_color('Loading dataset...', 'green'))
    dataset = create_dataset(config)
    logger.info(f'Users: {dataset.user_num}, Items: {dataset.item_num}')
    logger.info(f'Filtered interactions: {dataset.inter_num}')

    train_data, valid_data, test_data = data_preparation(config, dataset)
    logger.info(
        f'|Train|={train_data.dataset.inter_num}, '
        f'|Valid|={valid_data.dataset.inter_num}, '
        f'|Test|={test_data.dataset.inter_num}'
    )

    logger.info(set_color('Building model...', 'green'))
    model = LightGCNDiffusion(config, train_data.dataset).to(config['device'])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Parameters: {trainable_params:,} trainable / {total_params:,} total')

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=not args.no_progress,
    )
    logger.info(f'Best valid {config["valid_metric"]}: {best_valid_score:.4f}')
    logger.info(f'Best valid result: {best_valid_result}')

    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=not args.no_progress,
    )
    logger.info(f'Test result: {test_result}')

    ensure_dir(config['checkpoint_dir'])
    save_path = os.path.join(config['checkpoint_dir'], 'model_checkpoint.pt')
    checkpoint = build_checkpoint_payload(config, dataset, trainer.model.cpu(), test_result)
    torch.save(checkpoint, save_path)
    logger.info(f'Checkpoint saved to {save_path}')

    print('\nTest metrics:')
    for key, value in test_result.items():
        print(f'  {key}: {value:.4f}')


if __name__ == '__main__':
    main()

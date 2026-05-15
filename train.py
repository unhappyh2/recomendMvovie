"""
训练脚本: LightGCN 模型训练与评估
基于 RecBole 框架
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
import logging

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, set_color, ensure_dir
from recbole.trainer import Trainer

from models.lightgcn_diffusion import LightGCNDiffusion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_progress', action='store_true')
    args = parser.parse_args()

    # 使用 BPR 作为配置模型名以兼容 RecBole，实际模型为 LightGCNDiffusion（当前仅启用 LightGCN）
    config = Config(model='BPR', dataset='ml-100k',
                    config_file_list=[args.config])
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.device:
        device = torch.device(args.device)
        config['device'] = device

    set_seed(config['seed'])

    init_logger(config)
    logger = logging.getLogger()
    logger.info('=' * 60)
    logger.info('LightGCN 推荐模型训练')
    logger.info('=' * 60)

    logger.info(set_color('Loading dataset...', 'green'))
    dataset = create_dataset(config)
    logger.info(f'Users: {dataset.user_num}, Items: {dataset.item_num}')
    logger.info(f'Train interactions: {dataset.inter_num}')

    train_data, valid_data, test_data = data_preparation(config, dataset)
    logger.info(f'|Train|={train_data.dataset.inter_num}, '
                f'|Valid|={valid_data.dataset.inter_num}, '
                f'|Test|={test_data.dataset.inter_num}')

    logger.info(set_color('Building model...', 'green'))
    model = LightGCNDiffusion(config, train_data._dataset).to(config['device'])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Parameters: {trainable_params:,} trainable / {total_params:,} total')

    logger.info(set_color('Start training...', 'green'))
    trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data,
        saved=True,
        show_progress=not args.no_progress
    )
    logger.info(f'Best valid {config["valid_metric"]}: {best_valid_score:.4f}')
    logger.info(f'Best valid result: {best_valid_result}')

    logger.info(set_color('Evaluating on test set...', 'green'))
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=not args.no_progress
    )
    logger.info(f'Test result: {test_result}')

    # 保存模型和嵌入用于推荐系统
    ensure_dir(config['checkpoint_dir'])
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.forward()
        save_dict = {
            'user_embedding': user_emb.cpu().numpy(),
            'item_embedding': item_emb.cpu().numpy(),
            'model_state': model.state_dict(),
            'config': dict(config.final_config_dict),
            'user_num': model.n_users,
            'item_num': model.n_items,
        }
        save_path = os.path.join(config['checkpoint_dir'], 'model_checkpoint.pt')
        torch.save(save_dict, save_path)
        logger.info(f'Checkpoint saved to {save_path}')

        # 单独保存嵌入供 Web 使用
        np.save(os.path.join(config['checkpoint_dir'], 'user_emb.npy'), user_emb.cpu().numpy())
        np.save(os.path.join(config['checkpoint_dir'], 'item_emb.npy'), item_emb.cpu().numpy())

    logger.info(set_color('Done!', 'green'))
    print(f'\nTest metrics:')
    for k, v in test_result.items():
        print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    main()

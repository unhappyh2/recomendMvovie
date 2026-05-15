"""
评估指标工具
"""
import numpy as np
import torch


def recall_at_k(scores, labels, k):
    """Recall@K"""
    top_k = np.argsort(scores)[::-1][:k]
    hit = len(set(top_k) & set(labels))
    return hit / len(labels) if len(labels) > 0 else 0.0


def precision_at_k(scores, labels, k):
    """Precision@K"""
    top_k = np.argsort(scores)[::-1][:k]
    hit = len(set(top_k) & set(labels))
    return hit / k


def ndcg_at_k(scores, labels, k):
    """NDCG@K"""
    top_k = np.argsort(scores)[::-1][:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in labels:
            dcg += 1.0 / np.log2(i + 2)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(labels), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def hit_rate_at_k(scores, labels, k):
    """Hit Rate@K: 至少命中一个"""
    top_k = np.argsort(scores)[::-1][:k]
    return 1.0 if len(set(top_k) & set(labels)) > 0 else 0.0


def compute_all_metrics(scores, labels, k_list=[10, 20, 50]):
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = recall_at_k(scores, labels, k)
        metrics[f'Precision@{k}'] = precision_at_k(scores, labels, k)
        metrics[f'NDCG@{k}'] = ndcg_at_k(scores, labels, k)
        metrics[f'Hit@{k}'] = hit_rate_at_k(scores, labels, k)
    return metrics

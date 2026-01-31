import math
import torch


def get_metrics(targets, results, device, k):
    metrics = torch.zeros([2]).to(device)
    hits, batch = hit_at_k(targets, results, k)
    ndcg, _ = ndcg_at_k(targets, results, k)
    metrics[0] = hits
    metrics[1] = ndcg
    return metrics, batch


def hit_at_k(labels, results, k):
    '''
    labels.shape = [batch]
    results.shape = [batch, item_num]
    '''
    hit = 0.0
    batch = results.shape[0]
    for i in range(batch):
        res = results[i, :k]
        label = labels[i]
        if label in res:
            hit += 1
    return hit, batch


def ndcg_at_k(labels, results, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    batch = results.shape[0]
    for i in range(batch):
        res = results[i, :k]
        label = labels[i]
        one_ndcg = 0.0
        # Check rank
        # Optimization: use torch.where or similar if k is large, loop is fine for small k
        for j in range(k):
             if res[j] == label:
                 one_ndcg += 1.0 / math.log(j + 2, 2)
                 break
        ndcg += one_ndcg
    return ndcg, batch
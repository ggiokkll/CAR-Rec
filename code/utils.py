import torch
import torch.nn as nn
import sys
import numpy as np
import pandas as pd
import random
import time
import os
import torch.nn.functional as F


def reconstruct(model, emb, device):
    model.to(device)
    model.eval()
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        if hasattr(model, 'mq_semantic'):
            (emb_hat, _), _, _ = model(emb, emb)
        else:
            emb_hat, _, _ = model(emb)
        loss = mse_loss(emb_hat, emb)
        print('test loss:', loss.item())
    return emb_hat


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def read_semantic_embeddings(data_name, device='cpu'):
    path = f'../data/{data_name}/semantic_emb.pt'
    if not os.path.exists(path):
        print(f"[Warning] Semantic file not found at {path}")
        return None
    print(f"[Data] Loading semantic embeddings from {path}")
    emb = torch.load(path, map_location=device)
    return emb


def get_user_head():
    return ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def item_codebook_to_str(vq_id):
    id_num = vq_id.shape[0]
    codebook_num = vq_id.shape[1]
    sample = []
    for i in range(id_num):
        temp = ['item_']
        for j in range(codebook_num):
            token = "".join(["<", str(j), '-', str(vq_id[i, j].item()), '>'])
            temp.append(token)
        temp = " ".join(temp)
        sample.append(temp)
    sample = " ".join(sample)
    return sample


def user_codebook_to_str(vq_id):
    user_head = get_user_head()
    id_num = vq_id.shape[0]
    codebook_num = vq_id.shape[1]
    sample = []
    for i in range(id_num):
        temp = ['user_']
        for j in range(codebook_num):
            head_idx = j if j < len(user_head) else j % len(user_head)
            token = "".join(["<", user_head[head_idx], '-', str(vq_id[i, j].item()), '>'])
            temp.append(token)
        temp = " ".join(temp)
        sample.append(temp)
    sample = " ".join(sample)
    return sample


def group_model_params(model1, model2, decay):
    grouped_params = [
        {
            "params": [p for n, p in model1.named_parameters()],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model2.named_parameters()],
            "weight_decay": decay,
        },
    ]
    return grouped_params


def group_model_emb_params(model1, model2, emb, decay):
    grouped_params = [
        {
            "params": [p for n, p in model1.named_parameters()],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model2.named_parameters()],
            "weight_decay": decay,
        },
        {
            "params": emb.parameters(),
            "weight_decay": decay,
        },
    ]
    return grouped_params


# [Restored Function]
def seq_construct(item_id, train_item_cb_id, user_cb_id):
    user_list = []
    seq_list = []
    label_list = []
    for tx_id, cb_id, user in zip(item_id, train_item_cb_id, user_cb_id):
        items = cb_id.split(' ')
        ids = tx_id.split(' ')
        temp_list = []
        for j, item in enumerate(items):
            temp_list.append(item)
            if j >= 2:
                temp = " ".join(temp_list)
                seq_list.append(temp)
                label_list.append(int(ids[j + 1]))
                user_list.append(user)
    label_list = torch.tensor(np.array(label_list))
    return seq_list, label_list, user_list


# [Restored Function]
def seq_construct_back(item_id, train_item_cb_id, user_cb_id):
    user_list = []
    seq_list = []
    label_list = []
    seq_num = 0
    for tx_id, cb_id, user in zip(item_id, train_item_cb_id, user_cb_id):
        if seq_num < 20:
            items = cb_id.split(' ')
            ids = tx_id.split(' ')
            for j in range(1, len(items)):
                if len(items) - j > 2:
                    temp_list = items[:-j]
                    temp = " ".join(temp_list)
                    seq_list.append(temp)
                    label_list.append(int(ids[-j - 1]))
                    user_list.append(user)
                    seq_num += 1
        else:
            break
    label_list = torch.tensor(np.array(label_list))
    return seq_list, label_list, user_list


# [Restored Function]
def seq_construct_v2(item_id, train_item_cb_id, user_cb_id):
    user_list = []
    seq_list = []
    label_list = []
    label_cb_list = []
    for tx_id, cb_id, user in zip(item_id, train_item_cb_id, user_cb_id):
        items = cb_id.split(' ')
        ids = tx_id.split(' ')
        temp_list = []
        for j in range(len(items) - 1):
            item = items[j]
            temp_list.append(item)
            if j > 2:
                temp = " ".join(temp_list)
                seq_list.append(temp)
                label_cb_list.append(items[j + 1])
                label_list.append(int(ids[j + 1]))
                user_list.append(user)
    label_list = torch.tensor(np.array(label_list))
    return seq_list, label_list, user_list, label_cb_list


def precompute_hard_negatives(item_emb, k=50, device='cpu'):
    print("Pre-computing Hard Negatives candidates...")
    n_items = item_emb.shape[0]
    item_emb = item_emb.to(device)
    norm_emb = torch.nn.functional.normalize(item_emb, p=2, dim=1)
    chunk_size = 1000
    hard_neg_indices = []
    with torch.no_grad():
        for i in range(0, n_items, chunk_size):
            end = min(i + chunk_size, n_items)
            chunk = norm_emb[i:end]
            sim = torch.matmul(chunk, norm_emb.T)
            _, top_indices = torch.topk(sim, k + 1, dim=1)
            chunk_hard = top_indices[:, 1:]
            hard_neg_indices.append(chunk_hard.cpu())
    all_hard_negs = torch.cat(hard_neg_indices, dim=0)
    print(f"Hard Negatives Map computed. Shape: {all_hard_negs.shape}")
    return all_hard_negs


def prompt(user_batch, hist_batch, recent_batch, is_test=False):
    prefix = "You are an intelligent recommender assistant. "
    sentences = []
    for user, hist, recent in zip(user_batch, hist_batch, recent_batch):
        hist_text = hist if len(hist) > 0 else "None"
        if is_test:
            p = (f"The user {user} has a long-term preference for {hist_text}. "
                 f"However, their recent interactions are {recent}. "
                 f"Based on this evolution, recommend the next item.")
        else:
            rand_opt = random.random()
            if rand_opt < 0.33:
                p = (f"Analyze {user}'s history: {hist_text} and recent behavior: {recent}. "
                     f"Step 1: Identify long-term taste. "
                     f"Step 2: Detect recent intent shift. "
                     f"Step 3: Predict the next item.")
            elif rand_opt < 0.66:
                p = (f"Although {user} historically liked {hist_text}, "
                     f"they recently switched focus to {recent}. "
                     f"What item best fits this new interest?")
            else:
                p = f"User: {user}. History: {hist_text}. Recent: {recent}. Prediction:"
        sentences.append(prefix + p)
    return sentences


def train_prompt(user, items):
    prompts = dict()
    prompts[0] = f'Given the following purchase history for the {user}: {items}. Predict the user preferences.'
    idx = int(np.random.randint(len(prompts), size=1))
    return prompts[0]


def prefix_prompt(RP=False, ICL=False, CoT=False):
    output = ''
    if RP:
        output += 'You are an expert at recommending products to users based on their purchase histories. '
    return output + '\n'


class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def read_cf_embeddings(model_name, checkpoint_name):
    path = '../src/' + model_name + '/' + checkpoint_name + '.pth.tar'
    print("Loading CF Embeddings from:", path)
    model = torch.load(path)
    user_emb = model['embedding_user.weight']
    item_emb = model['embedding_item.weight']
    return user_emb, item_emb


def get_target_emb(item_emb, labels):
    target = item_emb[labels]
    return target


def codebook_tokens(n_book, n_token):
    add_tokens = []
    for i in range(n_book):
        for j in range(n_token):
            token = "<" + str(i) + '-' + str(j) + '>'
            add_tokens.append(token)
    user_head = get_user_head()
    for i in range(n_book):
        for j in range(n_token):
            head_idx = i if i < len(user_head) else i % len(user_head)
            token = "<" + user_head[head_idx] + '-' + str(j) + '>'
            add_tokens.append(token)
    return add_tokens


def similarity_score(predicts, item_emb, item_id):
    predicts_norm = F.normalize(predicts, p=2, dim=1)
    item_emb_norm = F.normalize(item_emb, p=2, dim=1)
    score = torch.matmul(predicts_norm, item_emb_norm.T)
    batch = predicts.shape[0]
    for i in range(batch):
        if len(item_id[i].strip()) > 0:
            items = [int(item) for item in item_id[i].split(" ")]
            score[i, items] = -1e9
    return score


def MSE_distance(predicts, item_emb):
    p_sq = predicts.pow(2).sum(dim=1, keepdim=True)
    i_sq = item_emb.pow(2).sum(dim=1, keepdim=True).t()
    term2 = 2 * torch.matmul(predicts, item_emb.t())
    dist_sq = p_sq + i_sq - term2
    dist = dist_sq.clamp(min=1e-12).sqrt()
    return dist


# [Restored Function]
def whole_word_embedding(tokenizer, emb, input_ids, n_book):
    batch_whole_word_emb = []
    batch, source_l = input_ids.shape
    mark_id = tokenizer.convert_tokens_to_ids('_')
    for i in range(batch):
        sentence = input_ids[i, :]
        ids = torch.arange(len(sentence)) + 1
        ids[(sentence == 0).nonzero(as_tuple=True)] = 0
        marks = (sentence == mark_id).nonzero(as_tuple=False)
        curr = torch.max(ids)
        for j in range(marks.shape[0]):
            curr += 1
            temp = marks[j, :]
            idx = torch.arange(-1, n_book + 1, 1) + temp
            ids[idx] = curr
        curr_emb = torch.stack([emb.weight[ids[l]] for l in range(source_l)], dim=0)
        batch_whole_word_emb.append(curr_emb)
    batch_whole_word_emb = torch.stack(batch_whole_word_emb, dim=0)
    return batch_whole_word_emb


# [Restored Function]
def whole_word_embedding_v2(tokenizer, emb, input_ids, n_book):
    tic = time.time()
    batch, source_l = input_ids.shape
    mark_id = tokenizer.convert_tokens_to_ids('_')
    whole_word_ids = torch.arange(source_l) + 1
    whole_word_ids = whole_word_ids.unsqueeze(dim=0).expand(batch, -1)
    whole_word_ids[(input_ids == 0).nonzero(as_tuple=True)] = 0
    marks = (input_ids == mark_id).nonzero(as_tuple=True)
    rows, columns = marks
    for n in range(-1, n_book + 1, 1):
        whole_word_ids[rows, columns + n] = whole_word_ids[rows, columns]
    batch_whole_word_emb = []
    for b in range(batch):
        sentence_emb = []
        for l in range(source_l):
            sentence_emb.append(emb.weight[whole_word_ids[b, l]])
        sentence_emb = torch.stack(sentence_emb, dim=0)
        batch_whole_word_emb.append(sentence_emb)
    batch_whole_word_emb = torch.stack(batch_whole_word_emb, dim=0)
    toc = time.time()
    print(f'elapsed {(toc - tic):.2f}s')
    return batch_whole_word_emb


# [HELPER: 用于正确解析 item_ <t> <t> 格式]
def parse_item_codebook(item_cb_str):
    if not isinstance(item_cb_str, str): return []
    parts = item_cb_str.strip().split('item_')
    items = ["item_" + part.strip() for part in parts if part.strip()]
    return items


# [CORE FIX: 适配 CSV 读取 + 正确 Token 解析]
def data_augment(id_list, codebook_id_list, shred=2, item_limit=20):
    num = len(id_list)
    samples = []
    codebook_samples = []

    for n in range(num):
        ids = id_list[n].strip('\n').split(" ")
        user_id = ids[0]
        item_id = ids[1:]

        # [FIX] 通过 List 索引访问，非 Dict
        user_codebook_id = codebook_id_list[n][0]

        # [FIX] 使用 parse_item_codebook 正确切分物品
        item_cb_str = str(codebook_id_list[n][1]).strip('\n')
        item_codebook_list = parse_item_codebook(item_cb_str)

        loop_len = min(len(item_id), len(item_codebook_list))

        temp_sample = []
        temp_item_cb = []

        temp_sample.append(user_id)

        for k in range(loop_len):
            if k > item_limit:
                break
            temp_sample.append(item_id[k])
            temp_item_cb.append(item_codebook_list[k])

            if k > shred:
                sample = " ".join(temp_sample)
                item_seq_str = " ".join(temp_item_cb)
                samples.append(sample)
                codebook_samples.append([user_codebook_id, item_seq_str])

    return samples, codebook_samples


# [CORE FIX: 适配 CSV 读取 + 正确 Token 解析]
def data_construction(id_list, codebook_id_list, item_limit=100):
    num = len(id_list)
    samples = []
    codebook_samples = []

    for n in range(num):
        ids = id_list[n].strip('\n').split(" ")
        user_id = ids[0]
        item_id = ids[1:]

        # [FIX] 通过 List 索引访问
        user_codebook_id = codebook_id_list[n][0]
        item_cb_str = str(codebook_id_list[n][1]).strip('\n')
        item_codebook_list = parse_item_codebook(item_cb_str)

        loop_len = min(len(item_id), len(item_codebook_list))

        temp_sample = []
        temp_item_cb = []

        temp_sample.append(user_id)

        for k in range(loop_len):
            if k > item_limit:
                break
            temp_sample.append(item_id[k])
            temp_item_cb.append(item_codebook_list[k])

        sample = " ".join(temp_sample)
        item_seq_str = " ".join(temp_item_cb)

        samples.append(sample)
        codebook_samples.append([user_codebook_id, item_seq_str])

    return samples, codebook_samples
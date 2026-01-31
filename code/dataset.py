import os
from torch.utils.data import DataLoader, Dataset
import utils
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import random


def read_data(data_name):
    # 1. 读取 Train Codebook (CSV)
    file_path = '../data/' + data_name + '/train_codebook.txt'
    df_train = pd.read_csv(file_path)
    df_train.columns = [c.strip() for c in df_train.columns]
    # 转为 List 供 utils 处理
    train_codebook_data = df_train[['user_cb_id', 'item_cb_id']].values.tolist()

    # 2. 读取 Train Data
    file_path = '../data/' + data_name + '/train.txt'
    with open(file_path, 'r') as f:
        train_data = [line.strip() for line in f.readlines()]

    # 3. 读取 Test Codebook (CSV)
    file_path = '../data/' + data_name + '/test_codebook.txt'
    df_test = pd.read_csv(file_path)
    df_test.columns = [c.strip() for c in df_test.columns]
    test_codebook_data = df_test[['user_cb_id', 'item_cb_id']].values.tolist()

    # 4. 读取 Test Data
    file_path = '../data/' + data_name + '/test.txt'
    with open(file_path, 'r') as f:
        test_data = [line.strip() for line in f.readlines()]

    # 5. 读取 Item List (原生 open，防止逗号干扰)
    item_path = '../data/' + data_name + '/item_list.txt'
    with open(item_path, 'r', encoding='utf-8') as f:
        item_list = [line.strip() for line in f.readlines()]

    # 6. 读取 User List
    user_list = pd.read_csv('../data/' + data_name + '/user_list.txt', header=0)

    item_num = len(item_list)
    user_num = len(user_list)

    return train_data, test_data, train_codebook_data, test_codebook_data, item_num, user_num


def read_cold_or_warm_data(data_name, mode='cold'):
    file_path = '../data/' + data_name + '_' + mode + '/train_codebook.txt'
    df_train = pd.read_csv(file_path)
    df_train.columns = [c.strip() for c in df_train.columns]
    train_codebook_data = df_train[['user_cb_id', 'item_cb_id']].values.tolist()

    file_path = '../data/' + data_name + '_' + mode + '/train.txt'
    with open(file_path, 'r') as f:
        train_data = [line.strip() for line in f.readlines()]

    file_path = '../data/' + data_name + '_' + mode + '/test_codebook.txt'
    df_test = pd.read_csv(file_path)
    df_test.columns = [c.strip() for c in df_test.columns]
    test_codebook_data = df_test[['user_cb_id', 'item_cb_id']].values.tolist()

    file_path = '../data/' + data_name + '_' + mode + '/test.txt'
    with open(file_path, 'r') as f:
        test_data = [line.strip() for line in f.readlines()]

    return train_data, test_data, train_codebook_data, test_codebook_data


class VQDataset(Dataset):
    def __init__(self, embs):
        super().__init__()
        self.embs = embs

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, index: int):
        emb = self.embs[index]
        return emb


class DualVQDataset(Dataset):
    def __init__(self, collab_emb, semantic_emb):
        super().__init__()
        self.collab_emb = collab_emb
        self.semantic_emb = semantic_emb
        assert len(self.collab_emb) == len(self.semantic_emb), \
            f"Embeddings length mismatch: {len(collab_emb)} vs {len(semantic_emb)}"

    def __len__(self):
        return len(self.collab_emb)

    def __getitem__(self, index: int):
        return self.collab_emb[index], self.semantic_emb[index]


# [CRITICAL UPDATE] RecDataset for Codebook Generation
class RecDataset(Dataset):
    def __init__(self, data, user_emb, user_vq, item_emb, item_vq, item_sem_emb=None, max_item_num=1e10):
        super().__init__()
        self.data = data
        self.max_item_num = max_item_num
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.item_sem_emb = item_sem_emb
        self.user_vq = user_vq
        self.item_vq = item_vq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        sample = sample.split()
        user = sample[0]
        item_list = sample[1:]
        if len(item_list) > self.max_item_num:
            item_list = item_list[:self.max_item_num]

        user_id = int(user)
        item_idx = [int(x) for x in item_list]

        user_emb = self.user_emb[user_id].unsqueeze(0)
        item_emb_vec = self.item_emb[item_idx]

        # [CRITICAL FIX] Normalize input before encoding to match Training behavior!
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb_vec = F.normalize(item_emb_vec, p=2, dim=1)

        with torch.no_grad():
            if hasattr(self.user_vq, 'mq_semantic'):
                user_vq_id = self.user_vq.encode(user_emb)
            else:
                user_vq_id = self.user_vq.encode(user_emb)

            if hasattr(self.item_vq, 'mq_semantic'):
                item_vq_idx = self.item_vq.encode(item_emb_vec)
            else:
                item_vq_idx = self.item_vq.encode(item_emb_vec)

        user_cb_id = utils.user_codebook_to_str(user_vq_id)
        item_cb_id = utils.item_codebook_to_str(item_vq_idx)
        return user_id, user_cb_id, item_cb_id


# [HELPER] Correctly parse "item_ <t1> <t2>" structure
def parse_item_codebook_str(item_cb_seq_str):
    if pd.isna(item_cb_seq_str): return []
    raw_str = str(item_cb_seq_str).strip()
    segments = raw_str.split('item_')
    items = ["item_" + seg.strip() for seg in segments if seg.strip()]
    return items


class LLM4RecDataset(Dataset):
    def __init__(self, data, codebook_data, no_shuffle=False, recent_k=3):
        super().__init__()
        self.data = data
        self.codebook_data = codebook_data
        self.no_shuffle = no_shuffle
        self.recent_k = recent_k
        self.processed_data = []

        print("Pre-processing Validation/Test Dataset...")
        for idx in range(len(data)):
            sample = self.data[idx].split(" ")
            user_id = int(sample[0])
            item_list = sample[1:]

            if len(item_list) < 1:
                continue

            target_id = int(item_list[-1])
            item_id_str = " ".join(item_list[:-1])

            # Codebook Parsing (适配 utils 返回的 List 结构)
            if isinstance(self.codebook_data, list):
                if isinstance(self.codebook_data[idx], list):
                    # List of Lists: [user_str, item_seq_str]
                    user_cb_id = self.codebook_data[idx][0]
                    item_cb_seq_str = self.codebook_data[idx][1]
                else:
                    # List of strings (fallback)
                    parts = self.codebook_data[idx].split(" item_", 1)
                    user_cb_id = parts[0]
                    item_cb_seq_str = "item_" + parts[1] if len(parts) > 1 else ""
            else:
                # DataFrame
                user_cb_id = self.codebook_data['user_cb_id'][idx]
                item_cb_seq_str = self.codebook_data['item_cb_id'][idx]

            item_cb_id_list = parse_item_codebook_str(item_cb_seq_str)

            if len(item_cb_id_list) > 0:
                target_cb_id = item_cb_id_list[-1]
                items = item_cb_id_list[:-1]
            else:
                target_cb_id = ""
                items = []

            if len(items) > self.recent_k:
                hist_items = items[:-self.recent_k]
                recent_items = items[-self.recent_k:]
            else:
                hist_items = []
                recent_items = items

            hist_str = " ".join(hist_items)
            recent_str = " ".join(recent_items)

            self.processed_data.append(
                (user_id, item_id_str, target_id, user_cb_id, hist_str, recent_str, target_cb_id)
            )
        print("Dataset Pre-processing Done.")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, index: int):
        return self.processed_data[index]


class LLM4RecTrainDataset(Dataset):
    def __init__(self, data, codebook_data, no_shuffle=False, recent_k=3):
        super().__init__()
        self.data = data
        self.codebook_data = codebook_data
        self.no_shuffle = no_shuffle
        self.recent_k = recent_k
        self.processed_data = []

        print("Pre-processing Train Dataset...")
        for idx in range(len(data)):
            sample = self.data[idx].split(" ")
            user_id = int(sample[0])
            item_list = sample[1:]

            if len(item_list) < 2:
                continue

            train_target_id = int(item_list[-2])
            valid_target_id = int(item_list[-1])

            # Codebook Parsing
            if isinstance(self.codebook_data, list) and isinstance(self.codebook_data[idx], list):
                user_cb_id = self.codebook_data[idx][0]
                item_cb_seq_str = self.codebook_data[idx][1]
            elif isinstance(self.codebook_data, list) and isinstance(self.codebook_data[idx], str):
                parts = self.codebook_data[idx].split(" item_", 1)
                user_cb_id = parts[0]
                item_cb_seq_str = "item_" + parts[1] if len(parts) > 1 else ""
            else:
                user_cb_id = self.codebook_data['user_cb_id'][idx]
                item_cb_seq_str = self.codebook_data['item_cb_id'][idx]

            item_cb_id_list = parse_item_codebook_str(item_cb_seq_str)

            if len(item_cb_id_list) >= 2:
                train_target_cb_id = item_cb_id_list[-2]
                valid_target_cb_id = item_cb_id_list[-1]
                train_items = item_cb_id_list[:-2]
            else:
                continue

            self.processed_data.append(
                (user_id, train_target_id, valid_target_id, user_cb_id, train_items, train_target_cb_id,
                 valid_target_cb_id)
            )
        print("Train Dataset Pre-processing Done.")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, index: int):
        user_id, train_target_id, valid_target_id, user_cb_id, train_items, train_target_cb_id, valid_target_cb_id = \
            self.processed_data[index]

        current_train_items = list(train_items)

        if self.no_shuffle is False:
            random.shuffle(current_train_items)

        if len(current_train_items) > self.recent_k:
            hist_items = current_train_items[:-self.recent_k]
            recent_items = current_train_items[-self.recent_k:]
        else:
            hist_items = []
            recent_items = current_train_items

        hist_str = " ".join(hist_items)
        recent_str = " ".join(recent_items)

        return user_id, train_target_id, valid_target_id, user_cb_id, hist_str, recent_str, train_target_cb_id, valid_target_cb_id
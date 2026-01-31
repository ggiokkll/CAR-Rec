import torch
import torch.nn as nn
import utils
import copy
import sys


# K-way + masking
# [FILE: model.py]
# 请完全替换原有的 MQ 类

class MQ(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book, mask_ratio=0.2):
        super(MQ, self).__init__()
        self.m_book = m_book
        self.encoders = nn.ModuleList()
        self.codebooks = nn.ModuleList()

        # [创新修复 1] 使用每路独立的 Query Token 替代全局 Positional Embedding
        # 作用：明确告知每个 Encoder 它正在处理哪一部分特征，防止特征混淆
        self.query_tokens = nn.Parameter(torch.randn(m_book, input_dim))

        for m in range(m_book):
            codebook = nn.Embedding(n_embedding, dim)
            codebook.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)

            # [修复 2] 降低 Dropout (0.5 -> 0.1) 防止在分词过程中丢失关键信息
            encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(256, dim),
            )
            self.codebooks.append(codebook)
            self.encoders.append(encoder)

        self.mask_ratio = mask_ratio

        # Decoder 同样降低 Dropout
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def _process_input(self, x, m, apply_random_mask=False):
        """
        统一输入处理：Masking + Query Injection
        """
        b, e = x.shape
        patch = int(e / self.m_book)

        # 1. 掩码逻辑 (Masked K-way)
        mask = torch.ones((e,), device=x.device).bool()
        mask[m * patch: (m + 1) * patch] = 0

        # 将 Mask 区域置为 0
        x_m = torch.masked_fill(x, mask, 0)

        # [关键修复] 注入该路特有的 Query Token
        # 替代原有的 x_m += self.pos，使用 broadcasting 加法
        x_m = x_m + self.query_tokens[m]

        return x_m

    def forward(self, x):
        res_list = []
        ce_list = []

        for m in range(self.m_book):
            # 处理输入
            x_m = self._process_input(x, m, apply_random_mask=True)

            # 编码
            ze = self.encoders[m](x_m)

            # 量化距离计算
            embedding = self.codebooks[m].weight
            N, C = ze.shape
            K, _ = embedding.shape
            ze_broadcast = ze.reshape(N, 1, C)
            emb_broadcast = embedding.reshape(1, K, C)

            distance = torch.sum((emb_broadcast - ze_broadcast) ** 2, 2)
            nearest_neighbor = torch.argmin(distance, 1)

            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)

            # 直通估计 (STE)
            ce = ze + (ce - ze).detach()

        # 聚合 + 解码
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        x_hat = self.decoder(zq)

        return x_hat, res_list, ce_list


    def valid(self, x):
        # [Fix] 推理时也必须归一化，与训练保持一致
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            x_m = self._process_input(x, m, apply_random_mask=False)
            ze = self.encoders[m](x_m)
            embedding = self.codebooks[m].weight.data
            N, C = ze.shape
            K, _ = embedding.shape
            ze_broadcast = ze.reshape(N, 1, C)
            emb_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((emb_broadcast - ze_broadcast) ** 2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        x_hat = self.decoder(zq)
        return x_hat, res_list, ce_list
    def encode(self, x):
        # [Fix] 推理时也必须归一化，与训练保持一致
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        nearest_neighbor_list = []
        for m in range(self.m_book):
            x_m = self._process_input(x, m, apply_random_mask=False)
            ze = self.encoders[m](x_m)
            embedding = self.codebooks[m].weight.data
            N, C = ze.shape
            K, _ = embedding.shape
            ze_broadcast = ze.reshape(N, 1, C)
            emb_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((emb_broadcast - ze_broadcast) ** 2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            nearest_neighbor_list.append(nearest_neighbor)
        codeword_idx = torch.stack(nearest_neighbor_list, dim=0).transpose(0, 1)
        return codeword_idx


# [INNOVATION: Dual-Alignment Tokenizer]
class DualMQ(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book, mask_ratio=0.2, semantic_dim=768):
        super(DualMQ, self).__init__()
        # View A: Collaborative (e.g., LightGCN, 64-dim)
        self.mq_collab = MQ(input_dim, dim, n_embedding, m_book, mask_ratio)

        # View B: Semantic (e.g., T5, 768-dim)
        # [Innovation] Projection Layer: 768 -> 64
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, input_dim),
            nn.Tanh(),
            nn.LayerNorm(input_dim)
        )
        self.mq_semantic = MQ(input_dim, dim, n_embedding, m_book, mask_ratio)
        self.m_book = m_book

    def forward(self, x_collab, x_semantic):
        x_hat_c, res_c, ce_c = self.mq_collab(x_collab)

        x_sem_proj = self.semantic_proj(x_semantic)
        x_hat_s, res_s, ce_s = self.mq_semantic(x_sem_proj)

        return (x_hat_c, x_hat_s), (res_c, res_s), (ce_c, ce_s)

    def valid(self, x_collab, x_semantic=None):
        x_hat_c, res_c, ce_c = self.mq_collab.valid(x_collab)
        if x_semantic is not None:
            x_sem_proj = self.semantic_proj(x_semantic)
            x_hat_s, res_s, ce_s = self.mq_semantic.valid(x_sem_proj)
        else:
            x_hat_s, res_s, ce_s = None, None, None

        return (x_hat_c, x_hat_s), (res_c, res_s), (ce_c, ce_s)

    def encode(self, x_collab, x_semantic=None):
        idx_c = self.mq_collab.encode(x_collab)
        return idx_c


class ResidualVQVAE(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book):
        super(ResidualVQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, dim),
        )

        self.m_book = m_book
        self.codebooks = nn.ModuleList()
        for m in range(m_book):
            codebook = nn.Embedding(n_embedding, dim)
            codebook.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
            self.codebooks.append(codebook)

        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebooks[0].weight
                N, C = ze.shape
                K, _ = embedding.shape
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
            else:
                res = res_list[m - 1] - ce_list[m - 1]
                embedding = self.codebooks[m].weight
                N, C = res.shape
                K, _ = embedding.shape
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast) ** 2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = ze + (zq - ze).detach()

        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list

    def valid(self, x):
        return self.forward(x)

    def encode(self, x):
        nearest_neighbor_list = []
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebooks[0].weight
                N, C = ze.shape
                K, _ = embedding.shape
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
                nearest_neighbor_list.append(nearest_neighbor)
            else:
                res = res_list[m - 1] - ce_list[m - 1]
                embedding = self.codebooks[m].weight
                N, C = res.shape
                K, _ = embedding.shape
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast) ** 2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
                nearest_neighbor_list.append(nearest_neighbor)
        codeword_idx = torch.stack(nearest_neighbor_list, dim=0).transpose(0, 1)
        return codeword_idx


class projection(nn.Module):
    def __init__(self, input_dim, output_dim, target_length, hidden_dim=256):
        super(projection, self).__init__()
        self.l1 = nn.Linear(int(target_length * input_dim), hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.dropout(self.l1(x)))
        x = self.l2(x)
        return x
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5EncoderModel, T5ForConditionalGeneration, T5Model, \
    AutoTokenizer
import pandas as pd
import model
import sys
from tqdm import tqdm
import dataset
import myevaluate
import utils


# ==========================================
# Helper: Simple K-Means for Initialization
# ==========================================
def run_kmeans(x, n_clusters, n_iter=10):
    """
    Simple K-Means implementation using PyTorch to initialize codebooks.
    x: [N, Dim] data
    """
    N, D = x.shape
    # Randomly select initial centroids from data
    indices = torch.randperm(N)[:n_clusters]
    centroids = x[indices]  # [K, D]

    with torch.no_grad():
        for _ in range(n_iter):
            # Compute distances: (x - c)^2 = x^2 + c^2 - 2xc
            # Using simple cdist
            dists = torch.cdist(x, centroids)  # [N, K]

            # Assign to nearest centroid
            labels = torch.argmin(dists, dim=1)

            # Update centroids
            new_centroids = []
            for k in range(n_clusters):
                mask = (labels == k)
                if mask.sum() > 0:
                    new_centroids.append(x[mask].mean(0))
                else:
                    # Re-initialize empty cluster with random data point
                    random_idx = torch.randint(0, N, (1,)).item()
                    new_centroids.append(x[random_idx])
            centroids = torch.stack(new_centroids)

    return centroids


# [FILE: train.py]

# 1. 新增对比损失函数 (请放在 vqvae_cold 之前)
def contrastive_loss(zq, temperature=0.1):
    """
    [COST Paper Idea] InfoNCE Contrastive Loss
    强迫 Batch 内不同物品的量化特征 zq 尽可能不同，解决 ID 塌陷问题。
    """
    # zq: [Batch, Dim]
    zq = F.normalize(zq, p=2, dim=1)

    # 计算相似度矩阵
    logits = torch.matmul(zq, zq.T) / temperature

    # 目标：对角线为正例 (自己与自己相似)，其余为负例
    labels = torch.arange(zq.size(0)).to(zq.device)

    return F.cross_entropy(logits, labels)


# 2. 替换初始化函数 (适配 removed pos.weight)
def initialize_mq_codebooks(mq_module, data_loader, device, n_token, m_book):
    print(f"   > Initializing Codebooks with K-Means (Clusters={n_token})...")
    mq_module.to(device)
    mq_module.eval()

    all_zes = [[] for _ in range(m_book)]
    limit_batches = 50

    with torch.no_grad():
        for i, x in enumerate(data_loader):
            if i >= limit_batches: break
            if isinstance(x, list): x = x[0]

            x = x.to(device)
            x = F.normalize(x, p=2, dim=1)
            b, e = x.shape
            patch = int(e / m_book)

            for m in range(m_book):
                # 重新实现 _process_input 的逻辑
                mask = torch.ones((e,), device=device).bool()
                mask[m * patch: (m + 1) * patch] = 0
                x_m = torch.masked_fill(x, mask, 0)

                # [Fix] 使用 query_tokens
                x_m = x_m + mq_module.query_tokens[m]

                ze = mq_module.encoders[m](x_m)
                all_zes[m].append(ze.cpu())

    for m in range(m_book):
        if len(all_zes[m]) > 0:
            features = torch.cat(all_zes[m], dim=0)
            # 调用原本的 run_kmeans
            centroids = run_kmeans(features, n_token)
            mq_module.codebooks[m].weight.data.copy_(centroids.to(device))
            print(f"     [Book {m}] Initialized.")
        else:
            print(f"     [Book {m}] Warning: No data for initialization.")


# 3. 替换 vqvae_cold 函数 (加入 contrastive loss)
def vqvae_cold(model, model_name, device, co_emb, n_embedding, kmean_epoch=50, m_book=3, batch_size=512, lr=1e-3,
               n_epochs=1000, l_w_embedding=1, l_w_commitment=0.25, args=None):
    train_dataset = dataset.VQDataset(co_emb)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialization
    if hasattr(model, 'mq_collab'):
        initialize_mq_codebooks(model.mq_collab, train_dataloader, device, n_embedding, m_book)
    else:
        initialize_mq_codebooks(model, train_dataloader, device, n_embedding, m_book)

    is_dual = hasattr(model, 'mq_semantic')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 稍微降低 weight decay
    mse_loss = nn.MSELoss()
    w_commit = l_w_commitment if l_w_commitment else 0.25

    # [新增] 对比损失权重
    w_contrastive = 0.1

    print(f"Start Training VQ (Contrastive Enhanced)...")

    for e in range(n_epochs):
        total_loss = 0
        model.train()
        for i, x in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            x = F.normalize(x, p=2, dim=1)

            if is_dual:
                x_hat, res_dict, ce_dict = model.mq_collab(x)
            else:
                x_hat, res_dict, ce_dict = model(x)

            # 1. 重建损失
            l_recon = mse_loss(x_hat, x)

            # 2. VQ 损失
            loss_vq = 0
            zq_list = []
            for m in range(m_book):
                loss_vq += l_w_embedding * mse_loss(res_dict[m].detach(), ce_dict[m])
                loss_vq += w_commit * mse_loss(res_dict[m], ce_dict[m].detach())
                zq_list.append(ce_dict[m])

            # 3. [新增] 跨路聚合对比损失
            # 将 K 个 Codebook 的量化向量相加，作为物品的最终离散表示
            zq_sum = torch.sum(torch.stack(zq_list, dim=0), dim=0)
            l_con = contrastive_loss(zq_sum, temperature=0.1)

            # 总损失
            loss = l_recon + loss_vq + w_contrastive * l_con

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if e % 10 == 0:  # 打印频率改高一点
            print(f'Cold VQ epoch {e} train_loss: {total_loss / len(train_dataloader):.5f} (ConLoss included)')

    torch.save(model.state_dict(), '../checkpoints/vq_cold/' + model_name + '.pth')


# [INNOVATION Strategy 1: Dual-View Semantic Alignment]
# [FILE: train.py]
# 请替换原有的 vqvae 函数

def vqvae(model, model_name, device, co_emb, semantic_emb=None, n_embedding=None, kmean_epoch=50, m_book=3,
          valid_ratio=0.2, batch_size=512,
          lr=1e-3, n_epochs=1000, l_w_embedding=1, l_w_commitment=0.25, args=None):
    # 1. 检查是否开启双视图模式
    is_dual = hasattr(model, 'mq_semantic') and (semantic_emb is not None)

    # 2. 划分训练/验证集
    idx = torch.randperm(co_emb.size(0))
    sh = int(co_emb.size(0) * (1 - valid_ratio))
    train_idx = idx[:sh]
    # valid_idx = idx[sh:] # 暂时不用验证集做 Early Stopping，简化流程

    train_co_emb = co_emb[train_idx]

    if is_dual:
        train_sem_emb = semantic_emb[train_idx]
        train_dataset = dataset.DualVQDataset(train_co_emb, train_sem_emb)
        print(f"[Train] Dual-View Mode Active. Collab: {train_co_emb.shape}, Semantic: {train_sem_emb.shape}")
    else:
        train_dataset = dataset.VQDataset(train_co_emb)
        print(f"[Train] Single-View Mode Active.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # 3. 初始化 (重要：防止初始塌陷)
    if hasattr(model, 'mq_collab'):
        # Dual Mode Initialization
        print("Initializing Collab View...")
        initialize_mq_codebooks(model.mq_collab, train_dataloader, device, n_embedding, m_book)

        print("Initializing Semantic View...")
        # 语义投影层需要先处理数据
        model.semantic_proj.to(device)
        with torch.no_grad():
            raw_sem_gpu = F.normalize(train_sem_emb.to(device), p=2, dim=1)
            projected_sem = model.semantic_proj(raw_sem_gpu)

        sem_loader = DataLoader(dataset.VQDataset(projected_sem.cpu()), batch_size=batch_size)
        initialize_mq_codebooks(model.mq_semantic, sem_loader, device, n_embedding, m_book)

    elif hasattr(model, 'encoders'):
        # Single Mode Initialization
        print("Initializing Single View...")
        initialize_mq_codebooks(model, train_dataloader, device, n_embedding, m_book)

    # 4. 优化器配置
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse_loss = nn.MSELoss()

    # 超参数
    con_alpha = args.con_alpha if args else 0.1  # 一致性损失权重
    alpha_align = args.alpha_align if args else 0.1  # 对齐损失权重
    w_commit = l_w_commitment if l_w_commitment else 0.25

    # [新增] 对比损失权重 (与 vqvae_cold 保持一致)
    w_contrastive = 0.1

    print(
        f"Start VQ Training (Contrastive). Dual: {is_dual}, Align: {alpha_align}, Commit: {w_commit}, Contrast: {w_contrastive}")

    for e in range(n_epochs):
        total_loss = 0
        model.train()

        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            if is_dual:
                # ==========================
                # A. 双视图训练逻辑
                # ==========================
                x_collab, x_semantic_real = data
                x_collab = x_collab.to(device)
                x_semantic_real = x_semantic_real.to(device)

                # Normalize
                x_collab = F.normalize(x_collab, p=2, dim=1)
                x_semantic_real = F.normalize(x_semantic_real, p=2, dim=1)

                # Forward
                (x_hat_c, x_hat_s), (res_c, res_s), (ce_c, ce_s) = model(x_collab, x_semantic_real)

                # Loss 1: Reconstruction (Collab)
                l_recon_c = mse_loss(x_hat_c, x_collab)

                # Loss 2: Alignment (Semantic <-> Collab)
                l_align = mse_loss(x_hat_s, x_hat_c)

                # Loss 3: VQ Loss & Consistency & [New] Contrastive
                loss_vq = 0
                l_consistency = 0

                # 收集用于对比的向量
                zq_c_list = []
                zq_s_list = []

                for m in range(m_book):
                    # VQ Dictionary + Commitment
                    loss_vq += l_w_embedding * mse_loss(res_c[m].detach(), ce_c[m])
                    loss_vq += w_commit * mse_loss(res_c[m], ce_c[m].detach())

                    loss_vq += l_w_embedding * mse_loss(res_s[m].detach(), ce_s[m])
                    loss_vq += w_commit * mse_loss(res_s[m], ce_s[m].detach())

                    # Consistency (Collab Code 应该和 Semantic Code 接近)
                    l_consistency += mse_loss(res_s[m], res_c[m])

                    zq_c_list.append(ce_c[m])
                    zq_s_list.append(ce_s[m])

                # [新增] 对比损失 (关键修改)
                # 确保 Collab 分支生成的 ID 具有区分度
                zq_c_sum = torch.sum(torch.stack(zq_c_list, dim=0), dim=0)
                l_con_c = contrastive_loss(zq_c_sum, temperature=0.1)

                # 可选：确保 Semantic 分支也具有区分度
                zq_s_sum = torch.sum(torch.stack(zq_s_list, dim=0), dim=0)
                l_con_s = contrastive_loss(zq_s_sum, temperature=0.1)

                # 总 Loss
                loss = l_recon_c + alpha_align * l_align + loss_vq + con_alpha * l_consistency + \
                       w_contrastive * (l_con_c + l_con_s)

            else:
                # ==========================
                # B. 单视图训练逻辑
                # ==========================
                x = data.to(device)
                x = F.normalize(x, p=2, dim=1)

                x_hat, res_dict, ce_dict = model(x)

                l_recon = mse_loss(x_hat, x)
                loss_vq = 0
                zq_list = []

                for m in range(m_book):
                    loss_vq += l_w_embedding * mse_loss(res_dict[m].detach(), ce_dict[m])
                    loss_vq += w_commit * mse_loss(res_dict[m], ce_dict[m].detach())
                    zq_list.append(ce_dict[m])

                # [新增] 对比损失
                zq_sum = torch.sum(torch.stack(zq_list, dim=0), dim=0)
                l_con = contrastive_loss(zq_sum, temperature=0.1)

                loss = l_recon + loss_vq + w_contrastive * l_con

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        if e % 10 == 0:
            print(f'VQ epoch {e} train_loss: {avg_loss:.5f}')

    # 保存模型
    torch.save(model.state_dict(), '../checkpoints/vq/' + model_name + '.pth')

# Backbone (LLM Training) - Unchanged
def backbone(data_name, train_rec_loader, valid_rec_loader, user_emb, item_emb, item_num, args, device):
    # 1. Hard Negative Setup
    hard_neg_map = None
    if args.enable_hard_neg:
        hard_neg_map = utils.precompute_hard_negatives(item_emb, k=args.hard_neg_k, device=device)

    if args.train_from_checkpoint is True:
        linear_projection = model.projection(input_dim=512, output_dim=item_emb.shape[1],
                                             target_length=args.target_length)
        t5 = T5Model.from_pretrained('../checkpoints/backbone/' + data_name)
        tokenizer = AutoTokenizer.from_pretrained("../checkpoints/backbone/" + data_name, legacy=False)
        linear_projection.load_state_dict(torch.load('../checkpoints/backbone/' + data_name + '/projection.pt'))
        grouped_params = utils.group_model_params(t5, linear_projection, decay=args.decay)
    else:
        linear_projection = model.projection(input_dim=512, output_dim=item_emb.shape[1],
                                             target_length=args.target_length)
        # Use local t5-small to avoid connection issues
        tokenizer = AutoTokenizer.from_pretrained("../src/t5-small", legacy=False, local_files_only=True)
        t5 = T5Model.from_pretrained("../src/t5-small", local_files_only=True)

        real_n_book = args.n_book * 2 if args.enable_dual else args.n_book
        add_tokens = utils.codebook_tokens(real_n_book, args.n_token)

        num_added_toks = tokenizer.add_tokens(add_tokens)
        t5.resize_token_embeddings(len(tokenizer))
        print('added token number =', num_added_toks)
        grouped_params = utils.group_model_params(t5, linear_projection, decay=args.decay)

    optimizer = torch.optim.AdamW(grouped_params, lr=args.lr)
    loss_func = torch.nn.CrossEntropyLoss()

    t5.to(device)
    linear_projection.to(device)

    max_epoch = args.epochs
    global_metric = 0
    loss_list = []
    metric_list = []
    temperature = args.temperature

    for epoch in range(max_epoch):
        t5.train()
        linear_projection.train()
        loss_record = 0
        batch_record = 0
        for i, sample in enumerate(train_rec_loader):
            user_id, train_target_id, valid_target_id, user_cb_id, hist_str, recent_str, train_target_cb_id, valid_target_cb_id = sample
            train_batch = len(user_id)

            input_sentences = utils.prompt(user_cb_id, hist_str, recent_str, is_test=False)

            if len(input_sentences) == 0: continue

            batch_target_emb = utils.get_target_emb(item_emb, train_target_id).to(device)

            input_encoding = tokenizer(input_sentences, return_tensors='pt', max_length=args.source_length,
                                       padding="max_length", truncation=True)
            input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
            decoder_input_encoding = tokenizer([args.decoder_prepend for _ in range(len(train_target_cb_id))],
                                               return_tensors="pt", max_length=args.target_length, padding="max_length",
                                               truncation=True)
            decoder_input_ids, decoder_attention_mask = decoder_input_encoding.input_ids, decoder_input_encoding.attention_mask
            decoder_input_ids = t5._shift_right(decoder_input_ids)

            outputs = t5(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                         decoder_input_ids=decoder_input_ids.to(device))
            predicts = linear_projection(outputs.last_hidden_state)

            predicts_norm = F.normalize(predicts, p=2, dim=1)
            batch_target_norm = F.normalize(batch_target_emb, p=2, dim=1)
            logits_batch = torch.matmul(predicts_norm, batch_target_norm.T)

            logits = logits_batch
            if args.enable_hard_neg and hard_neg_map is not None:
                batch_candidates = hard_neg_map[train_target_id.cpu()].to(device)
                random_indices = torch.randint(0, args.hard_neg_k, (train_batch, 1)).to(device)
                selected_hard_indices = torch.gather(batch_candidates, 1, random_indices).squeeze(1)
                hard_neg_emb = item_emb[selected_hard_indices].to(device)
                hard_neg_norm = F.normalize(hard_neg_emb, p=2, dim=1)
                logits_hard = (predicts_norm * hard_neg_norm).sum(dim=1, keepdim=True)
                logits = torch.cat([logits_batch, logits_hard], dim=1)

            logits = torch.clamp(logits, min=-100, max=100)
            logits = logits / temperature
            labels = torch.arange(logits.shape[0]).to(device)
            loss = loss_func(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_record += loss.item() * train_batch
            batch_record += train_batch
            if i % 100 == 0:
                print('epoch =', epoch, 'batch =', i, 'train_loss =', loss.item())

        loss_list.append(loss_record / batch_record)

        # Validation every 1 epoch
        if epoch % 1 == 0:
            metrics = torch.zeros([2]).to(device)
            n_batch = 0
            t5.eval()
            linear_projection.eval()
            for i, sample in enumerate(tqdm(valid_rec_loader)):
                user_id, item_id, target_id, user_cb_id, hist_str, recent_str, target_cb_id = sample

                input_sentences = utils.prompt(user_cb_id, hist_str, recent_str, is_test=True)

                input_encoding = tokenizer(input_sentences, return_tensors='pt', max_length=args.source_length,
                                           padding="max_length", truncation=True)
                input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
                decoder_input_encoding = tokenizer([args.decoder_prepend for _ in range(len(target_cb_id))],
                                                   return_tensors="pt", max_length=args.target_length,
                                                   padding="max_length", truncation=True)
                decoder_input_ids, decoder_attention_mask = decoder_input_encoding.input_ids, decoder_input_encoding.attention_mask
                decoder_input_ids = t5._shift_right(decoder_input_ids)

                with torch.no_grad():
                    outputs = t5(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                                 decoder_input_ids=decoder_input_ids.to(device))
                    predicts = linear_projection(outputs.last_hidden_state)

                if args.similarity == 'cos':
                    scores = utils.similarity_score(predicts, item_emb, item_id)
                    results = torch.argsort(scores, dim=1, descending=True)
                elif args.similarity == 'MSE':
                    scores = utils.MSE_distance(predicts, item_emb)
                    results = torch.argsort(scores, dim=1, descending=False)
                else:
                    NotImplementedError

                metr, batch = myevaluate.get_metrics(target_id, results, device, args.k)
                metrics += metr
                n_batch += batch

            print(data_name, 'valid_hit@%s =' % args.k, metrics[0].item() / n_batch, 'valid_ndcg@%s =' % args.k,
                  metrics[1].item() / n_batch)
            metric_list.append(metrics / n_batch)

            if torch.mean(metrics / n_batch) > global_metric:
                global_metric = torch.mean(metrics / n_batch)
                t5.save_pretrained('../checkpoints/backbone/' + data_name)
                tokenizer.save_pretrained("../checkpoints/backbone/" + data_name)
                torch.save(linear_projection.state_dict(), '../checkpoints/backbone/' + data_name + '/projection.pt')

            metric_output = torch.stack(metric_list, dim=0)
            metric_save = pd.DataFrame(metric_output.detach().cpu().numpy(),
                                       columns=['hit@%s' % args.k, 'ncdg@%s' % args.k])
            metric_save.to_csv('../results/' + data_name + '/valid_metric_record.csv', index=False)
            loss_output = pd.DataFrame(loss_list, columns=['loss'])
            loss_output.to_csv('../results/' + data_name + '/train_loss_record.csv', index=False)
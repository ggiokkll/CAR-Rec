import torch
import pandas as pd
import numpy as np
import csv
import torch.nn as nn
import dataset
import utils
import sys
import model
import train
from torch.utils.data import DataLoader, Dataset
import os


def learning(args):
    # hype-params
    lgn_dim = 64
    codebook_dim = 512
    device = torch.device("cuda:0" if True and torch.cuda.is_available() else "cpu")
    data_name = args.dataset
    lgn_name = 'lgn-' + data_name + '-' + str(lgn_dim)
    vq_name = 'MQ-' + lgn_name
    print('Process: VQ is working:', vq_name)

    # 1. read lgn-embeddings (Collaborative)
    LightGCN = torch.load('../src/lgn/' + lgn_name + '.pth.tar')
    user_emb = LightGCN['embedding_user.weight']
    item_emb = LightGCN['embedding_item.weight']
    print('total number of items:', item_emb.shape[0])
    print('total number of users:', user_emb.shape[0])

    # 2. [INNOVATION] Load Semantic Embeddings
    item_sem_emb = None
    if args.enable_dual:
        item_sem_emb = utils.read_semantic_embeddings(data_name)
        if item_sem_emb is None:
            print("❌ Error: Enable Dual but no semantic file found! Please run pre-processing.")
            sys.exit()

    # 3. codebook initial
    if args.enable_dual and args.vq_model == 'MQ':
        print(f"Initializing DualMQ with Semantic Dim {args.semantic_dim}...")
        user_vq = model.MQ(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = model.DualMQ(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token,
                               m_book=args.n_book, semantic_dim=args.semantic_dim)
    elif args.vq_model == 'MQ':
        user_vq = model.MQ(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = model.MQ(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
    elif args.vq_model == 'RQ':
        user_vq = model.ResidualVQVAE(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token,
                                      m_book=args.n_book)
        item_vq = model.ResidualVQVAE(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token,
                                      m_book=args.n_book)
    else:
        NotImplementedError

    # ---------------------- training -------------------------------
    # item vq
    item_vq_name = 'item-' + vq_name
    if args.train_vq is True:
        train.vqvae(item_vq, item_vq_name, device, item_emb, semantic_emb=item_sem_emb, n_embedding=args.n_token,
                    m_book=args.n_book, args=args)
    item_vq.load_state_dict(torch.load('../checkpoints/vq/' + item_vq_name + '.pth'))
    item_vq.to(device)

    # user vq
    user_vq_name = 'user-' + vq_name
    if args.train_vq is True:
        train.vqvae(user_vq, user_vq_name, device, user_emb, n_embedding=args.n_token, m_book=args.n_book, args=args)
    user_vq.load_state_dict(torch.load('../checkpoints/vq/' + user_vq_name + '.pth'))
    user_vq.to(device)

    # -------------------- output -------------------------
    def generate_codebook(input_file, output_file):
        if not os.path.exists(input_file): return
        with open(input_file, 'r') as f:
            data_lines = f.readlines()
        item_vq.eval()
        user_vq.eval()
        dataset_obj = dataset.RecDataset(data_lines, user_emb, user_vq, item_emb, item_vq)
        loader = DataLoader(dataset_obj, batch_size=256, shuffle=False)
        item_list = []
        user_list = []
        for i, sample in enumerate(loader):
            user_id, user_cb_id, item_cb_id = sample
            user_list += user_cb_id
            item_list += item_cb_id
        data_df = {'user_cb_id': user_list, "item_cb_id": item_list}
        df = pd.DataFrame(data_df)
        df.to_csv(output_file)
        print(f"Generated {output_file}")

    generate_codebook('../data/' + data_name + '/train.txt', '../data/' + data_name + '/train_codebook.txt')
    generate_codebook('../data/' + data_name + '/test.txt', '../data/' + data_name + '/test_codebook.txt')


# [RESTORED] Cold Start Learning
def learning_cold(args):
    data_name = args.dataset

    # hype-params
    lgn_dim = 64
    codebook_dim = 512
    device = torch.device("cuda:" + str(args.cuda) if True and torch.cuda.is_available() else "cpu")
    lgn_name = 'lgn-' + data_name + '-' + str(lgn_dim)
    vq_name = 'MQ-' + lgn_name
    print('Process: Cold VQ is working:', vq_name)

    LightGCN = torch.load('../src/lgn/' + lgn_name + '.pth.tar')
    user_emb = LightGCN['embedding_user.weight']
    item_emb = LightGCN['embedding_item.weight']

    # codebook initial
    if args.enable_dual and args.vq_model == 'MQ':
        # Even for cold start, we use DualMQ structure to load weights correctly later
        user_vq = model.MQ(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = model.DualMQ(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token,
                               m_book=args.n_book, semantic_dim=args.semantic_dim)
    else:
        user_vq = model.MQ(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = model.MQ(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)

    # ---------------------- training -------------------------------
    # item vq
    item_vq_name = 'item-' + vq_name
    if args.train_vq is True:
        train.vqvae_cold(item_vq, item_vq_name, device, item_emb, n_embedding=args.n_token, m_book=args.n_book,
                         args=args)
    item_vq.load_state_dict(torch.load('../checkpoints/vq_cold/' + item_vq_name + '.pth'))
    item_vq.to(device)

    # user vq
    user_vq_name = 'user-' + vq_name
    if args.train_vq is True:
        train.vqvae_cold(user_vq, user_vq_name, device, user_emb, n_embedding=args.n_token, m_book=args.n_book,
                         args=args)
    user_vq.load_state_dict(torch.load('../checkpoints/vq_cold/' + user_vq_name + '.pth'))
    user_vq.to(device)

    # -------------------- output -------------------------
    def generate_codebook(input_file, output_file):
        if not os.path.exists(input_file): return
        with open(input_file, 'r') as f:
            data_lines = f.readlines()
        item_vq.eval()
        user_vq.eval()
        dataset_obj = dataset.RecDataset(data_lines, user_emb, user_vq, item_emb, item_vq)
        loader = DataLoader(dataset_obj, batch_size=256, shuffle=False)
        item_list = []
        user_list = []
        for i, sample in enumerate(loader):
            user_id, user_cb_id, item_cb_id = sample
            user_list += user_cb_id
            item_list += item_cb_id
        data_df = {'user_cb_id': user_list, "item_cb_id": item_list}
        df = pd.DataFrame(data_df)
        df.to_csv(output_file)
        print(f"Generated {output_file}")

    # Generate for both warm and cold
    for mode in ['_warm', '_cold']:
        generate_codebook('../data/' + data_name + mode + '/train.txt',
                          '../data/' + data_name + mode + '/train_codebook.txt')
        generate_codebook('../data/' + data_name + mode + '/test.txt',
                          '../data/' + data_name + mode + '/test_codebook.txt')
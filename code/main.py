import torch
import numpy as np
import pandas as pd
import sys
import train
from tqdm import tqdm
import dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import utils
import test
from parse import parse_args
import vq
import os

# hype-params
torch.autograd.set_detect_anomaly(True)
args = parse_args()

# pipeline: phase 1: vq, phase 2: llm
data_name = args.dataset
print('CAR-Rec is working on', data_name)

use_cuda = True
device = torch.device("cuda:" + str(args.cuda) if use_cuda and torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# [Safeguard] Create necessary directories
os.makedirs('../checkpoints/vq', exist_ok=True)
os.makedirs('../checkpoints/vq_cold', exist_ok=True)
os.makedirs('../checkpoints/backbone', exist_ok=True)
os.makedirs(f'../results/{data_name}', exist_ok=True)

# 1. VQ Training / Generation
if args.vq is True:
    vq.learning(args)

# --------------------- read data -----------------------
lgn_dim = 64
model_name = 'lgn'
checkpoint_name = model_name + '-' + data_name + '-' + str(lgn_dim)
user_emb, item_emb = utils.read_cf_embeddings(model_name, checkpoint_name)
train_data, test_data, train_codebook_data, test_codebook_data, item_num, user_num = dataset.read_data(data_name)

valid_data, valid_codebook_data = utils.data_construction(test_data, test_codebook_data, args.item_limit)
test_data, test_codebook_data = utils.data_construction(test_data, test_codebook_data, args.item_limit)

if args.no_data_augment is False:
    train_data, train_codebook_data = utils.data_augment(train_data, train_codebook_data, item_limit=args.item_limit)
elif args.no_data_augment is True:
    train_data, train_codebook_data = utils.data_construction(train_data, train_codebook_data, args.item_limit)
else:
    NotImplementedError

# 2. LLM Backbone Training
if args.no_train is False:
    # [INNOVATION 2] Pass recent_k to constructor
    train_rec_dataset = dataset.LLM4RecTrainDataset(train_data, train_codebook_data, args.no_shuffle,
                                                    recent_k=args.recent_k)
    train_rec_loader = DataLoader(train_rec_dataset, batch_size=args.batch, shuffle=True, drop_last=False)

    valid_rec_dataset = dataset.LLM4RecDataset(valid_data, valid_codebook_data, args.no_shuffle, recent_k=args.recent_k)
    valid_rec_loader = DataLoader(valid_rec_dataset, batch_size=args.batch, shuffle=False, drop_last=False)

    train.backbone(data_name, train_rec_loader, valid_rec_loader, user_emb, item_emb, item_num, args, device)

# 3. Evaluation
test_rec_dataset = dataset.LLM4RecDataset(test_data, test_codebook_data, no_shuffle=True, recent_k=args.recent_k)
test_rec_loader = DataLoader(test_rec_dataset, batch_size=args.batch, shuffle=False, drop_last=False)

test.backbone(data_name, test_rec_loader, user_emb, item_emb, item_num, args, device)
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go CAR-Rec")

    # ----------------- t5 ----------------
    # [FIX] Reduced item_limit to prevent T5 truncation of recent items
    parser.add_argument('--item_limit', type=int, default=50,
                        help="the maximum number of item list")
    parser.add_argument('--k', type=int, default=10,
                        help="the number of top k setting")
    parser.add_argument('--kernel', type=int, default=32,
                        help="the kernel size of average pooling")
    parser.add_argument('--lr', type=float, default=5e-5,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-2,
                        help="the weight decay for optimizer")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--batch', type=int, default=32,
                        help="the batch size of training and test")
    parser.add_argument('--valid_batch', type=int, default=24,
                        help="the batch size of valid set")
    parser.add_argument('--no_data_augment', action='store_true', default=False,
                        help="whether apply the data augmentation process or not")
    parser.add_argument('--no_shuffle', action='store_true', default=True,
                        help="if shuffle the item list of each sample or not. Default=True for Temporal Prompting.")
    parser.add_argument('--train_from_checkpoint', action='store_true', default=False,
                        help="if training from the checkpoint: True or False (default).")
    parser.add_argument('--source_length', type=int, default=512,
                        help="the maximum length of input tokens")
    parser.add_argument('--similarity', type=str, default='cos',
                        help="the similarity metric. available: [cos, MSE]")
    parser.add_argument('--target_length', type=int, default=20,
                        help="the maximum length of output tokens")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--no_train', action='store_true', default=False)
    parser.add_argument('--is_unseen', action='store_true', default=False)
    parser.add_argument('--decoder_prepend', type=str,
                        default="The interaction history shows that the user preference could be")

    # ---------------- vq -----------------------
    parser.add_argument('--n_token', type=int, default=256,
                        help="the token number of each codebook")
    parser.add_argument('--n_book', type=int, default=3,
                        help="the number of codebooks")
    parser.add_argument('--vq', action='store_true', default=False,
                        help="if run vq: True or False (default).")
    parser.add_argument('--train_vq', action='store_true', default=False,
                        help="if training the VQ checkpoint: True or False (default).")
    parser.add_argument('--vq_model', type=str, default='MQ',
                        help="available indexing model: [RQ, MQ]")

    # --------------- general --------------------
    parser.add_argument('--cuda', type=int, default=0,
                        help="the used cuda")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--dataset', type=str, default='LastFM',
                        help="available datasets: [Beauty, LastFM, ML1M, Clothing]")

    # [INNOVATION ARGS]
    # Strategy 1: Dual-View Semantic Alignment
    parser.add_argument('--enable_dual', action='store_true', default=False,
                        help="Enable Dual-Alignment Tokenizer (Innovation Strategy 1)")
    parser.add_argument('--semantic_dim', type=int, default=768,
                        help="Dimension of the pre-trained semantic embeddings (Sentence-T5)")
    parser.add_argument('--fgm_eps', type=float, default=0.02,
                        help="Epsilon for FGM adversarial perturbation")
    parser.add_argument('--con_alpha', type=float, default=0.1,
                        help="Weight alpha for Consistency Loss")
    parser.add_argument('--alpha_align', type=float, default=0.1,
                        help="Weight for the semantic alignment loss")

    # [FIXED MISSING ARGUMENT]
    parser.add_argument('--temperature', type=float, default=0.07,
                        help="Temperature for contrastive-like logits scaling")

    # Strategy 2: Temporal-Aware Prompting
    parser.add_argument('--recent_k', type=int, default=3,
                        help="Number of recent items to highlight in Temporal-Aware Prompting")

    # Strategy 3: Hard Negatives
    parser.add_argument('--enable_hard_neg', action='store_true', default=True,
                        help="Enable Hard Negative Mining (Innovation Strategy 3)")
    parser.add_argument('--hard_neg_k', type=int, default=50,
                        help="Pool size for hard negatives candidates")

    return parser.parse_args()
import torch
import os

# 定义路径 (基于您的目录结构，脚本在 code/ 目录下)
ITEM_LIST_PATH = '../data/Clothing/item_list.txt'
LGN_PATH = '../src/lgn/lgn-Clothing-64.pth.tar'
SEMANTIC_PATH = '../data/Clothing/semantic_emb.pt'


def check():
    print("====== 开始 ID 对齐“X光”检查 ======")

    # 1. 检查 item_list.txt (CAR-Rec 的字典)
    if os.path.exists(ITEM_LIST_PATH):
        with open(ITEM_LIST_PATH, 'r', encoding='utf-8') as f:
            items = [line.strip() for line in f.readlines()]
        n_items_txt = len(items)
        print(f"[1] item_list.txt 物品数量: {n_items_txt}")
        print(f"    -> ID 0 物品名: {items[0]}")
        print(f"    -> ID 10 物品名: {items[10] if len(items) > 10 else 'N/A'}")
    else:
        print(f"❌ 找不到文件: {ITEM_LIST_PATH}")
        return

    # 2. 检查 LightGCN 权重 (LightGCN 的字典)
    if os.path.exists(LGN_PATH):
        # 注意：这里需要适配 map_location 以防你在只有 CPU 的机器上跑
        lgn_data = torch.load(LGN_PATH, map_location='cpu')

        # 通常 LightGCN 的权重保存在 'embedding.weight' 或类似的 key 中
        # 我们尝试打印 keys 来看看结构
        print(f"[2] LightGCN 权重文件 Keys: {lgn_data.keys()}")

        # 假设是标准结构，尝试获取 embedding
        if 'embedding_user.weight' in lgn_data:  # 可能是这种结构
            n_users = lgn_data['embedding_user.weight'].shape[0]
            n_items_lgn = lgn_data['embedding_item.weight'].shape[0]
            print(f"    -> LightGCN 里的 User 数量: {n_users}")
            print(f"    -> LightGCN 里的 Item 数量: {n_items_lgn}")
        elif 'embedding.weight' in lgn_data:  # 或者是这种
            print(f"    -> Embedding Shape: {lgn_data['embedding.weight'].shape}")
            # 这里很难区分 user/item，通常是混在一起的，需要看代码逻辑
        else:
            # 直接打印 model state_dict 的形状
            for k, v in lgn_data.items():
                if 'item' in k or 'embedding' in k:
                    print(f"    -> Key: {k}, Shape: {v.shape}")
                    if 'item' in k:
                        n_items_lgn = v.shape[0]
    else:
        print(f"❌ 找不到 LightGCN 权重: {LGN_PATH}")

    # 3. 检查 语义向量 (Semantic Embeddings)
    if os.path.exists(SEMANTIC_PATH):
        sem_data = torch.load(SEMANTIC_PATH, map_location='cpu')
        n_items_sem = sem_data.shape[0]
        print(f"[3] 语义向量 (semantic_emb.pt) 数量: {n_items_sem}")
    else:
        print(f"❌ 找不到语义向量: {SEMANTIC_PATH}")

    print("\n====== 诊断结果 ======")

    # 核心判断逻辑
    try:
        if n_items_txt != n_items_lgn:
            print(f"🚨🚨🚨 严重警报：数量不匹配！(Mismatch)")
            print(f"CAR-Rec 认为是 {n_items_txt} 个物品，但 LightGCN 是为 {n_items_lgn} 个物品训练的。")
            print("结论：这是 100% 的 ID 错位。必须重训 LightGCN。")
        elif n_items_txt != n_items_sem:
            print(f"🚨 警报：语义向量数量 ({n_items_sem}) 与 物品列表 ({n_items_txt}) 不一致。")
        else:
            print(f"✅ 数量一致 ({n_items_txt})。但仍需警惕 ID 顺序是否打乱。")
            print("建议：如果数量一致但效果极差，通常是因为 ID 0 在两个系统中代表了不同的物品。")
    except:
        print("无法自动对比，请人工查看上述输出的数字。")


if __name__ == "__main__":
    check()
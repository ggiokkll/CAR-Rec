import os
import torch
import gzip
import json
import ast
from sentence_transformers import SentenceTransformer
import sys

# ================= 配置区域 =================
# 1. 基础路径
BASE_DATA_DIR = '../data/Clothing'

# 2. 关键文件路径
# [输入] 原始元数据 (文件名通常很长，请确认您的实际文件名)
# 常见文件名: meta_Clothing_Shoes_and_Jewelry.json.gz
RAW_META_FILE = os.path.join(BASE_DATA_DIR, 'meta_Clothing_Shoes_and_Jewelry.json.gz')

# [输入] 需要修复的 Item List
BAD_ITEM_LIST = os.path.join(BASE_DATA_DIR, 'item_list.txt')

# [输出]
OUTPUT_TEXT_FILE = os.path.join(BASE_DATA_DIR, 'item_list_fixed.txt')
OUTPUT_EMB_FILE = os.path.join(BASE_DATA_DIR, 'semantic_emb.pt')

# 3. 模型路径
MODEL_PATH = '../src/sentence-t5-base'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ===========================================

def check_paths():
    if not os.path.exists(RAW_META_FILE):
        print(f"❌ 错误: 找不到元数据文件: {RAW_META_FILE}")
        print("   -> 请检查文件名是否正确 (例如是否是 meta_Clothing...json.gz)")
        sys.exit(1)
    if not os.path.exists(BAD_ITEM_LIST):
        print(f"❌ 错误: 找不到 item_list.txt: {BAD_ITEM_LIST}")
        sys.exit(1)
    print("✅ 路径检查通过。")


def load_clothing_meta():
    """双模读取 Amazon Clothing 元数据"""
    print(f"正在读取原始元数据: {RAW_META_FILE}")
    meta_dict = {}

    count = 0
    success_count = 0

    with gzip.open(RAW_META_FILE, 'rt', encoding='utf-8') as f:
        for line in f:
            count += 1
            data = None

            # --- 双模解析 (兼容 JSON 和 Python Eval) ---
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(line)
                except Exception:
                    pass

            if data is None:
                continue

            # 提取数据
            try:
                asin = data.get('asin', '')
                title = data.get('title', '')
                brand = data.get('brand', '')

                # 提取分类 (categories 通常是列表的列表)
                categories = data.get('categories', [[]])
                cat_str = ""
                if categories and isinstance(categories[0], list):
                    cat_str = " ".join(categories[0])
                elif isinstance(categories, list):
                    cat_str = " ".join(categories)

                # 组合丰富语义
                full_text = f"{title} {brand} {cat_str}".strip()

                if asin:
                    meta_dict[str(asin)] = full_text
                    success_count += 1
            except Exception:
                continue

    print(f"✅ 元数据加载完成。")
    print(f"   - 总行数: {count}")
    print(f"   - 成功解析: {success_count}")
    return meta_dict


def fix_item_list(meta_dict):
    print(f"\nSTEP 1: 修复 Item List 文本")

    fixed_lines = []
    missing_count = 0

    with open(BAD_ITEM_LIST, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"原始 item_list 共 {len(lines)} 行。")

    # ⚠️ 关键修正：检测并跳过表头
    start_idx = 0
    if len(lines) > 0 and "org_id" in lines[0]:
        print("ℹ️ 检测到表头 'org_id'，已跳过第一行。")
        start_idx = 1

    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line: continue

        parts = line.split()
        org_id = parts[0]  # ASIN

        if org_id in meta_dict:
            real_text = meta_dict[org_id]
        else:
            real_text = f"Unknown Product {org_id}"
            missing_count += 1
            if missing_count <= 5:
                print(f"⚠️ 警告: ASIN {org_id} 未找到元数据。")

        fixed_lines.append(real_text)

    # 写入新文件
    with open(OUTPUT_TEXT_FILE, 'w', encoding='utf-8') as f:
        for text in fixed_lines:
            f.write(text + '\n')

    print(f"✅ 文本修复完成！有效物品数: {len(fixed_lines)} (应为 23033)")
    print(f"缺失元数据数: {missing_count}")

    # 预览
    print("-" * 30)
    print("预览前 3 行内容:")
    for k in range(min(3, len(fixed_lines))):
        print(f"[{k}] {fixed_lines[k]}")
    print("-" * 30)

    return fixed_lines


def generate_embeddings(text_list):
    print(f"\nSTEP 2: 生成语义向量")
    print(f"加载模型: {MODEL_PATH}")

    try:
        model = SentenceTransformer(MODEL_PATH, device=device)
        embeddings = model.encode(text_list, show_progress_bar=True, convert_to_tensor=True)

        print(f"生成的 Embedding 形状: {embeddings.shape}")
        torch.save(embeddings.cpu(), OUTPUT_EMB_FILE)
        print(f"✅ 向量已保存至: {OUTPUT_EMB_FILE}")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")


if __name__ == "__main__":
    check_paths()
    meta_dict = load_clothing_meta()
    clean_texts = fix_item_list(meta_dict)
    if clean_texts:
        generate_embeddings(clean_texts)
        print("\n🎉 Clothing 数据修复完成！")
        print("请执行最后一步：")
        print("1. 进入 data/Clothing 目录")
        print("2. 删除旧的 item_list.txt")
        print("3. 重命名 item_list_fixed.txt -> item_list.txt")
        print("4. 回到 code 目录运行: python main.py --dataset Clothing --vq --train_vq")
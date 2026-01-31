import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import sys

# ================= 配置区域 =================
# 1. 基础路径 (根据您的 dataset.py 推断)
BASE_DATA_DIR = '../data/ML1M'

# 2. 关键文件路径
# [输入] 原始元数据 (格式: MovieID::Title::Genres)
# 注意：MovieLens 数据集通常在解压后的 ml-1m 文件夹里，请根据实际情况确认
# 如果您的 movies.dat 直接在 ML1M 下，请用第一行；如果在子目录，请用第二行
RAW_META_FILE = os.path.join(BASE_DATA_DIR, 'movies.dat')
# RAW_META_FILE = os.path.join(BASE_DATA_DIR, 'ml-1m', 'movies.dat')

# [输入] 需要修复的 Item List (当前包含 "org_id remap_id" 表头)
BAD_ITEM_LIST = os.path.join(BASE_DATA_DIR, 'item_list.txt')

# [输出] 修复后的纯文本文件
OUTPUT_TEXT_FILE = os.path.join(BASE_DATA_DIR, 'item_list_fixed.txt')
# [输出] 生成的语义向量文件
OUTPUT_EMB_FILE = os.path.join(BASE_DATA_DIR, 'semantic_emb.pt')

# 3. 模型配置
# 建议使用本地模型路径，或者使用 HuggingFace 在线模型名称 (如 'all-MiniLM-L6-v2')
# 如果您之前用的是 sentence-t5-base，请确保路径正确
MODEL_PATH = '../src/sentence-t5-base'
# 如果本地没有，可以取消下面这行的注释使用在线模型：
# MODEL_PATH = 'sentence-transformers/all-MiniLM-L6-v2'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ===========================================

def check_paths():
    """检查文件是否存在"""
    if not os.path.exists(RAW_META_FILE):
        print(f"❌ 错误: 找不到元数据文件: {RAW_META_FILE}")
        print("   -> 请确认 movies.dat 是否在 data/ML1M/ 目录下，或者在 data/ML1M/ml-1m/ 下。")
        sys.exit(1)
    if not os.path.exists(BAD_ITEM_LIST):
        print(f"❌ 错误: 找不到 item_list.txt: {BAD_ITEM_LIST}")
        sys.exit(1)
    print("✅ 路径检查通过。")


def load_movie_meta():
    """
    读取 ML1M movies.dat
    格式: MovieID::Title::Genres
    编码: Latin-1
    """
    print(f"正在读取原始元数据: {RAW_META_FILE}")
    meta_dict = {}

    try:
        # 使用 python 引擎处理多字符分隔符 '::'
        # MovieLens 通常是 Latin-1 编码
        df = pd.read_csv(RAW_META_FILE, sep='::', header=None,
                         names=['id', 'title', 'genres'],
                         engine='python', encoding='latin-1')

        for _, row in df.iterrows():
            # 数据清洗: 将流派中的 '|' 替换为空格，增加语义可读性
            # 例如: "Animation|Children's" -> "Animation Children's"
            clean_genres = str(row['genres']).replace('|', ' ')

            # 组合文本: Title + Genres
            # 例如: "Toy Story (1995) Animation Children's Comedy"
            full_text = f"{row['title']} {clean_genres}"

            # 存入字典: str(ID) -> Text
            meta_dict[str(row['id'])] = full_text

        print(f"✅ 元数据加载完成，共 {len(meta_dict)} 部电影信息。")
        return meta_dict

    except Exception as e:
        print(f"❌ 读取 movies.dat 失败: {e}")
        sys.exit(1)


def fix_item_list(meta_dict):
    """
    读取坏的 item_list.txt (带表头, 格式: org_id remap_id)
    生成好的 item_list_fixed.txt (纯文本)
    """
    print(f"\nSTEP 1: 修复 Item List 文本")

    fixed_lines = []
    missing_count = 0

    with open(BAD_ITEM_LIST, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"原始文件共 {len(lines)} 行。")

    # 检测并跳过表头
    start_idx = 0
    if len(lines) > 0 and "org_id" in lines[0]:
        print("ℹ️ 检测到表头 'org_id'，已跳过第一行。")
        start_idx = 1

    # 遍历处理
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line: continue

        parts = line.split()
        # 第一列通常是 org_id
        org_id = parts[0]

        if org_id in meta_dict:
            real_text = meta_dict[org_id]
        else:
            # 如果找不到，用占位符防止报错，但记录警告
            real_text = f"Unknown Movie {org_id}"
            missing_count += 1
            if missing_count <= 5:
                print(f"⚠️ 警告: ID {org_id} 在 movies.dat 中未找到。")

        fixed_lines.append(real_text)

    # 写入新文件
    with open(OUTPUT_TEXT_FILE, 'w', encoding='utf-8') as f:
        for text in fixed_lines:
            f.write(text + '\n')

    print(f"✅ 文本修复完成！有效物品数: {len(fixed_lines)}")
    if missing_count > 0:
        print(f"⚠️ 共有 {missing_count} 个物品缺失元数据。")

    # 预览前几行
    print("-" * 30)
    print("预览前 3 行内容:")
    for k in range(min(3, len(fixed_lines))):
        print(f"[{k}] {fixed_lines[k]}")
    print("-" * 30)

    return fixed_lines


def generate_embeddings(text_list):
    """使用 Sentence-Transformer 生成向量"""
    print(f"\nSTEP 2: 生成语义向量 (Semantic Embeddings)")
    print(f"加载模型: {MODEL_PATH}")
    print(f"运行设备: {device}")

    try:
        model = SentenceTransformer(MODEL_PATH, device=device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请检查 MODEL_PATH 是否正确，或尝试联网下载 'all-MiniLM-L6-v2'")
        return

    print("开始编码 (Encoding)...")
    # 生成向量
    embeddings = model.encode(text_list, show_progress_bar=True, convert_to_tensor=True)

    print(f"生成的 Embedding 形状: {embeddings.shape}")

    # 保存 .pt 文件
    torch.save(embeddings.cpu(), OUTPUT_EMB_FILE)
    print(f"✅ 向量已保存至: {OUTPUT_EMB_FILE}")


if __name__ == "__main__":
    # 1. 检查路径
    check_paths()

    # 2. 加载元数据
    meta_dict = load_movie_meta()

    # 3. 修复文本列表
    clean_texts = fix_item_list(meta_dict)

    # 4. 生成语义向量
    if clean_texts:
        generate_embeddings(clean_texts)

        print("\n🎉 ====== 全部完成 ======")
        print("请执行以下最后一步操作：")
        print(f"1. 进入目录: cd {BASE_DATA_DIR}")
        print(f"2. 备份原文件 (可选): mv item_list.txt item_list.bak")
        print(f"3. 替换新文件: mv item_list_fixed.txt item_list.txt")
        print("4. 重新运行 main.py --vq --train_vq")

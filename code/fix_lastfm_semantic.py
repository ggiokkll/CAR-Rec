import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import sys

# ================= 配置区域 (基于你的目录结构) =================
# 1. 相对路径设置 (假设脚本运行在 code/ 目录下)
BASE_DATA_DIR = '../data/LastFM'
RAW_META_DIR = os.path.join(BASE_DATA_DIR, 'hetrec2011-lastfm-2k')
LOCAL_MODEL_PATH = '../src/sentence-t5-base'  # 指向你本地的 sentence-t5

# 2. 关键文件路径
# [输入] 原始元数据 (包含真实歌手名字)
RAW_ARTISTS_FILE = os.path.join(RAW_META_DIR, 'artists.dat')
# [输入] 当前错误的 ID 映射文件 (包含 ID 顺序: org_id remap_id)
BAD_ITEM_LIST = os.path.join(BASE_DATA_DIR, 'item_list.txt')

# [输出] 修复后的文件
OUTPUT_TEXT_FILE = os.path.join(BASE_DATA_DIR, 'item_list_fixed.txt')
OUTPUT_EMB_FILE = os.path.join(BASE_DATA_DIR, 'semantic_emb.pt')


# =============================================================

def check_paths():
    """检查所有必要文件是否存在"""
    print("正在检查文件路径...")
    if not os.path.exists(RAW_ARTISTS_FILE):
        raise FileNotFoundError(f"找不到原始元数据文件: {RAW_ARTISTS_FILE}")
    if not os.path.exists(BAD_ITEM_LIST):
        raise FileNotFoundError(f"找不到需要修复的 Item List: {BAD_ITEM_LIST}")
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"找不到本地模型目录: {LOCAL_MODEL_PATH}")
    print("✅ 路径检查通过。")


def load_artist_meta():
    """加载原始 LastFM 歌手数据: ID -> Name"""
    print(f"正在读取原始元数据: {RAW_ARTISTS_FILE}")
    meta_dict = {}

    # LastFM artists.dat 格式通常为: id \t name \t url ...
    # 可能会有编码问题，先尝试 utf-8，不行换 latin-1
    try:
        df = pd.read_csv(RAW_ARTISTS_FILE, sep='\t', usecols=[0, 1], names=['id', 'name'], encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 读取失败，切换为 Latin-1 编码...")
        df = pd.read_csv(RAW_ARTISTS_FILE, sep='\t', usecols=[0, 1], names=['id', 'name'], encoding='latin-1')

    # 构建字典: string(id) -> string(name)
    for _, row in df.iterrows():
        meta_dict[str(row['id'])] = str(row['name'])

    print(f"✅ 原始元数据加载完成，共 {len(meta_dict)} 个歌手信息。")
    return meta_dict


def fix_text_list(meta_dict):
    """根据坏文件的顺序，匹配出正确的歌手名字"""
    print(f"\nSTEP 1: 修复 item_list.txt 内容")

    fixed_lines = []
    missing_count = 0

    with open(BAD_ITEM_LIST, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"当前 item_list.txt 共有 {len(lines)} 行 (即 Total Items)。")

    # 逐行处理，保持顺序绝对不变
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue

        # 格式通常是: org_id remap_id (例如 "1 0")
        org_id = parts[0]

        if org_id in meta_dict:
            real_name = meta_dict[org_id]
        else:
            # 如果元数据里找不到这个ID，用占位符，避免报错
            real_name = f"Unknown Artist {org_id}"
            missing_count += 1
            if missing_count < 5:  # 只打印前几个缺失的
                print(f"⚠️ 警告: ID {org_id} 在 artists.dat 中找不到，已替换为占位符。")

        fixed_lines.append(real_name)

    # 写入新文件
    with open(OUTPUT_TEXT_FILE, 'w', encoding='utf-8') as f:
        for name in fixed_lines:
            f.write(name + '\n')

    print(f"✅ 文本列表修复完成！缺失数: {missing_count}")
    print(f"新文件已保存至: {OUTPUT_TEXT_FILE}")

    # 打印前几行预览
    print(f"--- 预览前 3 行 ---")
    for k in range(min(3, len(fixed_lines))):
        print(f"ID {k}: {fixed_lines[k]}")
    print("-------------------")

    return fixed_lines


def generate_embeddings(text_list):
    """使用本地 Sentence-T5 生成 Embedding"""
    print(f"\nSTEP 2: 使用 Sentence-T5 生成语义向量")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在加载本地模型: {LOCAL_MODEL_PATH}")
    print(f"运行设备: {device}")

    # 加载本地 Sentence-T5
    try:
        model = SentenceTransformer(LOCAL_MODEL_PATH, device=device)
    except Exception as e:
        print(f"❌ 模型加载失败，请检查路径。错误信息: {e}")
        return

    print("开始编码 (Encoding)... 这可能需要几分钟...")
    # encode 方法会自动处理 batching
    embeddings = model.encode(text_list, show_progress_bar=True, convert_to_tensor=True)

    print(f"生成 Embeddings 形状: {embeddings.shape}")

    # 保存 .pt 文件
    torch.save(embeddings.cpu(), OUTPUT_EMB_FILE)
    print(f"✅ 语义向量已保存至: {OUTPUT_EMB_FILE}")


if __name__ == "__main__":
    try:
        check_paths()

        # 1. 加载字典
        meta_dict = load_artist_meta()

        # 2. 修复文本
        clean_texts = fix_text_list(meta_dict)

        # 3. 生成向量
        if clean_texts:
            generate_embeddings(clean_texts)

        print("\n🎉 ====== 全部完成 ======")
        print("请执行最后一步操作：")
        print(f"1. 备份原文件: rename {BAD_ITEM_LIST} item_list.bak")
        print(f"2. 替换新文件: rename {OUTPUT_TEXT_FILE} item_list.txt")
        print("3. 重新运行 main.py")

    except Exception as e:
        print(f"\n❌ 程序发生错误: {e}")
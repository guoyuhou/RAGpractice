import os
import sys
import json
import ast
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

# --- [关键] 确保能导入您的RAG函数和FRAMES_bench中的函数 ---
# 假设此脚本与RAG_Bench文件夹和simple_rag文件夹在同一目录下
# (如果目录结构不同，请相应调整)
from simple_rag import main as my_rag_functions
from RAG_Bench.FRAMES_bench import get_content_from_urls # 复用网页抓取函数

# --- 配置区 ---
DASHSCOPE_API_KEY = "sk-fb8191fb105b439d9ffd2880a9d9be7c"
# 初始化客户端
dashscope_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def analyze_chunking(test_samples):
    """
    分析不同分块方法对文本的处理结果。
    """
    print("\n" + "="*50)
    print("🚀 开始分析不同分块(Chunking)方法的尺寸...")
    print("="*50)

    chunking_methods = {
        'fix_size': my_rag_functions.chunk_by_fix_size,
        'recursive': my_rag_functions.chunk_recursively,
        'semantic': lambda text: my_rag_functions.chunk_semantic(text, client=dashscope_client)
    }
    
    # 存储所有分析结果
    results = {name: [] for name in chunking_methods.keys()}

    for sample in tqdm(test_samples, desc="Analyzing documents"):
        knowledge_urls = ast.literal_eval(sample['wiki_links'])
        full_context_text = get_content_from_urls(knowledge_urls)
        
        if not full_context_text:
            continue

        for name, chunk_func in chunking_methods.items():
            chunks = chunk_func(full_context_text)
            chunk_lengths = [len(c) for c in chunks]
            
            if not chunk_lengths: # 如果没有产生任何chunk
                continue

            stats = {
                'num_chunks': len(chunks),
                'avg_chunk_size': np.mean(chunk_lengths),
                'std_dev_size': np.std(chunk_lengths), # 标准差，看尺寸是否稳定
                'min_size': np.min(chunk_lengths),
                'max_size': np.max(chunk_lengths)
            }
            results[name].append(stats)

    # --- 打印最终的汇总报告 ---
    print("\n" + "="*60)
    print("📊 分块策略分析报告")
    print("="*60)
    
    for name, stats_list in results.items():
        if not stats_list:
            print(f"\n--- 方法: {name} ---")
            print("未能生成任何分块。")
            continue
            
        avg_num_chunks = np.mean([s['num_chunks'] for s in stats_list])
        avg_chunk_size = np.mean([s['avg_chunk_size'] for s in stats_list])
        avg_std_dev = np.mean([s['std_dev_size'] for s in stats_list])
        
        print(f"\n--- 方法 (Method): {name} ---")
        print(f"    平均生成分块数量 (Avg. Chunks): {avg_num_chunks:.2f}")
        print(f"    平均分块长度 (Avg. Chunk Size): {avg_chunk_size:.2f} 字符")
        print(f"    平均尺寸标准差 (Avg. Std Dev): {avg_std_dev:.2f}")
    print("="*60)


if __name__ == "__main__":
    # 1. 加载数据集
    print("正在加载FRAMES数据集...")
    frames_dataset = load_dataset("google/frames-benchmark")
    test_set = frames_dataset['test']
    
    # 随机挑选3个样本进行快速分析
    shuffled_test_set = test_set.shuffle(seed=42)
    test_samples = list(shuffled_test_set.select(range(3)))
    print(f"已随机挑选 {len(test_samples)} 个样本进行尺寸分析。")

    # 2. 运行分析
    analyze_chunking(test_samples)
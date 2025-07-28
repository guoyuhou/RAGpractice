from datasets import load_dataset
from bs4 import BeautifulSoup
import requests
from simple_rag import main as my_rag_functions
from openai import OpenAI
from mistralai import Mistral
import os
from tqdm import tqdm
import ast
import json
import nltk
import time
from simple_rag import main as my_rag_function
import numpy as np
import pandas as pd

DASHSCOPE_API_KEY = "sk-fb8191fb105b439d9ffd2880a9d9be7c"
dashscope_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def get_content_from_urls(urls, max_retries=3, initial_delay=1):
    """
    接收一个URL列表,抓取每个页面的文本内容并合并。
    """
    full_text_from_web = ''
    for url in urls:
        if url is None:
            continue
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    full_text_from_web += p.get_text() + '\n'

                break
            
            except requests.exceptions.RequestException as e:
                print(f'抓取URL失败:{url},尝试次数{attempt + 1}/{max_retries}...')
                print(f'错误{e}')

                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f'将在{delay}秒后尝试')
                    time.sleep(delay)
                else:
                    print(f'已达到最大重试次数,放弃抓取URL:{url}')
    
    return full_text_from_web

def analyze_and_collect_chunks(test_samples):
    """
    运行semantic  chunking并收集所有chunk数据
    """
    print("\n" + "="*50)
    print("🚀 开始运行Semantic Chunker并收集数据...")
    print("="*50)

    all_chunks_data = []

    for i, sample in enumerate(tqdm(test_samples, desc='Processing documents')):
        try:
            knowledge_urls = ast.literal_eval(sample['wiki_links'])
        except (ValueError, SyntaxError):
            continue

        full_context_text = get_content_from_urls(knowledge_urls)
        if not full_context_text:
            continue
        
        semantic_chunks = my_rag_function.chunk_semantic(full_context_text, client=dashscope_client)

        for chunk in semantic_chunks:
            all_chunks_data.append({
                'source_id': i,
                'chunk_content': chunk,
                'chunk_size': len(chunk)
            })

    df = pd.DataFrame(all_chunks_data)
    output_path = 'semantic_chunks_analysis.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 数据收集完成！共生成 {len(df)} 个语义块。")
    print(f"结果已保存至: {output_path}")
    return df

if __name__ == "__main__":
    # 1. 加载数据集
    print("正在加载FRAMES数据集...")
    frames_dataset = load_dataset('google/frames-benchmark')
    test_set = frames_dataset['test']
    
    # 随机挑选样本进行分析 (可以增加数量以获得更可靠的统计结果)
    shuffled_test_set = test_set.shuffle(seed=42)
    test_samples = list(shuffled_test_set.select(range(50))) # 使用全部50个样本
    print(f"已随机挑选 {len(test_samples)} 个样本进行分块分析。")

    # 2. 运行分析和收集
    analyze_and_collect_chunks(test_samples)
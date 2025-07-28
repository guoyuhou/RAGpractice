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
nltk.download('punkt')


dashscope_key = 'sk-fb8191fb105b439d9ffd2880a9d9be7c'
mistral_key = 'NMRPcolmwB7PEegVbzi6TId4IUVCfCFB'
deepseek_key = "sk-09IfmX7mg7MJQg0U0OOM8AVXBrGC7Um887xvdb4H3Tbn16sQ"
Kimi_key = 'sk-vqaNEYl0Z3vf8GITnaebDySglXyuReqPLzGDlISr9qyKYEYo'


dashscope_client = OpenAI(api_key=dashscope_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
mistral_client = Mistral(api_key=mistral_key)
deepseek_client = OpenAI(api_key=deepseek_key, base_url="https://api.lkeap.cloud.tencent.com/v1")
kimik2_client = OpenAI(api_key=Kimi_key, base_url="https://api.moonshot.cn/v1")




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

def save_results(results_data, filename_prefix):
    json_filename = f'{filename_prefix}_results.json'

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    print(f'JSON结果已经保存至: {json_filename}')

def run_chunking_experiment(test_samples):
    print('\n' + '='*50)
    print('开始实验一: 测试不同Chunking方法(动态Top-K)')
    print('\n' + '='*50)

    
    chunking_configs = {
        'fix_size': {
            'function': my_rag_functions.chunk_by_fix_size,
            'k': 5
        },
        'recursive':{
            'function': my_rag_functions.chunk_recursively,
            'k': 14
        },
        'semantic': {
            'function': my_rag_functions.chunk_semantic,
            'k': 68
        }
    }

    all_results = []

    for sample in tqdm(test_samples, desc='Processing Samples for Chunking experiment'):
        prompt = sample['Prompt']
        ground_truth_answer = sample['Answer']
        knowledge_urls = ast.literal_eval(sample['wiki_links'])

        full_context_text = get_content_from_urls(knowledge_urls)
        if not full_context_text: continue

        sample_results = {
            'question': prompt,
            'standard_answer': ground_truth_answer
        }
        
        for name, config in chunking_configs.items():
            chunk_func = config['function']
            top_k = config['k']

            if name == 'semantic':
                text_chunks = chunk_func(full_context_text, client=dashscope_client)
            else:
                text_chunks = chunk_func(full_context_text)

            chunks_embeddings = my_rag_functions.vector_chunks_embedding(text_chunks, client=dashscope_client)
            retrieved_list = my_rag_functions.semantic_search(dashscope_client, prompt, text_chunks, chunks_embeddings, k=top_k)
            retrieved_content = ' '.join(retrieved_list)
            generate_answer = my_rag_functions.generate_response_deepseek(deepseek_client, retrieved_content, prompt)

            sample_results[f'generated_answer_{name}'] = generate_answer
            sample_results[f'is_correct_{name}'] = (generate_answer.strip().lower() == ground_truth_answer.strip().lower())

        all_results.append(sample_results)

    save_results(all_results, 'experiment_chunking')
    print('✅ 实验一(动态Top-K)完成！')

def run_retrieval_experiment(test_sample):
    print('\n' + '='*50)
    print('实验二，检测不同检索(Retrieval)方法')
    print('='*50)

    all_results = []

    for sample in tqdm(test_sample, desc='Processing Samples for Retrieval Experiment'):
        prompt = sample['Prompt']
        ground_truth_answer = sample['Answer']
        knowledge_urls = ast.literal_eval(sample['wiki_links'])

        full_context_text = get_content_from_urls(knowledge_urls)
        if not full_context_text: continue

        text_chunks = my_rag_functions.chunk_by_fix_size(full_context_text)
        chunks_embeddings = my_rag_functions.vector_chunks_embedding(text_chunks, client=dashscope_client)

        sample_results = {
            'question': prompt,
            'standard_answer': ground_truth_answer
        }
        # 测试Dense Retrieval
        dense_results = my_rag_functions.semantic_search(dashscope_client, prompt, text_chunks, chunks_embeddings)
        retrieved_context_dense = ' '.join(dense_results)
        generated_answer_dense = my_rag_functions.generate_response_deepseek(deepseek_client, retrieved_context_dense, prompt)
        sample_results["generated_answer_dense"] = generated_answer_dense
        sample_results["is_correct_dense"] = (generated_answer_dense.strip().lower() == ground_truth_answer.strip().lower())

        # 测试Sparse Retrieval
        sparse_results = my_rag_functions.sparse_search_bm25(prompt, text_chunks)
        retrieved_context_sparse = ' '.join(sparse_results)
        generated_answer_sparse = my_rag_functions.generate_response_deepseek(deepseek_client, retrieved_context_sparse, prompt)
        sample_results["generated_answer_sparse"] = generated_answer_sparse
        sample_results["is_correct_sparse"] = (generated_answer_sparse.strip().lower() == ground_truth_answer.strip().lower())
            
        # 测试Hybrid Retrieval
        hybrid_results = my_rag_functions.hybrid_search_rrf(prompt, text_chunks, dense_results, sparse_results)
        retrieved_context_hybrid = ' '.join(hybrid_results)
        generated_answer_hybrid = my_rag_functions.generate_response_deepseek(deepseek_client, retrieved_context_hybrid, prompt)
        sample_results["generated_answer_hybrid"] = generated_answer_hybrid
        sample_results["is_correct_hybrid"] = (generated_answer_hybrid.strip().lower() == ground_truth_answer.strip().lower())

        # 测试Hybrid Retrieval + Rerank
        initial_candidates = my_rag_functions.hybrid_search_rrf(prompt, text_chunks, dense_results, sparse_results, k=20)
        reranked_results = my_rag_functions.rerank_results(prompt, initial_candidates, top_n=5)
        retrieved_context_reranked = ' '.join(reranked_results)
        generated_answer_reranked = my_rag_functions.generate_response_deepseek(deepseek_client, retrieved_context_reranked, prompt)
        sample_results['generated_answer_reranked'] = generated_answer_reranked
        sample_results['is_correct_reranked'] = (generated_answer_reranked.strip().lower() == ground_truth_answer.strip().lower())

        all_results.append(sample_results)

    save_results(all_results, 'experiment_retrieval')
    print('✅ 实验二完成！')

def run_best_rag_experiment(test_samples):
    print('\n' + '='*50)
    print('🚀 开始最终测试: 运行最强RAG组合 (修正版)')
    print('   - 分块: Semantic Chunking')
    print('   - 检索: Hybrid Search + Rerank')
    print('   - 生成: DeepSeek')
    print("="*50)
    
    all_results = []
    
    # --- [核心修正] 调整K值以匹配上下文预算 ---
    # 初步检索的K值应该采用为semantic chunking计算出的公平K值
    INITIAL_K_FOR_SEMANTIC = 68
    # Rerank后最终喂给LLM的数量
    FINAL_K = 5

    for sample in tqdm(test_samples, desc="Processing Samples for Final Test"):
        prompt = sample['Prompt']
        ground_truth_answer = sample['Answer']
        try:
            knowledge_urls = ast.literal_eval(sample['wiki_links'])
        except (ValueError, SyntaxError):
            knowledge_urls = []

        full_context_text = get_content_from_urls(knowledge_urls)
        if not full_context_text: continue

        # --- 执行最强RAG流水线 ---
        
        # 1. 分块 (Semantic Chunking)
        text_chunks = my_rag_functions.chunk_semantic(full_context_text, client=dashscope_client)
        if not text_chunks: continue

        # 2. 向量化
        chunks_embeddings = my_rag_functions.vector_chunks_embedding(text_chunks, client=dashscope_client)
        if not chunks_embeddings: continue

        # 3. 混合检索 (获取Top-68粗排结果)
        dense_results = my_rag_functions.semantic_search(dashscope_client, prompt, text_chunks, chunks_embeddings, k=INITIAL_K_FOR_SEMANTIC)
        sparse_results = my_rag_functions.sparse_search_bm25(prompt, text_chunks, k=INITIAL_K_FOR_SEMANTIC)
        retrieval_results = my_rag_functions.hybrid_search_rrf(prompt, text_chunks, dense_results, sparse_results, k=INITIAL_K_FOR_SEMANTIC)

        # 4. 重排序 (从68个候选中，获取Top-5精排结果)
        # reranked_results = my_rag_functions.rerank_results(prompt, initial_candidates, top_n=FINAL_K)
        
        # 5. 生成答案
        retrieval_context = ' '.join(retrieval_results)
        generated_answer = my_rag_functions.generate_response(dashscope_client, retrieval_context, prompt)
        
        is_correct = (generated_answer.strip().lower() == ground_truth_answer.strip().lower())

        # 记录结果
        all_results.append({
            "question": prompt,
            "standard_answer": ground_truth_answer,
            "generated_answer": generated_answer,
            "is_correct": is_correct
        })

    save_results(all_results, "experiment_best_rag_combination_fixed")
    
    # 计算并打印最终准确率
    correct_count = sum(1 for r in all_results if r['is_correct'])
    total_count = len(all_results)
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f'\n🏆 最强RAG组合在 {total_count} 个样本上最终正确率为: {accuracy:.2%}')
    
    print("✅ 最终测试完成！")

def run_final_experiment(test_samples):
    print('\n' + '='*50)
    print('🚀 开始最终实验：公平条件下对比分块策略')
    print('   - 检索策略: Hybrid Retrieval (固定)')
    print('   - 生成模型: Kimi-k2 (固定)')
    print("="*50)

    chunking_configs = {
        'fix_size_150': {
            'function': lambda text: my_rag_functions.chunk_by_fix_size(text, chunk_size=150),
            'k': 65
        },
        'recursive_150':{
            'function': lambda text: my_rag_functions.chunk_recursively(text, chunk_size=150),
            'k': 65
        },
        'semantic':{
            'function': lambda text: my_rag_functions.chunk_semantic(text, client=dashscope_client),
            'k': 65
        }
    }

    all_results = []

    for sample in tqdm(test_samples, desc="Processing Samples for Final Experiment"):
        prompt = sample['Prompt']
        ground_truth_answer = sample['Answer']
        try:
            knowledge_urls = ast.literal_eval(sample['wiki_links'])
        except (ValueError, SyntaxError):
            knowledge_urls = []

        full_context_text = get_content_from_urls(knowledge_urls)
        if not full_context_text: continue

        sample_results = {"question": prompt, "standard_answer": ground_truth_answer}

        for name, config in chunking_configs.items():
            chunk_func = config["function"]
            top_k = config["k"]

            # 1. 分块
            text_chunks = chunk_func(full_context_text)
            if not text_chunks: continue

            # 2. 向量化
            chunks_embeddings = my_rag_functions.vector_chunks_embedding(text_chunks, client=dashscope_client)
            if not chunks_embeddings or len(text_chunks) != len(chunks_embeddings): continue
            
            # 3. 混合检索 (固定使用Hybrid)
            dense_results = my_rag_functions.semantic_search(dashscope_client, prompt, text_chunks, chunks_embeddings, k=top_k)
            sparse_results = my_rag_functions.sparse_search_bm25(prompt, text_chunks, k=top_k)
            retrieved_results = my_rag_functions.hybrid_search_rrf(prompt, text_chunks, dense_results, sparse_results, k=top_k)
            
            # 4. 生成答案 (固定使用Kimik2)
            retrieved_context = ' '.join(retrieved_results)
            generated_answer = my_rag_functions.generate_response_kimik2(kimik2_client, retrieved_context, prompt)
            
            # 5. 记录结果
            sample_results[f"generated_answer_{name}"] = generated_answer
            sample_results[f"is_correct_{name}"] = (generated_answer.strip().lower() == ground_truth_answer.strip().lower())
        
        all_results.append(sample_results)

    save_results(all_results, "experiment_final_chunking_vs_hybrid")
    print("✅ 最终实验完成！")

def experiment_rag0728(test_samples):
    """
    RAG终极对决实验 (2025/07/28)
    目标：在最强的“Hybrid Search + Rerank”后端加持下，横向对比不同分块策略的最终性能。
    """
    print('\n' + '='*80)
    print('🚀🧪 Hybrid + Rerank vs. 不同分块策略 🧪🚀')
    print('='*80)

    # --- 核心参数定义 ---
    # K_RETRIEVAL: 初步检索捞回的候选文档数量，用于Rerank的输入池
    K_RETRIEVAL = 50  # 一个在性能和算力之间较为均衡的选择
    # N_RERANK: 经过Rerank后，最终喂给LLM的最相关文档数量
    N_RERANK = 5      # 黄金上下文窗口大小

    print(f"[*] 实验配置: 初步检索 Top-K (k_retrieval) = {K_RETRIEVAL}, 最终精排 Top-N (n_rerank) = {N_RERANK}")
    
    chunking_configs = {
        'semantic': {
            'function': lambda text: my_rag_functions.chunk_semantic(text, client=dashscope_client),
        },
        'fix_size_250': {
            'function': lambda text: my_rag_functions.chunk_by_fix_size(text, chunk_size=250),
        },
        'recursive_250': {
            'function': lambda text: my_rag_functions.chunk_recursively(text, chunk_size=250, chunk_overlap=30),
        }
    }

    all_results = []
    # 用于分别统计每个策略的准确率
    accuracy_trackers = {name: {'correct': 0, 'total': 0} for name in chunking_configs.keys()}

    for sample in tqdm(test_samples, desc="[终极对决] 处理样本中..."):
        prompt = sample['Prompt']
        ground_truth_answer = sample['Answer']
        try:
            knowledge_urls = ast.literal_eval(sample['wiki_links'])
        except (ValueError, SyntaxError):
            knowledge_urls = []

        full_context_text = get_content_from_urls(knowledge_urls)
        if not full_context_text:
            continue

        sample_results = {"question": prompt, "standard_answer": ground_truth_answer}

        for name, config in chunking_configs.items():
            print(f"\n--- Running for: {name} ---")
            
            # 1. 分块 (Chunking)
            chunk_func = config["function"]
            text_chunks = chunk_func(full_context_text)
            if not text_chunks:
                print(f"警告: {name} 分块策略未产生任何chunks，跳过此样本。")
                continue
            
            # 2. 向量化 (Embedding)
            chunks_embeddings = my_rag_functions.vector_chunks_embedding(text_chunks, client=dashscope_client)
            if not chunks_embeddings or len(text_chunks) != len(chunks_embeddings):
                print(f"警告: 向量化失败或数量不匹配，跳过此样本。")
                continue

            # 3. 初步检索 (Retrieval - Hybrid Search)
            # 目标：广撒网，捞回 K_RETRIEVAL 个候选块
            print(f"   -> (1/3) 混合检索 (Hybrid Search) k={K_RETRIEVAL}...")
            dense_results = my_rag_functions.semantic_search(dashscope_client, prompt, text_chunks, chunks_embeddings, k=K_RETRIEVAL)
            sparse_results = my_rag_functions.sparse_search_bm25(prompt, text_chunks, k=K_RETRIEVAL)
            initial_candidates = my_rag_functions.hybrid_search_rrf(prompt, text_chunks, dense_results, sparse_results, k=K_RETRIEVAL)
            
            # 4. 精确重排 (Rerank)
            # 目标：精挑细选，从候选池中选出最相关的 N_RERANK 个
            print(f"   -> (2/3) 精确重排 (Rerank) n={N_RERANK}...")
            final_chunks = my_rag_functions.rerank_results(prompt, initial_candidates, top_n=N_RERANK)
            
            # 5. 答案生成 (Generation)
            # 目标：基于黄金上下文，生成最终答案
            print(f"   -> (3/3) 生成答案 (Generation)...")
            retrieved_context = ' '.join(final_chunks)
            generated_answer = my_rag_functions.generate_response_kimik2(kimik2_client, retrieved_context, prompt)
            
            # 6. 记录结果
            is_correct = (str(generated_answer).strip().lower() == str(ground_truth_answer).strip().lower())
            sample_results[f"generated_answer_{name}"] = generated_answer
            sample_results[f"is_correct_{name}"] = is_correct
            
            accuracy_trackers[name]['correct'] += 1 if is_correct else 0
            accuracy_trackers[name]['total'] += 1
            
        all_results.append(sample_results)

    # 保存详细的JSON结果
    save_results(all_results, "experiment_rag0728_ultimate_showdown")
    
    # 打印最终的准确率对比报告
    print('\n' + '='*80)
    print("📊 终极对决实验结果报告 📊")
    print('='*80)
    for name, stats in accuracy_trackers.items():
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            print(f"🏆 策略: {name:<20} | 准确率: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
        else:
            print(f"⚠️ 策略: {name:<20} | 未运行或无有效样本。")
    print('='*80)
    print("✅ 所有实验已完成！")



if __name__ == '__main__':
    print("正在加载FRAMES数据集...")
    frames_dataset = load_dataset('google/frames-benchmark')
    test_set = frames_dataset['test']

    # 建议测试样本数量不要太多，因为Rerank会比较耗时
    shuffled_test_set = test_set.shuffle(seed=42)
    test_samples = list(shuffled_test_set.select(range(20))) 
    print(f"已随机挑选 {len(test_samples)} 个样本进行最终对决测试。")

    # --- 调用你的新实验函数 ---
    experiment_rag0728(test_samples)

    print("\n🎉 所有实验已完成！")

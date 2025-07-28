import fitz
import os
import numpy as np
import json
from openai import OpenAI
import openai
from dotenv import load_dotenv
import pymupdf
from google import genai
from google.genai import types
from openai import OpenAI
import requests
import json
from mistralai import Mistral
import io
import base64
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi 
from sentence_transformers import CrossEncoder
import nltk
import time
import torch
nltk.download('punkt')

if torch.cuda.is_available():
    device = 'cuda:3'
    print(f'检测到GPT,将使用{device}进行rerank计算')
else:
    device = 'cpu'
    print('未检测到GPU,将使用CPU进行rerank计算')

try:
    reranker_model = CrossEncoder('BAAI/bge-reranker-large', device=device)
    print('rerank模型加载成功')
except Exception as e:
    print(f'加载rerank失败,请检查网络连接或模型路径:{e}')
    reranker_model = None


Dashscope_API_KEY = 'sk-fb8191fb105b439d9ffd2880a9d9be7c'
Mistral_API_KEY = 'NMRPcolmwB7PEegVbzi6TId4IUVCfCFB'
deepseek_key = "sk-09IfmX7mg7MJQg0U0OOM8AVXBrGC7Um887xvdb4H3Tbn16sQ"
Kimi_key = 'sk-vqaNEYl0Z3vf8GITnaebDySglXyuReqPLzGDlISr9qyKYEYo'

"""
RAG总览
1. 提取PDF的文本(PyMuPDF)
2. 将文本分块
3. 加载OpenAI客户端
4. 编码embeddings.
5. 
"""

# 1. Extract text from pdf
def extract_text(pdf):
    """
    Args:
    pdf_path(str): pdf link

    Returns:
    str: pdf text
    """
    full_text = ""
    with pymupdf.open(stream=pdf, filetype='pdf') as pdf:
        for page in pdf:
            text = page.get_text()
            full_text += text
    return full_text

# 1.2 OCRFlux-3B 替代pymupdf

API_UPLOAD_URL = 'http://172.16.120.14:8000/parse-pdf/'

def extract_text_OCRFlux(pdf_bytes: bytes, filename: str):

    print(f'正在通过API请求处理文件:{filename}')

    try:
        files_payload = {
            'file': (filename, pdf_bytes, 'application/pdf')
        }

        response = requests.post(API_UPLOAD_URL, files=files_payload, timeout=300)
        response.raise_for_status

        data = response.json()

        return data.get("markdown_content", '')
    except requests.exceptions.RequestException as e:
        print(f'调用API时发生错误:{e}')
        return ''
    except json.JSONDecodeError:
        print(f'无法解析服务器返回的JSON响应:{response.text}')
        return ''
    
def extract_text_MistralOCR(pdf_byte: bytes, client: Mistral):
    """
    Args:
    pdf_path(str): the path of pdf

    str: pdf_text
    """
    try:
        base64_pdf = base64.b64encode(pdf_byte).decode('utf-8')

        ocr_response = client.ocr.process(
            model='mistral-ocr-latest',
            document={
                'type': 'document_url',
                'document_url': f'data:application/pdf;base64,{base64_pdf}'
            }
        )
        full_text = ''
        if ocr_response.pages:
                full_text += "\n\n".join([page.markdown for page in ocr_response.pages])
        if not full_text:
            print('警告,OCR执行成功,但未提取到任何内容')
        return full_text.strip()
    
    except Exception as e:
        print(f'调用MistralOCR API时发生错误: {e}')
        return ''
        


# 2. Divide the text into small chunks
def chunk_by_fix_size(pdf_text, chunk_size: int = 150):
    """
    Args:
    pdf_text: the text from pdf

    Return:
    List[str]: Every chunks from pdf
    """
    pdf_chunks = []
    for  i in range(0, len(pdf_text), chunk_size):
        chunk = pdf_text[i : i + chunk_size]
        pdf_chunks.append(chunk)
    return pdf_chunks

def chunk_by_character(text: str, chunk_size: int = 150, chunk_overlap: int = 20):
    """
    Chunk level1: 按字符分块,统一使用Langchain分块器
    Args:
    text(str): 输入的长文本
    chunk_size(int): 每个块的最大字符数
    chunk_overlap(int): 块之间重叠的字符数
    
    Return:
    List[str]: 分割后的文本块列表
    """
    text_splitter = CharacterTextSplitter(
        separator='\n\n',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function = len,
    )
    return text_splitter.split_text(text)

def chunk_recursively(text: str, chunk_size: int = 150, chunk_overlap: int = 20, **kwargs):
    """
    Chunk Level 2 : 递归字符分块,推荐通用的方法。
    Args:
    text(str): 输入的长文本
    chunk_size(int): 每个块的最大字符数
    chunk_overlap(int): 块之间的重叠字符数
    **kwargs: 可以传入自定义的separtors的列表等参数

    Returns:
    list[str]: 分割后的文本块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        **kwargs
    )
    return text_splitter.split_text(text)

def chunk_by_markdown(text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Chunk Level 3: Markdown 结构化分块
    Args:
    text(str): 输入的Markdown文本
    chunk_size(int): 每个块的最大字符数
    chunk_overlap(int): 块之间重叠字符数

    Return:
    List[str]: 分割后的文本块列表
    """
    markdown_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return markdown_splitter.split_text(text)


def chunk_semantic(text: str,client: OpenAI, threshold_percentile: float = 0.95):
    """
    Chunk Level 4:语义分块,使用阿里云API
    Args:
    text(str): 输入的长文本
    dashscope_api_key(str): 需要的Openai的API key来初始化embedding 模型
    breakpoint_threshold_type(str): 切断割点的类型
    
    Return:
    List[str]: 分割后的文本块列表   
    """

    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []
    
    embeddings = vector_chunks_embedding(sentences, client=client)

    similarities = [cosine_similar(embeddings[i], embeddings[i+1]) for i in range(len(embeddings) - 1)]

    if not similarities:
        return sentences
    threhold = np.percentile(similarities, threshold_percentile * 100)

    chunks = []
    current_chunk = [sentences[0]]
    for i, similarity in enumerate(similarities):
        if similarity < threhold:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

    
def split_chunks(pdf_chunks, batch_size=9):
    return [pdf_chunks[i : i + batch_size] for i in range(0, len(pdf_chunks), batch_size)]


# 4. Text block create embed
def vector_chunks_embedding(chunks, client: genai.Client):
    """
    Args:
    chunks(List[str]): The divided chunks which in list

    return:
    List(vector)
    """ 
    all_embedding = []
    batch_chunks = split_chunks(chunks)

    for i, batch in enumerate(batch_chunks):
        response = client.embeddings.create(
        model = "text-embedding-v4",
        input = batch,
        dimensions=1024,
        encoding_format="float"
        )     
        
        batch_embeddings = [record.embedding for record in response.data]
        all_embedding.extend(batch_embeddings)
    return all_embedding
        

    
# 5. semantic_search
def cosine_similar(vector1, vector2):
    """
    Args:
    vector_chunks: The different vector chunks value in the chunks.

    return:
    List[int]

    function: Sort
    """
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

# 这个就是dense search，
def semantic_search(client: genai.Client, query, pdf_chunks, chunks_embeddings, k=5):
    """
    Args:
    chunks_embedding(List[dict]): the embeddings of different chunks
    pdf_chunks(List[str]): the text from pdf_chunks
    query(str): the prompt from someone
    k(int): 


    return:
    List[int]

    Function: 对提取到的文本进行按照prompt,根据不同chunk的embedding的值进行查找。
    """
    query_embedding_response = client.embeddings.create(
        model = "text-embedding-v4",
        input  = query,
        dimensions=1024, 
        encoding_format="float"
        )
    query_embedding = query_embedding_response.data[0].embedding
    similarity_scores = []
    for i, chunks_embedding in enumerate(chunks_embeddings):
        score = cosine_similar(np.array(query_embedding), np.array(chunks_embedding))
        similarity_scores.append((i, score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]

    return [pdf_chunks[index] for index in top_indices]

def sparse_search_bm25(query: str, pdf_chunks: list, k: int = 5):
    """
    使用BM25算法进行稀疏搜索:
    Args:
        query(str): 用户查询
        pdf_chunks(list): 所有文本块列表
        k(int): 需要返回Top-K个结果

    Returns:
        list: Top-K个最相关的文本块
    """
    print('=== Running Sparse Search(BM25)...---')
    tokenized_corpus = [chunk.split(' ') for chunk in pdf_chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split(' ')

    top_k_chunks = bm25.get_top_n(tokenized_query, pdf_chunks, n=k)

    return top_k_chunks

def hybrid_search_rrf(query: str, pdf_chunks: list, dense_results: list,
        sparse_results: list, k: int = 5, rrf_k: int = 60):
    """
    使用倒序排序融合(RRF)算法来合并稀疏和密集搜索的结果
    Args:
        query(str): 用户查询
        pdf_chunks(list): 原始的所有文本块列表
        dense_results(list): 密集检索返回的有序文本块列表
        sparse_results(list): 稀疏检索返回的有序文本块列表    
        k(int): 最终需要返回的Top-K个结果
        rrf_k(int): RRF算法的平滑参数
    
    Return:
        list: 经过RRF算法融合后Top-K最相关的文本块
    """
    print('--- Running Hybrid Search (RRF) ...')
    rrf_scores = {}

    for rank, chunk in enumerate(dense_results):
        if chunk not in rrf_scores:
            rrf_scores[chunk] = 0
        rrf_scores[chunk] += 1 / (rrf_k + rank + 1) # rank from 0 to start.
    
    for rank, chunk in enumerate(sparse_results):
        if chunk not in rrf_scores:
            rrf_scores[chunk] = 0
        rrf_scores[chunk] += 1 / (rrf_k + rank + 1)

    sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    final_top_k_chunks = [chunk for chunk, score in sorted_chunks[:k]]

    return final_top_k_chunks

# 5.5 Rerank
def rerank_results(query: str, retrieval_chunks: list, top_n: int = 5):
    """
    使用Cross-Encoder模型对初步检索出的文本块进行重排序
    Args:
        query(str): 用户查询
        retrieval_chunks(list): 经过初步检索得到的文本块列表
        top_n(int): 最终需要返回的经过精排后的Top-N个结果
    
    Returns:
        list: 经过重排序后,得分最高的Top-K个文本块
    """
    if reranker_model is None:
        print('Rerank模型未加载,跳过Rerank')
        return retrieval_chunks[:top_n]
    
    pairs = [(query, chunk) for chunk in retrieval_chunks]
    scores = reranker_model.predict(pairs)

    chunk_with_scores = list(zip(retrieval_chunks, scores))
    sorted_chunks = sorted(chunk_with_scores, key=lambda item: item[1], reverse=True)
    final_top_n_chunks = [chunk for chunk, score in sorted_chunks[:top_n]]

    return final_top_n_chunks


# 6 Finally answer the question based on rag=text
def generate_response(client: genai.Client, text ,query):
    prompt = f"""你是一个只输出最终答案的AI助手。严格根据给定的上下文回答问题。上下文：{text}
                重要：你的回答必须只包含最终答案本身，不要添加任何解释、推理过程或“答案是：”等多余的文字。
                如果无法从上下文中得出答案，请只回复：“信息不足，无法回答”。""" 
    response = client.chat.completions.create(
        model='qwen-plus',
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
        ]
    )
    return response.choices[0].message.content

# 6. DeepSeek answer question
def generate_response_deepseek(client: OpenAI, text, query):
    prompt = f"""你是一个只输出最终答案的AI助手。严格根据给定的上下文回答问题。上下文：{text}
                重要：你的回答必须只包含最终答案本身，不要添加任何解释、推理过程或“答案是：”等多余的文字。
                如果无法从上下文中得出答案，请只回复：“信息不足，无法回答”。"""    
    completion = client.chat.completions.create(
        model='deepseek-r1',
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
        ]
    )
    return completion.choices[0].message.content

def generate_response_kimik2(client: OpenAI, text, query, max_retries=3):
    """
    使用Kimi-k2模型生成答案，并增加了对限流的重试逻辑。
    """
    prompt = f"""你是一个只输出最终答案的AI助手。严格根据给定的上下文回答问题。上下文：{text}
                重要：你的回答必须只包含最终答案本身，不要添加任何解释、推理过程或“答案是：”等多余的文字。
                如果无法从上下文中得出答案，请只回复：“信息不足，无法回答”。"""
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                # 注意：Kimi的兼容模型名通常是 moonshot-v1-8k 或类似
                model="moonshot-v1-8k", 
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': query}
                ]
            )
            return completion.choices[0].message.content
        
        except openai.RateLimitError as e:
            print(f"Kimi API限流。尝试次数 {attempt + 1}/{max_retries}。")
            if attempt < max_retries - 1:
                wait_time = 21 # 针对3 RPM的限制，等待21秒是安全的
                print(f"将在 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("已达到最大重试次数，放弃此请求。")
                return "[错误: Kimi API超出最大重试次数]"
        
        except Exception as e:
            print(f"调用Kimi API时发生未知错误: {e}")
            return f"[错误: API调用失败 - {e}]"
            
    return "[错误: 所有重试均失败]"
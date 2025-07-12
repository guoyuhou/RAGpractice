import fitz
import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv
import pymupdf
from google import genai
from google.genai import types
from openai import OpenAI
import requests
import json


"""
RAD总览
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

API_URL = 'http://172.16.120.14:8000/parse-pdf/'

def extract_text_OCRFlux(pdf_path):
    pay_load = {'pdf_path': pdf_path}

    print(f'正在通过API请求处理文件:{pdf_path}')

    try:
        response = requests.post(API_URL, json=pay_load, timeout=300)

        response.raise_for_status

        data = response.json()

        return data.get("markdown_content", '')
    except requests.exceptions.RequestException as e:
        print(f'调用API时发生错误:{e}')
    except json.JSONDecodeError:
        print(f'无法解析服务器返回的JSON响应:{response.text}')
        return ''

# 2. Divide the text into small chunks
def divide_pdf_to_chunks(pdf_text):
    """
    Args:
    pdf_text: the text from pdf

    Return:
    List[str]: Every chunks from pdf
    """
    max_length = 2000
    pdf_chunks = []
    for i in range(0, len(pdf_text), max_length):
        chunk = pdf_text[i : i + max_length]
        pdf_chunks.append(chunk)
    return pdf_chunks
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

# 6 Finally answer the question based on rag=text
def generate_response(client: genai.Client, text ,query):
    prompt = f'{"你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"}, 上下文：{text}'
    response = client.chat.completions.create(
        model='qwen-plus',
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
        ]
    )
    return response.choices[0].message.content


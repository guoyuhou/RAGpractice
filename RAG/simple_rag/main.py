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

# 4. Text block create embed
def vector_chunks_embedding(chunks, client: genai.Client):
    """
    Args:
    chunks(List[str]): The divided chunks which in list

    return:
    List(vector)
    """ 
    response = client.embeddings.create(
        model = "text-embedding-v4",
        input = chunks,
        dimensions=1024,
        encoding_format="float"
        ) 
    embeddings = [record.embedding for record in response.data]

    return embeddings
        

    
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
    query_embedding = query_embedding_response.embeddings.data[0].embedding
    similarity_scores = []
    for i, chunks_embedding in enumerate(chunks_embeddings):
        score = cosine_similar(np.array(query_embedding), np.array(chunks_embedding))
        similarity_scores.append((i, score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]

    return [pdf_chunks[index] for index in top_indices]


import fitz
import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv


"""
RAD总览
1. 提取PDF的文本(PyMuPDF)
2. 将文本分块
3. 加载OpenAI客户端
4. 编码embeddings.
5. 
"""

# 1. Extract text from pdf
def extract_text(pdf_path):
    """
    Args:
    pdf_path(str): pdf link

    Returns:
    str: pdf text
    """
    pdf_text = 



# 2. Divide the text into small chunks
def divide_pdf_to_chunks(pdf_text):
    """
    Args:
    pdf_text: the text from pdf

    Return:
    List[str]: Every chunks from pdf
    """
    return pdf_chunks

# 3. Load the OpenAI client.
def 
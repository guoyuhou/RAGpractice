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
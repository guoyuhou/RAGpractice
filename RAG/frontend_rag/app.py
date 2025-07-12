# Strealit app
import streamlit as st
import sys
import os
import fitz
import numpy as np
import json
from dotenv import load_dotenv
import pymupdf
from google import genai
from google.genai import types
from openai import OpenAI
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from simple_rag import main


st.title("RAG")
st.subheader('刘曜畅2025.7.6')



api_key_str = st.text_input("请输入你的API_key", type='password')
prompt = st.text_input("请输入你的问题", type='default')
upload_file = st.file_uploader('请上传你的PDF文件.', type='pdf')

query = prompt


if st.button('处理PDF并进行RAG流程'):
    if not api_key_str:
        st.error("请输入api密钥")
    elif not upload_file:
        st.error("请上传文件")
    elif not prompt:
        st.error('请输入问题')
    else:
        with st.spinner("Please wait......"):
            client = OpenAI(api_key=api_key_str,
                                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
            pdf_bytes = upload_file.getvalue()  
            pdf_filename = upload_file.name

            st.write("The answer is :")
            pdf_text = main.extract_text_OCRFlux(pdf_bytes, pdf_filename)
            pdf_chunks = main.divide_pdf_to_chunks(pdf_text)
            chunks_embedding = main.vector_chunks_embedding(pdf_chunks, client)
            ans = main.semantic_search(client, query, pdf_chunks, chunks_embedding)

            st.write(ans)

            st.write('Answer:')
            ai_answer = main.generate_response(client, ans, query)
            st.write(ai_answer)



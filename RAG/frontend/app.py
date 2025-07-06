import streamlit as st
import sys
import os
from google import genai
from google.genai import types

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from simple_rag import main

st.title("RAG")
st.subheader('刘曜畅2025.7.6')


if 'pdf_processed_id' not in st.session_state:
    st.session_state.pdf_processed_id = ""


api_key = st.text_input("请输入你的API_key", type='password')
prompt = st.text_input("请输入你的问题", type='default')
upload_file = st.file_uploader('请上传你的PDF文件.', type='pdf')

query = prompt

if st.button('处理PDF并进行RAG流程'):
    if not api_key:
        st.error("请输入api密钥")
    elif not upload_file:
        st.error("请上传文件")
    elif not prompt:
        st.error('请输入问题')
    else:
        with st.spinner("正在处理，请稍候......"):
            client = genai.Client(api_key=api_key)
 
            st.info("检测到新文件，正在进行首次分析和向量化...")
                
            pdf_bytes = upload_file.getvalue()
            pdf_text = main.extract_text(pdf_bytes)
            pdf_chunks = main.divide_pdf_to_chunks(pdf_text)
                
            embedding_response = main.vector_chunks_embedding(client, pdf_chunks)
                    
            st.session_state.pdf_chunks = pdf_chunks
            st.session_state.chunks_embedding = embedding_response.embeddings

            st.success("文档分析完成并已缓存！")

            # 4. 【核心语义搜索】现在，无论是新分析的还是从缓存中读取的，我们都可以执行搜索
            # 这个操作相对轻量，只消耗一次查询的API配额
            st.info("正在使用缓存好的数据进行语义搜索...")
            ans = main.semantic_search(
                client, 
                query, 
                st.session_state.pdf_chunks, 
                st.session_state.chunks_embedding
            )

            # 5. 显示最终答案
            st.write("The answer is :")
            st.write(ans)
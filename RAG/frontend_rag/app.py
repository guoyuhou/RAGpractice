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
from mistralai import Mistral

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from simple_rag import main

def initialize_app():
    st.title("Organoid Paper RAG System")
    st.caption('A functionally structured RAG application')

    with st.sidebar:
        st.header('API Configuration')
        dashscope_key = st.text_input('Please input Dashscope/OpenAI API key', type='password')
        Mistral_key = st.text_input('Please input Mistral api key', type='password')
        deepseek_key = "sk-09IfmX7mg7MJQg0U0OOM8AVXBrGC7Um887xvdb4H3Tbn16sQ"

    return dashscope_key, Mistral_key, deepseek_key

def get_user_input():
    """Get the user's question and uploaded file"""
    prompt = st.text_input('Enter your question here:', placeholder='eg: what is the main idea of this paper')
    upload_file = st.file_uploader('Up load a PDF paper on organoids or others')
    run_button = st.button('Analyze and Answer')
    return prompt, upload_file, run_button

def run_rag_pipeline(dashscope_key: str, mistral_key: str, deepseek_key : str, prompt: str, upload_file):
    """
    Excute the entire RAG pipeline from PDF to final answer
    This function is focused on data processing, not UI
    """
    with st.spinner('Processing in progress... This may take a moment.'):
        # 1. Initialize clients
        dashscope_client = OpenAI(api_key=dashscope_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        mistral_client = Mistral(api_key=mistral_key)
        deepseek_client = OpenAI(api_key=deepseek_key, base_url="https://api.lkeap.cloud.tencent.com/v1")

        # 2. Extract text
        pdf_bytes = upload_file.getvalue()
        pdf_text = main.extract_text_MistralOCR(pdf_bytes, mistral_client)
        if not pdf_text:
            st.error('Text extraction failed! Please check your Mistral API')
            return None
        
        # 3. Chunking, embedding and semantic search
        pdf_chunks = main.chunk_by_fix_size(pdf_text)
        chunks_embedding = main.vector_chunks_embedding(pdf_chunks, dashscope_client)
        relevant_chunks = main.semantic_search(dashscope_client, prompt, pdf_chunks, chunks_embedding)

        # 4. Generate final response
        context_for_generate = '\n\n---\n\n'.join(relevant_chunks)
        ai_answer = main.generate_response_deepseek(deepseek_client, context_for_generate, prompt)

        return {
            'pdf_text': pdf_text,
            'pdf_chunks': pdf_chunks,
            'relevant_chunks': relevant_chunks,
            'ai_answer': ai_answer
        }

# Display function

def display_results(results: dict):
    if not results:
        return
    
    st.success('Analyasis complete!')
    st.markdown('---')
    
    # Display full extracted markdown
    with st.expander('View Full Extracted Markdown Text'):
        st.markdown(results['pdf_text'])

    # Display First 3 Chunks:
    with st.expander('View First 3 Extracted Markdown Text'):
        if results['pdf_text']:
            for i, chunk in enumerate(results['pdf_text'][:3]):
                st.text_area(f'Chunk {i+1}', chunk, height=150, key=f'chunk_display_{i}')
        else:
            st.write('No Text Chunks Were Generated')

    # Display finaly answer and sources
    st.subheader('AI generated Final Answer')
    st.markdown(results['ai_answer'])
    with st.expander('View AI cited source Passages'):
        for i, chunk in enumerate(results['relevant_chunks']):
            st.text_area(f'Cited Passage {i+1}', chunk, height=150, key=f'relevant_chunk_{i}')

def app():
    dashscope_key, mistral_key, deepseek_key = initialize_app()
    prompt, upload_file ,run_button = get_user_input()

    if run_button:
        if not(dashscope_key and mistral_key and upload_file and prompt):
            st.warning('Please ensure all API keys are provided, a file is uploaded, and a question is entered')
        else:
            st.session_state.results = run_rag_pipeline(dashscope_key, mistral_key, deepseek_key, prompt, upload_file)

    if 'results' in st.session_state and st.session_state.results:
        display_results(st.session_state.results)


if __name__ == "__main__":
    app()
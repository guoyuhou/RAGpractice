# Strealit app
import streamlit as st

st.title("RAG")
st.subheader('刘曜畅2025.7.6')

if "rag_chain" not in st.session_state:
    st.session_state = None

api_key = st.text_input("请输入你的API_key", type='password')
upload_file = st.file_uploader('请上传你的PDF文件.', type='pdf')

if st.button('处理PDF并进行RAG流程'):
    if not api_key:
        st.error("请输入api密钥")
    elif not upload_file:
        st.error("请上传文件")
    else:
        st.spinner("Please wait......")
    

    

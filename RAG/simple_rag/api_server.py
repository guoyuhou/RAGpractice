# OCRFlux-3B api server

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from vllm import LLM
from ocrflux.inference import parse
from tqdm import tqdm

llm_model = None

# 模型的加载与处理
def initialize_model(model_path: str, gpu_util_rate: float = 0.8):
    print("=" * 50)
    print(f'正在加载OCRFlux-3B模型:{model_path}')
    print(f'设置GPU的显存利用率为:{gpu_util_rate}')

    model = LLM(
        model = model_path,
        gpu_memory_utilization = gpu_util_rate,
        max_model_len = 8192,
        enforce_eager = True
    )

    print('模型加载成功')
    print('=' * 50)
    return model

def process_single_pdf(model: LLM, pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f'文件未找到:{pdf_path}')
    
    try:
        result = parse(model, pdf_path)
        if result and 'document_text' in result:
            return result['document_text']
        else:
            print(f'warning: file{os.path.basename(pdf_path)} 解洗失败')
            return ''
    except Exception as e:
        print(f'警告！处理文件{os.path.basename(pdf_path)}发生错误，:{e}')
        raise HTTPException(status_code=500, detail=f'处理PDF时发生错误:{str(e)}')
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_model

    model_dir = os.environ.get('MODEL_DIR', '/home/lh/OCRFlux/OCRFlux-3B')
    if not os.path.exists(model_dir):
        raise ValueError(f'模型目录不存在:{model_dir}, 请设置model dir')
    
    llm_model = initialize_model(model_dir)
    yield
    print('正在关闭应用')
    global llm_model
    del llm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('程序已关闭')

app = FastAPI(lifespan=lifespan)

class ParseRequest(BaseModel):
    pdf_path: str

@app.get('/')
def read_root():
    return {'status': 'OCRFlux API is running'}

@app.post('/parse-pdf/')
async def api_parse_pdf(request: ParseRequest):
    "接收一个包含PDF文件路径的POST请求,使用vllm模型进行解析,并返回markdown文档。"
    global llm_model
    if llm_model is None:
        raise HTTPException(status_code=503, detail='模型正在加载或加载失败，请稍后再试。')
    
    print(f'收到解析请求,文件路径:{request.pdf_path}')

    markdown_content  = process_single_pdf(llm_model, request.pdf_path)

    return {
        'source_path': request.pdf_path,
        'markdown_content' : markdown_content
    }

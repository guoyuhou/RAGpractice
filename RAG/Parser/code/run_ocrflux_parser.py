# 文件名: run_ocrflux_parser.py

import os
import argparse
from tqdm import tqdm
from vllm import LLM
from ocrflux.inference import parse
import time

def initialize_model(model_path: str, gpu_util_rate: float = 0.8):
    """加载OCRFlux模型到GPU内存中。"""
    print(f"正在加载OCRFlux模型，路径: {model_path}")
    print(f"设置GPU显存使用率为: {gpu_util_rate * 100}%")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_util_rate,
        max_model_len=8192
    )
    print("模型加载成功！")
    return llm

def process_single_pdf(llm_model: LLM, pdf_path: str) -> str:
    """使用加载好的模型处理单个PDF文件。"""
    try:
        result = parse(llm_model, pdf_path)
        if result and 'document_text' in result:
            return result['document_text']
        else:
            # 使用tqdm的write方法打印，避免与进度条冲突
            tqdm.write(f"文件 {os.path.basename(pdf_path)} 解析失败，返回空。")
            return ""
    except Exception as e:
        tqdm.write(f"处理文件 {os.path.basename(pdf_path)} 时发生严重错误: {e}")
        return ""

def main(model_dir: str, input_dir: str, output_dir: str):
    """主处理流程。"""
    llm = initialize_model(model_dir)

    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    print(f'共找到 {len(pdf_files)} 个文件，开始处理......')

    for filename in tqdm(pdf_files, desc="正在用OCRFlux解析PDF"):
        output_filename = os.path.splitext(filename)[0] + '.md'
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            continue

        pdf_path = os.path.join(input_dir, filename)
        markdown_content = process_single_pdf(llm, pdf_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        time.sleep(0.1)

    print(f"处理完成！所有结果已保存到: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用OCRFlux的直接API调用功能，批量处理PDF文件。")
    parser.add_argument('--model_dir', type=str, required=True, help="存放OCRFlux-3B模型文件的目录路径。")
    parser.add_argument('--input_dir', type=str, required=True, help="存放PDF文件的输入目录路径。")
    parser.add_argument('--output_dir', type=str, required=True, help="用于保存Markdown结果的输出目录路径。")
    args = parser.parse_args()

    main(args.model_dir, args.input_dir, args.output_dir)

    
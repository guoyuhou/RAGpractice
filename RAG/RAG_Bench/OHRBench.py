import os
import json
from tqdm import tqdm
from mistralai import Mistral
import base64

# --- 配置您的API密钥 ---
MISTRAL_API_KEY = "NMRPcolmwB7PEegVbzi6TId4IUVCfCFB" # 您的Mistral密钥

def ocr_pdf_per_page(pdf_bytes: bytes, client: Mistral) -> list:
    """
    这是一个适配版本，调用Mistral OCR并按页返回文本列表。
    """
    # ... 此函数内容与之前版本相同，无需修改 ...
    try:
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        ocr_response = client.ocr.process(
            model='mistral-ocr-latest',
            document={'type': 'document_url', 'document_url': f'data:application/pdf;base64,{base64_pdf}'}
        )
        if ocr_response.pages:
            return [page.markdown for page in ocr_response.pages]
        else:
            return []
    except Exception as e:
        print(f"调用Mistral OCR API时发生错误: {e}")
        return []

# --- 【关键修改处】 ---
def process_all_pdfs_by_domain(input_dir, output_dir, client: Mistral):
    """
    【新版函数】按领域处理输入目录中的所有PDF，并按OHRBench格式生成对应的JSON文件。
    """
    print(f"开始从根目录 '{input_dir}' 处理...")
    
    # 获取所有领域的子文件夹
    domain_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    if not domain_folders:
        print(f"错误: 在 '{input_dir}' 中未找到任何领域子文件夹。")
        return

    # 外层循环：遍历每个领域子文件夹
    for domain in tqdm(domain_folders, desc="Processing Domains"):
        domain_input_path = os.path.join(input_dir, domain)
        domain_output_path = os.path.join(output_dir, domain)
        
        # 根据领域名创建对应的输出子文件夹
        os.makedirs(domain_output_path, exist_ok=True)
        
        # 获取该领域内的所有PDF文件
        pdf_files = [f for f in os.listdir(domain_input_path) if f.lower().endswith('.pdf')]
        
        # 内层循环：处理该领域内的所有PDF文件
        for pdf_filename in tqdm(pdf_files, desc=f"  -> Processing PDFs in [{domain}]", leave=False):
            input_pdf_path = os.path.join(domain_input_path, pdf_filename)
            
            with open(input_pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            page_texts = ocr_pdf_per_page(pdf_bytes, client)
            
            if not page_texts:
                continue
            
            formatted_data = [{"page_idx": i, "text": text} for i, text in enumerate(page_texts)]
            
            output_json_filename = os.path.splitext(pdf_filename)[0] + '.json'
            output_json_path = os.path.join(domain_output_path, output_json_filename)
            
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, indent=4, ensure_ascii=False)

    print(f"\n处理完成！所有结果已按领域保存至 '{output_dir}' 目录。")


if __name__ == "__main__":
    # --- 用户配置区 ---
    # 1. 指定存放原始PDF的根目录 (该目录下应包含7个领域子文件夹)
    PDF_INPUT_DIRECTORY = "D:/Summer_Learning/OHR-Bench/data/pdfs" 
    
    # 2. 指定存放OCR结果的根目录
    OCR_OUTPUT_DIRECTORY = "D:/Summer_Learning/OHR-Bench/data/retrieval_base/my_mistral_results"

    # --- 执行 ---
    print("正在初始化Mistral客户端...")
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    
    if not os.path.isdir(PDF_INPUT_DIRECTORY):
        print(f"错误：输入目录 '{PDF_INPUT_DIRECTORY}' 不存在。")
        print("请先创建该目录，并将OHRBench的7个领域PDF文件夹放入其中。")
    else:
        process_all_pdfs_by_domain(PDF_INPUT_DIRECTORY, OCR_OUTPUT_DIRECTORY, mistral_client)
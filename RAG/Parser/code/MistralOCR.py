# MistralOCR

# APIkey:NMRPcolmwB7PEegVbzi6TId4IUVCfCFB

import os
from mistralai import Mistral
import base64
from tqdm import tqdm
import argparse

api_key = "NMRPcolmwB7PEegVbzi6TId4IUVCfCFB"
client = Mistral(api_key=api_key)
def encode_pdf_to_url(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            return f"data:application/pdf;base64,{base64.b64encode(pdf_file.read()).decode('utf-8')}"
    except FileNotFoundError:
        print(f'Error: the file{pdf_path} was not found.')
        return None
    except Exception as e:
        print(f'Error: {e}')
        return None

def extract_text(base64_pdf_path):
    ocr_response = client.ocr.process(
    model='mistral-ocr-latest',
    document={
        "type" : 'document_url',
        "document_url" : base64_pdf_path
    },
    include_image_base64=True
    )
    full_text = '\n\n'.join([page.markdown for page in ocr_response.pages])
    return full_text

def main(input_dir, output_dir):

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    print(f'共找到{len(pdf_files)}个文件，正在处理......')

    progress_bar = tqdm(pdf_files, desc='正在用MistralOCR解析PDF')


    for filename in tqdm(pdf_files, desc='正在解析PDF'):

        output_filename = os.path.splitext(filename)[0] + '.md'
        output_path = os.path.join(output_dir, output_filename)
        if os.path.exists(output_path):
            print(f"文件 {output_filename} 已存在，跳过。") 
            continue 
        try:
            pdf_path = os.path.join(input_dir, filename)
            base64_pdf_path = encode_pdf_to_url(pdf_path)
            if base64_pdf_path:
                text_content = extract_text(base64_pdf_path)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
        
        except Exception as e:
            progress_bar.write(f"\n处理文件 {filename} 时发生错误，已跳过。错误信息: {e}")
            continue

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用PyMupdf测评')
    parser.add_argument('--input_dir', type=str, required=True, help='存放PDF的保存路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出markdown的路径')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)

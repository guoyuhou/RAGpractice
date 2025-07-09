import os
import pymupdf
import argparse
from tqdm import tqdm

def extract_text(pdf_path):
    full_text = ''
    try:
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text()

    except Exception as e:
        print(f'处理文件{pdf_path}时出错{e}')
        return ''
    return full_text 

def main(input_dir, output_dir):

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    print(f'共找到{len(pdf_files)}个文件，正在处理......')

    for filename in tqdm(pdf_files, desc='正在解析PDF'):
        pdf_path = os.path.join(input_dir, filename)

        text_content = extract_text(pdf_path)

        output_filename = os.path.splitext(filename)[0] + '.md'
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用PyMupdf测评')
    parser.add_argument('--input_dir', type=str, required=True, help='存放PDF的保存路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出markdown的路径')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
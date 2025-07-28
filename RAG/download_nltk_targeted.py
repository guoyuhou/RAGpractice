import nltk
import os

download_dir = r"D:\Summer_Learning\rag_venv\nltk_data"
os.makedirs(download_dir, exist_ok=True)

# 将下载目标从 'punkt' 改为 'punkt_tab'
nltk.download('punkt_tab', download_dir=download_dir)

print(f"NLTK 'punkt_tab' package successfully downloaded to: {download_dir}")
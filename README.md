# RAGpractice
This repository was created to rewrite RAG 

## SimpleRAG实现流程：
- 1. 利用PyMupdf(fitz)库进行提取文本，并对文本进行分块
- 2. 对分块的文本进行embedding（使用"text-embedding-v4"）
- 3. 通过使用cosine-similar方法进行query和分块文本的embedding对比后排序，得到最符合的参考文本。
- 4. 根据参考文本进行回答。

## 关键问题：
- 1. PDF处理办法：PyMupdf库
- 2. embedding方法：使用阿里云大模型text-embedding-v4
- 3. rerank使用：暂无使用rerank
- 4. 检索用的方法：cosine-similarity的方法进行检索。
![概述](dataset/1751855634412.jpg)
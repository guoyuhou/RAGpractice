# 项目报告

## 刘曜畅 2025.7.11

## 1. 当前在做什么项目？

我们当前的核心项目是为“研途”这一特定应用场景，设计、构建并系统性地评估一个高性能的检索增强生成（RAG）系统。该系统的最终目标是能够深入理解一个庞大的、专业的研究生考试题库知识库，并基于现有的习题的趋势分析、知识点关联和逻辑推理等，智能地生成高质量的、有针对性的题目。

## 2. 为什么要做这个项目？项目的重要性和意义是什么？

传统的LLM因其知识的静态和通用性，很难直接应用于需要高度专业化、事实准确化的专业性领域。RAG是目前公认的，最高效的将大模型的通用能力与企业私有知识库相结合的最佳实践。本项目旨在跑通一个工程上最优的RAG系统，为“研途”这类B2C的知识密集型应用落地做基础。

该项目的意义，正如很多论文中都提到的，即使是最先进的RAG系统，在面对复杂布局的文档解析、深度的语义理解、多源信息的整合推理以及生成答案的幻觉部分，依然存在非常大的挑战。本项目跑通完整的RAG，在此基础上进行问题的优化，本身就具有重要意义。

## 3. 项目的背景

RAG的发展已经有了长足的发展，从早期的关键词检索，演进到更复杂的Modular RAG和 Graph RAG。学术界和工业界已经对RAG的各个模块如parser、Chunking、Embedding、Reranking、Generator等，进行了大量的研究。

关于我们的定位，在研途这个具体的、全新的应用场景之下，科学地对比和选择当前业界顶尖的开源组件(如OCRFlux-3B,MistralOCR)或闭源API(Gemini-2.5-pro eg)，以组建一个性能最强的基线。这个经过验证的基线系统，不仅是‘研途’项目快速落地的保障，更是我们后续进行算法创新和学术研究的坚实起点。只有在这个基线上，我们后续的创新(无论是在算法还是Agentic架构上)才是有意义且可量化的。

## 4. 当前成果展示：

- 1. 完成RAG基础流水线的搭建：已成功搭建一个包含“parser-chunks-embedding-semantic retrieval-response generate”端到端的RAG流程
在线演示：https://rag-diary.streamlit.app/
代码：https://github.com/guoyuhou/RAGpractice


- 2. 完成parser部分模块的定量测评
通过对PymuPDF、Mistral-latest、OCRFlux-3B在OmniDocBench的上的基准测试，重点关注于Mistral-latest、OCRFlux-3B的评定结果。
将OCRFlux-3B模型部署到实验室服务器上。

- 3. 完成前沿RAG评测系统的深入研究
RAG benchmark最终调研结果于GitHub仓库中。包含FRAEMS、RGB、RAGTruth、GraphRAG等

## 5. 后续计划：

1. 第一阶段：完成最终基线系统的构建与评估
    a. 暂时定型parser，将OCRFlux-3B整合到RAG系统之中。
    b. 使用FRAMES和RGB来测试基础RAG，得到Baseline

2. 第二阶段：核心模块优化
    a. 优化分块(Chunking)调研并实验不同的分块策略(如按句子、语义分块)及其评价标准，将最优策略集成到RAG中。
    b. 优化检索(Retrieval)在现有检索流程中，引入"rerank"模块，并且测试其带来的精度提升
    c. 优化生成(Generate)使用RAGTruth测试当前系统的幻觉率，调研其利用RAGTruth对一个中等规模的开源模型(如Qwen、DeepSeek)进行微调的可能性(RAGTruth中已经展示可能性)，最终使用最优方案集成到RAG中。
    d. 集成所有工具，获得理论上最优RAG系统，并进行FRAMES、RGB benchmark的测评。

3. 第三阶段：探索创新
    在工程化完备的RAG中，思考技术性的创新如算法、架构的创新等。



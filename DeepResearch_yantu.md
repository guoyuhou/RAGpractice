1. 因材施教

经过调研，传统的个性化的推荐方法如（协同过滤、基于内容推荐）较“Knowledge Tracing + RL”的技术范式效果差异大[1]。在现有的研究中，[2],[3]对这种技术做了详细的研究，得到了更好的效果。并且由于本项目考虑以人工智能为核心技术，因此考虑放弃传统思路，采用更能拟合学生知识掌握状态的认知追踪与决策方法。
主要构成：
    1. 知识图谱(KG)
    2. 知识追踪模型(Knowledge Tracing Model, KTM)
    3. 知识需求模型(Knowledge Demanding Model, KDM)

考虑到知识图谱部分是在数据准备过程就完成的。之后所需要准备的工作是：1. KTM的数据集，即(Student, knowledge, 1/0, sequence)，2. 以这个数据集展开KTM训练。(此训练成本相较于一般大模型非常低)3. 训练结束后，采用以KTM为模拟环境，采用A2C(Advantage Actor-Critic)算法进行KTM训练。最终得到一个端到端的个性化学习系统。

在得到基础的KTM-KDM系统之后，以此为base line，在此基础上进行系统的优化，从而解决性能、幻觉等问题。



2. 题目与教材知识结构化。

2.1 知识点标注工具。

一般常用的知识点工具有[4]doccano(10k stars), [5]brat(1.9k stars), [6]label-studio等。brat虽然功能良好，但开发时间较久。doccano和label-studio和doccano均为开源数据标注工具。但label-studio拥有良好的团队管理。若经济情况允许(每月99刀)则直接选择label-studio，若经济情况不支持则选择doccano和label-studio进行本地服务器部署。

2.2 数据集范围和知识图谱构建方法

**数据集范围:**
    1. 以学科划范围，首先集中于考研政治和高等数学的试点学科进行。
    2. 以资料类型划范围：考试真题和官方解析（用于知识图谱生成）、内部教材（内部教材主要是corpus）
    3. 以时间角度划范围：搜寻近5年的资料。

2.3 知识图谱构建方法

    1. E-提取(extract):从doccano或label-studio种提取标注好的数据文件。
    2. T-转换(Transform): 编写python脚本将数据文件转换成标准的三元组
    3. L-加载(Load): 使用[7]Neo4j进行知识图谱生成

[1]https://dl.acm.org/doi/pdf/10.1145/3569576
[2]https://ieeexplore.ieee.org/abstract/document/9064104
[3]https://www.sciencedirect.com/science/article/pii/S0950705125011207
[4]https://github.com/doccano/doccano
[5]https://github.com/nlplab/brat
[6]https://github.com/HumanSignal/label-studio
[7]https://github.com/neo4j/neo4j

1.3 RAG方案
RAG部分主要采取三个方面。
1. 采取GraphRAG来依据KDM推荐的知识点进行检索，从而得出和这个知识点关联度最高的知识点。
2. 采取AdvancedRAG，从习题库中得出关联度最高的习题。
3. 采取AdvancedRAG，从内部教材中得到相应的知识点对应的内容。

结合三个retrieval 部分得到的corpus，最终进行generate 题目。
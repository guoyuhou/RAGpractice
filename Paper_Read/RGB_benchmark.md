#  Benchmarking Large Language Models in Retrieval-Augmented Generation

*Address: https://ojs.aaai.org/index.php/AAAI/article/view/29728*

## 1. 这篇论文解决了什么问题？

过去没有严格评估RAG的benchmark，因此这篇文章制作了从四个RAG的基本方面：noise rubustness, negative rejection, information integration and counterfactural robustness四个角度来测评

## 2. 这个问题为什么重要，重要在什么地方？

与FRAMES出现的契机相同，均是之前没有统一的测试RAG效果的benchmark，故此文章设计了这个数据集。

## 3. 原来大家都是怎么解决这个问题的？

过去大家很少专注于RAG的整体功能，而且基本上都使用的传统的QA问答。这一点和FRAMES基本相同。

## 4. 这篇文章和原来文章的差异点是什么？创新点是什么？

这篇文章和FRAMES从事物的两个方向来进行RAG的评估。FRAMES是从正面的角度，在面对复杂信息，去测试RAG的效果好到什么程度。RGB是从反面进行思考，在面对错误信息，去测试RAG的效果稳定性如何。创新点是从反面的角度去思考，去看RAG“错到什么程度。”

## 5. 和我们做的关联是什么？

RGB和我们做的高度关联，它为我们提供了系统性压力测试和风险评估工具。我们需要一个完整的benchmark来评估RAG的性能，从而找到一个base line进而不断优化和创新。而这个RGB是从反面的角度看我们的RAB在面对错误信息的时候有多高的稳定性。

## 6. 我们想解决的问题和文章想解决的问题的共同点和差异点是什么？能不能用文章的思路解决？预期有什么困难？

共同点和FRAMES那篇仍然高度一致，都是为了提升RAG的可靠性与准确性。我们仍然可以用文章的思路解决，但是我们需要综合FRAMES来进行测评，从而保证最优使用FRAMES，而最差来用RGB来测评。从而可以计划出RAG的功能上限和下限。而且从创新点的角度来讲，就可以从这两方面入手来思考优化地方。来权衡激进和稳定两方面。一个敢于推理的模型，或许可能陷入到很深的错误之中；而一个推理稳健的模型，性能可能较差，但是稳定性更高。所以或许这里是一个非常不错的创新点，即权衡性能和稳定性之间的惯性系。预期的困难就是，完整地跑通FRAMES和RGB两个数据集需要大量的计算资源和时间，
## 题目一：热点预测模型：从 MoE 推理 trace 中学习专家命中规律

**方向简介：**
在 MoE（Mixture of Experts）模型的推理过程中，每个输入 token 通常只会被路由到少量专家。不同任务、不同语义模式和不同上下文长度会导致各层专家命中分布出现明显差异，其中部分专家会在特定请求中形成“热点”。本课题将提供 chat、math、code 三类任务的专家命中 trace 数据，每类各 1000 条，共 3000 条样本。项目需要基于这些 trace 数据训练一个热点专家预测模型，在给定新的请求输入时，预测 MoE 各层最可能被命中的 Top-K 专家。

------

**具体研究内容包括：**
(1) 学习 MoE 推理中的基本路由机制，理解 token routing、expert selection、expert demand、per-layer expert distribution、Top-K expert 等概念，并分析不同 prompt 为什么会触发不同专家；

(2) 熟悉三类 trace 数据的结构与语义，包括 chat、math、code 请求的 prompt token 序列、层编号、专家编号、专家命中次数或命中列表等字段，完成数据读取、清洗和统计分析；

(3) 设计并训练热点专家预测模型。模型结构不限，可尝试 Transformer Encoder、注意力池化、层感知 query pooling、多任务学习结构或其他序列建模方法，使模型能够学习 prompt token 与专家命中模式之间的隐式关系；

(4) 研究如何稳定训练超参数模型，包括训练/验证/测试集划分、类别不均衡处理、层间共享与层间独立输出头设计、损失函数选择、正则化、早停策略和随机种子控制等；

(5) 围绕 Top-K 专家预测任务设计评测流程，重点使用 Recall@K 和 NDCG@K 衡量模型是否能覆盖真实热点专家以及排序质量，并分别统计整体指标、分任务指标和分层指标；

(6) 构建完整实验程序，支持读取 trace 数据、生成训练样本、训练模型、保存 checkpoint、执行推理、输出每层 Top-K 专家预测结果，并生成指标报告和可视化分析；

(7) 总结模型效果与局限性，分析 Recall 和 NDCG 未达到预期时的原因，例如 trace 样本规模有限、专家命中随机性强、任务类别差异不足或模型过拟合等，并提出后续改进方向。

------

**主要工具：**
Python、PyTorch / TensorFlow、Transformers、NumPy / Pandas（数据处理）、Scikit-learn（baseline 与指标）、Matplotlib / Seaborn（结果可视化）、JSON/JSONL/CSV（trace 数据输入输出）。

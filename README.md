# MoE 热点专家预测项目说明

## 0. 项目信息

- 作者：万昊
- 学号：25012100064
- 提交文件命名：`wh_0064.jsonl`
- 项目方向：方向 A，MoE 热点专家预测

本项目用于完成方向 A“从 MoE 推理 trace 中学习专家命中规律”。输入是一条请求的 `input_ids`、`attention_mask`、`prompt_len` 等字段，输出每个 `request_id` 在 48 个 MoE 层上的专家排序列表，格式符合 `submission_guide.md`。

## 1. 任务理解

MoE 模型每层有 128 个 expert，每个 token 只会被路由到少数 expert。给定一段 prompt 后，不同层会形成不同的专家命中分布。评测方隐藏真实 `raw_counts`，要求我们预测每层最可能命中的 Top-K expert，并用多个 K 下的 Recall@K 与 NDCG@K 评价：

- Recall@K：真实热点专家是否被预测列表前 K 覆盖。
- NDCG@K：预测顺序是否把更高命中的 expert 排在更前。

因此该任务不是单标签分类，而是“每层一个 128 维排序/分布预测”问题。

## 2. 数据说明

原始材料位于 `../题目材料`：

- `trace数据/1k_chat_all.jsonl`
- `trace数据/1k_code_all.jsonl`
- `trace数据/1k_math_all.jsonl`
- `valid.jsonl`
- `tokenizer/`

每条训练样本主要字段：

- `request_id`：请求编号。
- `input_ids`、`attention_mask`、`prompt_len`：模型输入。
- `task_family`：任务类型。
- `num_layers=48`、`num_experts=128`。
- `raw_counts`：形状为 `[48, 128]`，表示每层每个 expert 的命中次数。

正式测试文件只会保留输入字段，不含 `raw_counts` 或真实标签。

## 3. 模型思路

当前实现采用轻量 PyTorch 模型 `HotExpertPredictor`：

1. 使用 token embedding 将 `input_ids` 转为向量。
2. 按 `attention_mask` 做 masked mean pooling，得到请求级表示。
3. 加入 `task_family` embedding 和 `prompt_len` 长度特征。
4. 为 48 个 MoE 层分别加入 layer embedding，使不同层拥有不同预测偏好。
5. 输出 `[batch, 48, 128]` logits，每层对应 128 个 expert 的预测分数。

训练标签由 `raw_counts` 按层归一化得到专家命中分布。损失函数使用 KLDivLoss，使模型学习完整分布；推理时对每层 logits 从高到低排序，得到提交要求的 expert id 列表。

这种结构比单纯统计全局热门 expert 更有表达能力，但仍能在 CPU 上较快训练，适合招新验收的可复现要求。

## 4. 文件结构

```text
hot_expert_predictor/
  config.json                 训练和模型超参数
  train.py                    训练脚本，保存 best_model.pt
  predict.py                  生成提交 JSONL
  baseline.py                 全局热门 expert 统计基线
  validate_submission.py      检查提交格式
  src/
    data.py                   数据读取和 Dataset
    metrics.py                Recall@K / NDCG@K
    model.py                  PyTorch 模型
  outputs/
    best_model.pt             已训练模型 checkpoint
    train_history.json        训练指标记录
    valid_prediction.jsonl    valid.jsonl 的样例预测
```

## 5. 运行方法

进入项目目录：

```bash
cd "C:\Users\25998\Desktop\Lab mission\hot_expert_predictor"
```

安装依赖：

```bash
pip install -r requirements.txt
```

训练模型：

```bash
python train.py --material-dir "..\题目材料" --output-dir outputs
```

快速自检训练：

```bash
python train.py --material-dir "..\题目材料" --output-dir outputs_smoke --limit-samples 180
```

对正式无标签测试文件生成提交结果：

```bash
python predict.py --input-jsonl "测试输入.jsonl" --checkpoint outputs\best_model.pt --output-jsonl "wh_0064.jsonl" --top-k 64
```

检查提交文件格式：

```bash
python validate_submission.py --submission "wh_0064.jsonl" --min-k 16
```

如果临近截止来不及重新训练，也可以用统计基线生成结果：

```bash
python baseline.py --material-dir "..\题目材料" --input-jsonl "测试输入.jsonl" --output-jsonl "baseline_submission.jsonl" --top-k 64
```

## 6. 当前实验结果

在 3000 条 trace 上按 85%/15% 划分训练验证，默认配置训练 8 轮，验证集最后一轮结果约为：

| 指标 | 数值 |
|---|---:|
| Recall@4 | 0.3033 |
| Recall@8 | 0.3793 |
| Recall@16 | 0.4649 |
| Recall@32 | 0.5788 |
| NDCG@4 | 0.5114 |
| NDCG@8 | 0.5406 |
| NDCG@16 | 0.5785 |
| NDCG@32 | 0.6271 |

已生成 `outputs/valid_prediction.jsonl`，并通过格式校验。

## 7. 提交内容建议

正式提交时至少包含：

- 按老师要求命名的预测结果 JSONL：`wh_0064.jsonl`。
- `README.md` 或本项目说明。
- 如老师要求复现，再附带 `hot_expert_predictor` 代码目录和 `outputs/best_model.pt`。

预测结果每行形如：

```json
{"request_id":"req_000001","predicted_experts":[[... 64 expert ids ...], ... 48 layers ...]}
```

每层 expert id 没有重复，范围为 `0 <= expert_id < 128`，并按置信度从高到低排列。

## 8. 局限与改进

当前模型主要使用 token 平均池化，训练速度快，但对长 prompt 中局部语义和 token 顺序建模有限。后续可改进为：

- 使用 Transformer Encoder 或轻量 CNN/attention pooling 建模 token 顺序。
- 按 chat/math/code 分任务训练或加入更强任务识别特征。
- 对不同层使用更细的独立输出头。
- 对高命中 expert 施加更大权重，提升 Top-4、Top-8 排序质量。
- 使用交叉验证或模型集成，让隐藏测试集更稳定。

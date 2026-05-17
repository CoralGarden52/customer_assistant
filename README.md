# customer_assistant

基于 GraphRAG 的客服问答项目，面向产品手册与售后知识场景，支持中文/英文问题检索与答案生成，并提供批量问答输出能力。

## 1. 项目目标

- 将分散的产品手册知识（文本+插图）统一索引
- 通过混合检索（BM25 + GraphRAG）提升答案相关性
- 支持批量问题自动生成提交结果（`submission.csv`）
- 支持多模态预处理（保留 `<PIC>` 与图文位置映射）

## 2. 主要能力

- GraphRAG 图谱构建与社区报告生成
- 文本向量索引（LanceDB）
- BM25 关键词检索 + GraphRAG 语义检索融合
- 可选 reranker 重排（`bge-reranker-v2-m3`）
- 批处理并发、断点续跑、失败兜底回复

## 3. 项目结构

```text
customer_assistant/
├─ data/                         # 原始数据、题目与提交文件
│  ├─ question_public.csv
│  ├─ submission.csv
│  └─ 手册/
├─ graphrag_project/             # 主 GraphRAG 工程
│  ├─ input/
│  ├─ output/
│  ├─ cache/
│  ├─ logs/
│  ├─ prompts/
│  ├─ settings.yaml
│  ├─ run_submission.py
│  ├─ query_with_rerank.py
│  └─ embedding_server.py
├─ graphrag_unified/             # 多模态/统一版本实验工程
│  ├─ preprocess_multimodal.py
│  ├─ build_text_unit_image_map.py
│  ├─ analyze_unmapped_images.py
│  └─ settings.yaml
├─ graphrag_common.py
├─ merge_submission.py
└─ README.md
```

## 4. 技术方案

1. 数据预处理  
将手册文本切分为 chunk，并在多模态流程中保留 `<PIC>` 标记，记录图片在 chunk 内的位置与上下文。

2. 索引构建（GraphRAG）  
执行实体/关系抽取、社区聚类、社区报告生成与文本 embedding，结果写入 `output/` 与 `cache/`。

3. 在线问答（混合检索）  
`run_submission.py` 先做 BM25 召回，再做 GraphRAG local search，融合上下文后调用 LLM 生成答案。

4. 结果生成  
批量读取问题集，支持并发执行、增量保存与断点续跑，输出 `submission.csv`。


### 5.1 准备环境

- Python 3.10+
- 安装 GraphRAG 及项目依赖（按你当前环境为准）
- 准备模型/API 配置

### 5.2 配置密钥

在环境变量中配置（示例）：

```powershell
$env:GRAPHRAG_API_KEY="your_api_key"
```


### 5.3 构建索引

在 `graphrag_project` 或 `graphrag_unified` 下按你的 GraphRAG 流程执行索引构建命令。

### 5.4 生成提交结果

```powershell
cd graphrag_project
python run_submission.py
```

## 6. 关键配置说明（`settings.yaml`）

- `models`: Chat/Embedding 模型与并发参数
- `chunks`: 文本切块大小与重叠
- `vector_store`: 向量库类型与存储路径（LanceDB）
- `local_search/global_search`: 查询策略与提示词




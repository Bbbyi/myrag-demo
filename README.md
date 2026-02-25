# myrag-demo
rag demo
核心技术: Python、自然语言处理（NLP）、大模型微调（LoRA）、向量检索（FAISS）、光学字符识别（OCR）
框架 / 库: LangChain、Transformers、PaddleOCR、Gradio、、jieba、HuggingFace Embeddings
工具 / 能力: 文本语义分块、混合检索（向量 + 关键词）、大模型推理、自定义 Prompt 工程、知识库构建与管理

本地私有化图文知识库问答系统

基于 Python 构建私有化本地知识库问答系统，支持 PDF / 图片 / TXT 多格式文件上传，结合 NLP 与大模型技术实现精准问答，数据全程本地化处理。
知识库构建：
实现多格式文件解析（PaddleOCR 解析图片、PyPDF2 解析 PDF、多编码兼容解析 TXT），解决不同格式文本提取问题；
基于 LangChain 实现文本语义分块，结合 KeyBERT / 正则完成主题关键词提取，提升检索精准度；
采用 FAISS 构建向量库，实现 Embedding 模型（HuggingFace）的向量存储与检索，支持维度校验、损坏库备份等容错机制。
问答核心能力：
设计混合检索策略（向量相似度 + 主题匹配 + 关键词过滤），三层过滤机制提升检索结果相关性；
基于 Qwen2-0.5B-Instruct 模型，通过 LoRA 轻量化微调定制问答能力，适配垂直场景问答；
实现多问题智能拆分（规则 + 大模型），支持批量问题解析与精准回答；
自定义 Prompt 工程约束回答规则，结合结果清洗逻辑，确保回答格式统一、无冗余信息。
交互层实现：
使用 Gradio 搭建可视化界面，支持文件上传、分类管理等功能；
全局异常处理与日志记录.

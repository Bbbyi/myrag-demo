import os
import faiss
import jieba
import logging
import yaml
from datetime import datetime

from paddleocr import PaddleOCR
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# 修正Document导入路径（兼容所有langchain版本）
try:
    from langchain.schema import Document  # 新版本
except ImportError:
    from langchain_core.documents import Document  # 旧版本

# 配置加载
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("kb_log.log", encoding="utf-8")]
)
logger = logging.getLogger("LocalKnowledgeBase")

# 初始化OCR（强制CPU）
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)


class LocalKnowledgeBase:
    def __init__(self):
        """初始化知识库"""
        # 1. 初始化Embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["embedding_model"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16}
        )
        # 校验Embedding维度
        try:
            self.embedding_dim = len(self.embeddings.embed_query("测试"))
            logger.info(f"Embedding模型加载成功，维度：{self.embedding_dim}")
        except Exception as e:
            raise Exception(f"Embedding初始化失败：{e}")

        # 2. 初始化文本分块器（备用，主要用语义分块）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "],
            keep_separator=True,
            is_separator_regex=False,
            length_function=len
        )

        # 3. 加载/初始化向量库
        self.vector_db = self._load_vector_db()

        # 4. 初始化KeyBERT模型（全局初始化，避免重复加载）
        try:
            from keybert import KeyBERT
            self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')
            logger.info("KeyBERT模型加载成功（无监督关键词抽取）")
        except Exception as e:
            logger.warning(f"KeyBERT加载失败，降级为正则关键词提取：{e}")
            self.kw_model = None

    def _load_vector_db(self):
        """加载/初始化向量库（带备份）"""
        db_path = os.path.abspath(CONFIG["vector_db_path"])
        # 加载已有库
        if os.path.exists(db_path):
            try:
                vector_db = FAISS.load_local(
                    db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # 校验维度
                if hasattr(vector_db.index, 'd') and vector_db.index.d != self.embedding_dim:
                    raise Exception(f"维度不匹配：库{vector_db.index.d} vs 模型{self.embedding_dim}")
                logger.info(f"向量库加载成功：{db_path}")
                return vector_db
            except Exception as e:
                logger.error(f"加载向量库失败：{e}")
                # 备份损坏库
                backup_path = f"{db_path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_broken"
                try:
                    os.rename(db_path, backup_path)
                    logger.info(f"损坏库已备份：{backup_path}")
                except Exception as rename_e:
                    raise Exception(f"向量库损坏且备份失败：{rename_e}")

        # 初始化新库
        logger.info(f"初始化新向量库：{db_path}")
        os.makedirs(db_path, exist_ok=True)
        index = faiss.IndexFlatL2(self.embedding_dim)
        vector_db = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vector_db.save_local(db_path)
        return vector_db

    def _parse_pdf(self, file_path):
        """解析PDF"""
        try:
            reader = PdfReader(file_path)
            if reader.is_encrypted:
                logger.error(f"PDF加密：{file_path}")
                return ""
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text += f"\n【页码：{i + 1}】{page_text}"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF解析失败：{file_path} - {e}")
            return ""

    def _parse_image(self, file_path):
        """解析图片（OCR）"""
        try:
            result = ocr.ocr(file_path, cls=True)
            if not result or len(result) == 0:
                logger.warning(f"OCR无结果：{file_path}")
                return ""
            text = ""
            for line in result:
                if line:
                    for elem in line:
                        if isinstance(elem, (list, tuple)) and len(elem) >= 2:
                            line_text = elem[1][0] if isinstance(elem[1], (list, tuple)) else str(elem[1])
                            if line_text.strip():
                                text += line_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"图片解析失败：{file_path} - {e}")
            return ""

    def _parse_file(self, file_path):
        """统一解析文件"""
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            return self._parse_pdf(file_path)
        elif ext in [".jpg", ".png", ".jpeg"]:
            return self._parse_image(file_path)
        elif ext == ".txt":
            encodings = ["utf-8", "gbk", "gb2312"]
            for enc in encodings:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        return f.read().strip()
                except UnicodeDecodeError:
                    continue
            logger.error(f"TXT编码不支持：{file_path}")
            return ""
        else:
            logger.warning(f"不支持的格式：{ext}")
            return ""

    def extract_topic(self, chunk):
        """
        混合关键词提取策略：优先KeyBERT无监督，失败则降级为正则
        """
        # 1. 优先使用KeyBERT
        if self.kw_model is not None:
            try:
                keywords = self.kw_model.extract_keywords(
                    chunk,
                    keyphrase_ngram_range=(1, 2),
                    top_n=1,
                    diversity=0.0
                )
                return keywords[0][0] if keywords else "未知"
            except Exception as e:
                logger.warning(f"KeyBERT提取失败，降级为正则：{e}")

        # 2. 降级为正则（兼容你的原有场景）
        import re
        topic_pattern = r'([\u4e00-\u9fff]+猫|[\u4e00-\u9fff]+鱼|[\u4e00-\u9fff]+肉|[\u4e00-\u9fff]+菜)'
        match = re.search(topic_pattern, chunk)
        return match.group(1) if match else "未知"

    def _extract_core_topic(self, topic):
        """提取主题核心词（解决修饰词导致的匹配失败）"""
        if topic == "未知":
            return "未知"
        import re
        # 匹配核心品类词（猫/鱼/肉/菜等），提取最核心的名词
        core_pattern = r'([\u4e00-\u9fff]+猫|[\u4e00-\u9fff]+鱼|[\u4e00-\u9fff]+肉|[\u4e00-\u9fff]+菜)'
        match = re.search(core_pattern, topic)
        return match.group(1) if match else topic

    def split_text_by_semantic(self, raw_text, max_chunk_len=128):
        """
        语义分块 + 主题关键词绑定
        返回：[(chunk_text, topic), ...]
        """
        import re
        # 第一步：基础清洗
        clean_raw = raw_text.strip().replace(" ", "").replace("\n", "。")

        # 第二步：按语义边界拆分
        semantic_separators = r'[。！？；]'
        sentences = re.split(semantic_separators, clean_raw)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

        # 第三步：合并成合理长度的chunk
        chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) <= max_chunk_len:
                current_chunk += sent + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sent + "。"
        if current_chunk:
            chunks.append(current_chunk)

        # 第四步：过滤有效chunk + 绑定主题
        chunks_with_topic = []
        for chunk in chunks:
            clean_chunk = chunk.strip().replace(" ", "").replace("\n", "")
            if clean_chunk and len(clean_chunk) >= 10 and any('\u4e00' <= c <= '\u9fff' for c in clean_chunk):
                topic = self.extract_topic(chunk)  # 绑定主题
                chunks_with_topic.append((chunk, topic))

        # 兜底
        if not chunks_with_topic:
            raise Exception("无有效语义块（全为空白/标点/非中文）")

        return chunks_with_topic

    def add_document(self, file_obj, category="default"):
        """添加文档到知识库（核心修改：带主题元数据入库）"""
        try:
            # 处理Gradio文件对象
            file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
            # 基础校验
            if not os.path.exists(file_path):
                raise Exception(f"文件不存在：{file_path}")
            if os.path.getsize(file_path) == 0:
                raise Exception(f"文件为空：{file_path}")

            # 解析文件
            raw_text = self._parse_file(file_path)
            # 长度校验
            ext = os.path.splitext(file_path)[-1].lower()
            min_len = 5 if ext in [".jpg", ".png", ".jpeg"] else 10
            if not raw_text or len(raw_text) < min_len:
                raise Exception(f"有效文本不足（需≥{min_len}字符）：{len(raw_text) if raw_text else 0}")

            # 语义分块 + 主题绑定
            chunks_with_topic = self.split_text_by_semantic(raw_text, max_chunk_len=128)
            if not chunks_with_topic:
                raise Exception("无有效语义块")

            # 封装带元数据的Document对象（核心：主题存入metadata）
            file_name = os.path.basename(file_path)
            docs_to_add = []
            for chunk_text, topic in chunks_with_topic:
                # 构建带元信息的文本内容
                content = f"【分类：{category}】【文档：{file_name}】【主题：{topic}】{chunk_text}"
                # 封装Document，主题同时存入metadata（方便检索时过滤）
                doc = Document(
                    page_content=content,
                    metadata={"topic": topic, "category": category, "file_name": file_name}
                )
                docs_to_add.append(doc)

            # 入库（用add_documents替代add_texts，保留metadata）
            added_ids = self.vector_db.add_documents(docs_to_add)
            if len(added_ids) != len(docs_to_add):
                raise Exception(f"入库失败：预期{len(docs_to_add)}条，实际{len(added_ids)}条")

            # 保存向量库
            self.vector_db.save_local(CONFIG["vector_db_path"])
            logger.info(f"文档添加成功：{file_path}（{len(added_ids)}块，主题：{[t for _, t in chunks_with_topic]}）")
            return True

        except Exception as e:
            error_msg = f"添加失败：{e}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)

    def _calculate_text_similarity(self, text1, text2):
        """
        计算两个文本的余弦相似度（基于已加载的embedding模型）
        返回：0~1之间的相似度值（1=完全相似，0=完全不相似）
        """
        try:
            # 生成向量
            emb1 = self.embeddings.embed_query(text1)
            emb2 = self.embeddings.embed_query(text2)

            # 计算余弦相似度（避免除以0）
            import numpy as np
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.warning(f"计算文本相似度失败：{e}")
            return 0.0

    def hybrid_search(self, query, top_k=3):
        """混合检索（核心优化：向量相似度主题匹配+三层过滤）"""
        stop_words = {"的", "了", "是", "我", "你", "他", "它", "在", "有", "和", "就", "都", "而", "及",
                      "与", "着", "也", "还", "把", "被", "为", "之", "这", "那", "哪", "怎么", "如何", "方法", "做法"}

        # 统一输入格式
        is_single = False
        if isinstance(query, str):
            is_single = True
            queries = [query.strip()]
        elif isinstance(query, list):
            queries = [q.strip() for q in query if q.strip()]
        else:
            logger.error("query类型错误（仅支持str/list）")
            return "查询格式错误", "" if is_single else {}

        if not queries:
            return "查询内容为空", "" if is_single else {}

        # 批量检索
        batch_results = {}
        # 主题相似度阈值（可调整）
        TOPIC_SIMILARITY_THRESHOLD = 0.55

        for q in queries:
            try:
                # 步骤1：提取查询的主题关键词
                query_topic = self.extract_topic(q)
                logger.info(f"查询「{q}」的核心主题：{query_topic}")

                # 步骤2：向量检索（先扩大范围，再过滤）
                enhanced_q = f"主题：{q}"
                candidates = self.vector_db.similarity_search_with_score(enhanced_q, k=top_k * 5)
                if not candidates:
                    batch_results[q] = ("未检索到相关知识库内容", "")
                    continue

                # 步骤3：三层过滤（向量相似度主题匹配+相似度+关键词）
                filtered_docs = []
                for doc, score in candidates:
                    # 过滤1：基于向量相似度的主题匹配
                    doc_topic = doc.metadata.get("topic", "未知")
                    # 提取核心词
                    query_core = self._extract_core_topic(query_topic)
                    doc_core = self._extract_core_topic(doc_topic)

                    # 主题匹配逻辑
                    need_filter = False
                    if query_core != "未知" and doc_core != "未知":
                        # 计算核心词语义相似度
                        topic_similarity = self._calculate_text_similarity(query_core, doc_core)
                        if topic_similarity < TOPIC_SIMILARITY_THRESHOLD:
                            need_filter = True
                            logger.debug(f"主题过滤：查询[{query_core}] 文档[{doc_core}] 相似度[{topic_similarity:.2f}]")

                    if need_filter:
                        continue

                    # 过滤2：向量库相似度阈值
                    if score > 0.85:
                        continue

                    # 过滤3：关键词匹配
                    core_words = [w.strip() for w in jieba.lcut(q, cut_all=False) if
                                  w.strip() and w not in stop_words and len(w) > 1]
                    if not core_words:
                        core_words = [q]
                    if not any(kw.lower() in doc.page_content.lower() for kw in core_words):
                        continue

                    filtered_docs.append(doc)

                # 步骤4：整理结果
                if not filtered_docs:
                    batch_results[q] = ("未检索到相关知识库内容", "")
                    continue

                final_docs = filtered_docs[:top_k]
                # 提取来源
                source = final_docs[0].page_content.split("】")[:3]
                source = "】".join(source) + "】" if final_docs else ""
                # 提取纯内容（去掉元信息）
                content = []
                for doc in final_docs:
                    split_content = doc.page_content.split("】")
                    content.append(split_content[3] if len(split_content) >= 4 else "")

                batch_results[q] = ("\n\n".join(content), source)

            except Exception as e:
                logger.error(f"检索失败（{q}）：{e}")
                batch_results[q] = (f"检索出错：{e}", "")

        # 适配返回格式
        if is_single:
            return batch_results[queries[0]][0], batch_results[queries[0]][1]
        else:
            return batch_results

    def search(self, query, top_k=3):
        """兼容旧接口的检索函数"""
        return self.hybrid_search(query, top_k)
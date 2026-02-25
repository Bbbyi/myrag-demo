import os
import re
import yaml
import logging
import gradio as gr
from knowledge_base import LocalKnowledgeBase

# 全局变量初始化
CONFIG = {}
kb = None
chater1 = None
question_splitter = None

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("kb_app")


def init_global():
    """初始化全局变量（模型/知识库/分词器）"""
    global CONFIG, kb, chater1, question_splitter
    # 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
    # 初始化知识库
    kb = LocalKnowledgeBase()
    # 初始化LLM和问题拆分器（保留你原有逻辑）
    try:
        from infer import chater
        from question_splitter import QuestionSplitter
        chater1 = chater(CONFIG["model_name"])
        question_splitter = QuestionSplitter(CONFIG["model_name"])
    except Exception as e:
        logger.warning(f"模型/拆分器初始化失败（仅影响问答，不影响文件上传）：{e}")

        # 兜底：模拟LLM返回
        class MockChater:
            def generate_answer(self, prompt):
                return prompt.split("【你的回答】：")[-1].strip()

        chater1 = MockChater()

        # 兜底：模拟拆分器
        class MockSplitter:
            def split(self, q):
                return q.split("？") if "？" in q else [q]

        question_splitter = MockSplitter()


def upload_file_to_kb(file, category, chat_history):
    """上传文件到知识库（保留你原有所有逻辑）"""
    global kb
    if file is None:
        chat_history.append(("系统", "请选择要上传的文件！"))
        return chat_history
    try:
        success = kb.add_document(file, category)
        if success:
            file_name = os.path.basename(file.name if hasattr(file, 'name') else str(file))
            chat_history.append(("系统", f"✅ 文件 {file_name} 已成功加入【{category}】分类！"))
    except Exception as e:
        chat_history.append(("系统", f"❌ 文件上传失败：{str(e)}"))
    return chat_history


def clean_model_answer(answer, file_source=None, is_general_answer=False):
    """精准清理模型回答（修复多问题截取bug+序号索引越界）"""
    # 1. 核心修复：只截取【你的回答】之后的真实回答（兼容单/多问题）
    real_answer = answer.strip()
    core_separator = "【你的回答】："
    if core_separator in real_answer:
        real_answer = real_answer.split(core_separator)[-1].strip()

    # 2. 移除模型返回的Prompt/知识库冗余内容（新增多问题场景清理）
    redundant_markers = [
        "【历史对话】：", "【知识库内容】：", "【问题】：①", "【相关内容】：",
        "你是专业的问答助手，严格按以下规则回答"
    ]
    for marker in redundant_markers:
        if marker in real_answer:
            real_answer = real_answer.split(marker)[0].strip()

    # 3. 清理模型编造的多余内容（补充多问题场景的冗余话术）
    clean_patterns = [
        "希望这个简单的步骤能够帮到你",
        "如果你有任何疑问或者需要进一步的帮助，请随时告诉我",
        "希望这个方法能帮助你",
        "如果你还有其他问题或者需要更多帮助，请随时告诉我",
        "！", "～", "。", "\n\n\n", "\n\n"
    ]
    for pattern in clean_patterns:
        real_answer = real_answer.replace(pattern, "")

    # 4. 格式标准化（修复：支持任意数字序号，避免索引越界）
    def replace_number(match):
        num = int(match.group(1))
        if 1 <= num <= 9:
            return f"{'①②③④⑤⑥⑦⑧⑨'[num - 1]} "
        else:
            return f"{num}、"
    real_answer = re.sub(r'(\d+)\. ', replace_number, real_answer)
    clean_answer = real_answer.replace(". ", "、").strip()
    # 5. 补充溯源信息（严格按规则，保留原有解析逻辑）
    if not is_general_answer and file_source:
        category = "默认"
        doc_name = "无"
        if "【分类：" in file_source:
            category = file_source.split("【分类：")[1].split("】")[0]
        if "【文档：" in file_source:
            doc_name = file_source.split("【文档：")[1].split("】")[0]
        clean_answer += f"\n\n【分类：{category}】【文档：{doc_name}】"
    else:
        clean_answer += "\n\n【分类：默认】【文档：无（基于通用知识回答）】"

    # 6. 兜底：无有效内容时按规则返回（保留原有逻辑）
    if not clean_answer.replace("\n", "").replace(" ", ""):
        clean_answer = "未找到相关答案\n\n【分类：默认】【文档：无】"

    return clean_answer


def batch_clean_answers(batch_answer_dict, file_source_dict=None, is_general_dict=None):
    """批量清理多问题回答（保留原有函数，做兼容）"""
    clean_batch = {}
    file_source_dict = file_source_dict or {}
    is_general_dict = is_general_dict or {}
    for q, ans in batch_answer_dict.items():
        clean_batch[q] = clean_model_answer(
            answer=ans,
            file_source=file_source_dict.get(q, ""),
            is_general_answer=is_general_dict.get(q, True)
        )
    return clean_batch


def chat_with_kb(question, chat_history):
    """基于知识库问答（修复多问题核心bug，保留所有原有功能）"""
    global chater1, kb, CONFIG, question_splitter
    # 空值校验（保留原有逻辑）
    if not question.strip():
        chat_history.append(("系统", "请输入有效的问题！"))
        return "", chat_history

    raw_question = question.strip()
    # 1. 拆分问题（保留原有拆分逻辑）
    split_questions = question_splitter.split(raw_question)
    if not split_questions:
        split_questions = [raw_question]

    final_answer_parts = []
    for single_q in split_questions:
        single_q = single_q.strip()
        if not single_q:
            continue

        # 2.1 单个问题检索（复用原有hybrid_search）
        search_res, file_source = kb.hybrid_search(single_q, top_k=CONFIG["top_k"])
        is_general = (search_res == "未检索到相关知识库内容" or not search_res)
        if is_general:
            search_res = "未检索到相关知识库内容，请基于通用知识回答。"

        # 2.2 构建单个问题的Prompt（保留历史对话、规则约束）
        # 历史对话（保留原有最近3轮逻辑）
        context_str = ""
        recent_history = chat_history[-3:] if len(chat_history) > 3 else chat_history
        if recent_history:
            for user_q, assistant_a in recent_history:
                if user_q != "系统":
                    context_str += f"用户：{user_q}\n助手：{assistant_a}\n"

        # 构建单问题Prompt（保留所有原有规则）
        single_prompt = f"""
                    你是专业的问答助手，严格按以下规则回答，只输出答案，不要多余内容：
                    1. 只使用【知识库内容】里的信息，不编造、不添加食材、不脑补步骤。
                    2. 分点说明，只输出内容，不要重复问题。
                    3. 忽略知识库中的【问题】【相关内容】等标记文字，只看真实正文。
                    4. 不要解释、不要开场白、不要结束语、不要建议、不要客套话。
                    5. 不要输出任何特殊符号。
                    6. 无信息只回复：未找到相关答案。
                    7. 结尾只标注引用来源，格式：【分类：xx】【文档：xxx】
                    
                    【历史对话】：{context_str}
                    【知识库内容】：{search_res}
                    【用户问题】：{single_q}
                    【你的回答】：
                            """

        # 2.3 单个问题推理+清理
        single_answer = chater1.generate_answer(single_prompt)
        single_clean_ans = clean_model_answer(single_answer, file_source, is_general)

        # 2.4 整理多问题结果（加标识区分不同问题）
        final_answer_parts.append(f"{single_q}\n{single_clean_ans}")

    # 3. 合并多问题结果（保留原有更新历史逻辑）
    clean_ans = "\n\n".join(final_answer_parts)
    chat_history.append((raw_question, clean_ans))
    return "", chat_history


# 初始化（保留原有逻辑）
init_global()

# 构建Gradio界面（保留你原有所有UI布局）
with gr.Blocks(title="本地图文知识库问答助手") as demo:
    gr.Markdown("# 本地私有化图文知识库问答助手")
    gr.Markdown("支持PDF/图片/文本上传，基于本地知识库回答问题，数据永不外泄")

    with gr.Tab("核心功能"):
        chat_history = gr.Chatbot(file_types=None, type="messages", label="问答历史", height=400)

        # 文件上传行（保留原有布局）
        with gr.Row():
            file = gr.File(label="选择文件（PDF/图片/TXT）", file_types=["pdf", "txt", "image"])
            category = gr.Dropdown(["面试技巧", "行业知识", "默认"], label="文件分类", value="默认")
            upload_btn = gr.Button("📤 上传到知识库", variant="primary")

        # 问答行（保留原有布局）
        with gr.Row():
            question = gr.Textbox(label="输入问题", placeholder="比如：鱼香肉丝怎么做？酸菜鱼怎么做？", scale=8)
            submit_btn = gr.Button("🚀 提交问题", variant="secondary", scale=2)

    with gr.Tab("知识库管理"):
        gr.Markdown("### 已上传文件列表（待实现）")
        clear_kb_btn = gr.Button("🗑 清空知识库", variant="stop")
        clear_kb_btn.click(
            lambda ch: ch.append(("系统", "清空知识库功能待实现！")),
            inputs=[chat_history],
            outputs=[chat_history]
        )

    # 绑定事件（保留原有绑定逻辑）
    upload_btn.click(upload_file_to_kb, inputs=[file, category, chat_history], outputs=[chat_history])
    submit_btn.click(chat_with_kb, inputs=[question, chat_history], outputs=[question, chat_history])

# 启动（保留原有启动逻辑）
if __name__ == "__main__":
    demo.launch(
        server_port=CONFIG["gradio_port"],
        share=True,
        server_name="127.0.0.1"
    )
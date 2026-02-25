import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


class QuestionSplitter:
    def __init__(self, model_path):
        # 初始化模型（仅多问题时使用）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        # 极简Prompt（只让模型做一件事：拆分多问题，不用判断）
        self.prompt_template = """把以下包含多个问题的文本拆分成独立的单一问题，每个问题一行，只输出问题，不要序号和解释：
{text}
拆分结果：
"""

    def _is_multi_question(self, text):
        """
        代码规则判定是否为多问题（核心：替代模型的判断逻辑）
        返回：True=多问题，False=单一问题
        """
        text = text.strip()
        # 规则1：包含多问题分隔符（最核心）
        multi_separators = ["和", "与", "、", "，", "；", "？？", "怎么做怎么做", "介绍下介绍下"]
        if any(sep in text for sep in multi_separators):
            return True

        # 规则2：包含多个独立问题的特征（如多个“？”、多个“怎么做”）
        question_marks = text.count("？") + text.count("?")
        if question_marks >= 2:
            return True

        # 规则3：文本长度过长且包含多个动词（如“介绍 做法”）
        if len(text) > 20:
            action_words = ["介绍", "怎么做", "做法", "特点", "区别", "价格"]
            action_count = sum(1 for word in action_words if word in text)
            if action_count >= 2:
                return True

        # 其他情况都判定为单一问题
        return False

    def split(self, text):
        """
        拆分主逻辑：规则优先，模型兜底
        """
        raw_text = text.strip()
        if not raw_text:
            return [""]

        # 第一步：代码规则判定（源头避免模型误判）
        if not self._is_multi_question(raw_text):
            return [raw_text]  # 单一问题直接返回，不调用模型

        # 第二步：仅多问题时调用模型拆分
        prompt = self.prompt_template.format(text=raw_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")

        # 调用模型（极简参数，避免警告）
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.01,  # 最低随机性，严格按指令输出
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 解析结果
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        split_result = result.split("拆分结果：")[-1].strip()

        # 第三步：结果清洗（过滤回答、去重、去空）
        final_lines = []
        for line in split_result.split("\n"):
            line = line.strip()
            # 过滤模型生成的回答/解释（只保留问题）
            if not line or any(word in line for word in ["是", "有", "包括", "答", "：", "="]):
                continue
            # 过滤序号（如1.、①、-）
            line = re.sub(r"^[0-9]+[.、)]*|^[①②③④⑤⑥⑦⑧⑨⑩]|^[-·]", "", line).strip()
            if line and line not in final_lines:
                final_lines.append(line)

        # 兜底：如果模型输出异常，返回按分隔符拆分的结果
        if not final_lines:
            # 按核心分隔符拆分
            for sep in ["和", "与", "、", "，"]:
                if sep in raw_text:
                    final_lines = [x.strip() for x in raw_text.split(sep) if x.strip()]
                    break
            # 仍无结果则返回原文本
            if not final_lines:
                final_lines = [raw_text]

        return final_lines


# 测试代码
if __name__ == "__main__":
    MODEL_PATH = r"//model/qwen"
    splitter = QuestionSplitter(MODEL_PATH)

    # 测试1：单一问题（核心验证）
    # test_text1 = "介绍下缅因猫"
    # result1 = splitter.split(test_text1)
    # print("测试1 - 单一问题：")
    # print(f"输入：{test_text1}")
    # print(f"输出：{result1}\n")  # 预期：['介绍下缅因猫']

    # # 测试2：多问题（验证拆分功能）
    test_text2 = "介绍下缅因猫，介绍下狸花猫"
    result2 = splitter.split(test_text2)
    print("测试2 - 多问题：")
    print(f"输入：{test_text2}")
    print(f"输出：{result2}\n")  # 预期：['介绍下缅因猫', '介绍下狸花猫']
    #
    # 测试3：复杂多问题
    test_text3 = "酸菜鱼怎么做？鱼香肉丝怎么做？"
    result3 = splitter.split(test_text3)
    print("测试3 - 复杂多问题：")
    print(f"输入：{test_text3}")
    print(f"输出：{result3}")  # 预期：['酸菜鱼怎么做？', '鱼香肉丝的做法', '鱼香肉丝的特点？']
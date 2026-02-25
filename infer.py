import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class chater():
    def __init__(self, model_name):
        self.model_name = model_name
        # 初始化时直接加载模型，避免重复加载
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        self.load_model()  # 初始化时就加载模型
        self.max_length = 1024

    def load_model(self):
        # 修复 torch_dtype 废弃警告：改为 dtype
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype=torch.float32,  # 替换 torch_dtype=dtype
            device_map="cpu",
            local_files_only = True,
            low_cpu_mem_usage=True,
        )

        # 加载LoRA权重
        self.lora_model = PeftModel.from_pretrained(
            self.base_model,
            "./custom_chat_model"  # 确保该路径下有LoRA权重文件
        )

    def generate_answer(self, question):
        # Qwen对话格式拼接
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        # 生成回答
        outputs = self.lora_model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        # 解码提取回答
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # answer = answer.split("<|im_start|>assistant\n")[1].replace("<|im_end|>", "").strip()
        answer = answer.split("<|im_start|>user\n")[1].replace("<|im_end|>", "").strip()
        return answer

if __name__ == '__main__':
    # 1. 加载基础模型和分词器（和训练时一致）
    # model_name = "Qwen/Qwen2-0.5B-Instruct"
    model_name = r"D:/python/myrag-demo/model/qwen"
    chater1 = chater(model_name)
    chater1.load_model()
    # 4. 测试微调效果（问我们训练过的问题）
    print("===== 测试微调效果 =====")
    questions = [
        "应届生简历没工作经验怎么办？",
        "简历中的项目经验怎么写？",
    ]
    for q in questions:
        print(f"问题：{q}")
        print(f"回答：{chater1.generate_answer(q)}\n")
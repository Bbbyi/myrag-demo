import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


# 1. 目标问答
TRAIN_DATA = [
    # 原始样本
    "<|im_start|>user\n简历中的项目经验怎么写？<|im_end|>\n<|im_start|>assistant\n项目经验用STAR法则写：S（背景）→T（任务）→A（行动）→R（结果），量化成果，比如「优化接口使响应速度提升30%」。<|im_end|>",
    # 同义改写样本（增强泛化）
    "<|im_start|>user\n简历里的项目经验该如何撰写？<|im_end|>\n<|im_start|>assistant\n撰写项目经验建议用STAR法则，分别说明背景、任务、行动和结果，尽量用数据量化成果，例如「优化接口让响应速度提升30%」。<|im_end|>",
    # 原始样本
    "<|im_start|>user\n应届生简历没工作经验怎么办？<|im_end|>\n<|im_start|>assistant\n突出校园项目、实习经历、竞赛获奖、自学成果，比如「参与XX课程设计，用Python实现XX功能」，体现学习能力和潜力。<|im_end|>",
    # 同义改写样本
    "<|im_start|>user\n应届生没有工作经验，简历该怎么写？<|im_end|>\n<|im_start|>assistant\n应届生可重点突出校园项目、实习经历、竞赛奖项或自学成果，比如「参与XX课程设计，用Python开发XX功能」，以此体现学习能力和潜力。<|im_end|>"
]
# 2. 训练参数
EPOCHS = 3          # 训练轮次
BATCH_SIZE = 1      # CPU固定1
OUTPUT_DIR = "./custom_chat_model"  # 权重保存目录
# =========================================================================

# 1. 构建数据集
data = {"text": TRAIN_DATA}
dataset = Dataset.from_dict(data)

# 2. 加载模型和分词器
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# 3. LoRA配置
lora_config = LoraConfig(
    r=8, #控制 LoRA 矩阵的「大小」，决定微调的「信息量」：越小 → 微调参数越少（速度快、占空间小） 越大 → 微调参数越多（效果可能更好，但 CPU/GPU 压力大）
    lora_alpha=32, #控制 LoRA 微调的「强度」：越大 → 微调对模型的影响越强 越小 → 微调影响越弱
    target_modules=["q_proj", "v_proj"], #指定只微调模型中「注意力机制」的 2 个核心模块：q_proj：查询（Query）投影层 v_proj：值（Value）投影层
    lora_dropout=0.05, # 训练时随机「关掉」5% 的 LoRA 参数，避免模型「死记硬背」（过拟合）
    bias="none", # 「偏置参数」是否参与微调："none"：不微调（默认） "all"：全微调 "lora_only"：只微调 LoRA 的偏置
    task_type="CAUSAL_LM" # 任务场景："CAUSAL_LM"：因果语言模型（生成式任务，比如问答、文本生成） "SEQ_CLS"：文本分类 "QUESTION_ANSWERING"：抽取式问答
)
"""
# LoRAConfig常用扩展参数（新手进阶用，代码中未启用，调优时参考）：
# 1. 控制微调范围：
# - modules_to_save：指定除LoRA模块外额外保存的模型模块（适配微调分词器/输出层、自定义词汇）
# - layers_to_transform：指定只微调模型某几层（如前10层，7B/13B大模型减参时用）
# - rank_pattern：给不同模块设不同r值（如q_proj=8、v_proj=4，精细化调优用，新手不用）
# 2. 训练稳定性：
# - inference_mode=False：训练设False/推理设True（推理设True可提速）
# - use_rslora=False：是否用RS-LoRA（改进版更稳定，7B+模型可设True，小模型不用）
# - loftq_config=None：低精度微调配置（4bit/8bit训练，GPU显存不足时用，CPU没必要）
# 3. 兼容适配：
# - fan_in_fan_out=False：适配输入输出维度不同的模块（仅微调mlp模块时设True）
# - merge_weights=False：合并LoRA与基础模型权重（推理提速用，合并后权重变大）
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
"""

model = get_peft_model(model, lora_config)
print(f"可训练参数占比：{model.print_trainable_parameters()}")

# 4. 分词函数
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 5. 训练参数（固定最优配置）
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=False,
    report_to="none",
    fp16=False,
    bf16=False,
)

# 6. 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("开始训练自定义问答模型...")
trainer.train()
# 手动保存
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成！权重已保存到 {OUTPUT_DIR}")
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, trainer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 初级训练配置对象，设置各种训练超参数
train_args = SFTConfig(
    per_device_train_batch_size=2,          # 每个设备上训练的批次大小
    gradient_accumulation_steps=4,          # 梯度累积步数
    max_steps=100,                          # 最大训练步数
    learning_rate=2e-4,                     # 学习率
    warmup_steps=10,                        # 预热步数
    logging_steps=10,                       # 每多少步打印一次日志
    optim="adamw_torch",                    # 优化器
    weight_decay=0.01,                      # 权重衰减
    lr_scheduler_type="linear",             # 学习率调度器类型
    seed=929,                               # 随机种子
    report_to="none",                       # 不报告指标到任何平台
)

import get_dataset

# 初始化 训练器，传入模型、数据收集器、训练数据集、训练参数
trainer = SFTTrainer(
    model=model,                                # 模型
    processing_class=tokenizer,                 # 文本的预处理和编码
    train_dataset=get_dataset.train_dataset,    # 训练数据集
    args=train_args,                            # 训练参数
)

# 开始执行训练，返回训练结果统计信息
trainer_stats = trainer.train()

print(trainer_stats)

torch.save(trainer.model.state_dict(), "./saver/cat_peft.pth")


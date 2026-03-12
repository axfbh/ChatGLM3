"""
多语言对话微调脚本：基于 Qwen + LoRA 的对话数据微调。
"""
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
from tqdm import tqdm
from dataclasses import dataclass
import torch
import json
import os

# 在 fork（多进程 DataLoader / DDP）前禁用 tokenizers 内部并行，避免死锁与警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """训练与数据相关配置。"""
    model_name: str = "Qwen/Qwen3-1.7B"
    output_dir: str = "./qwen_lora_finetuned"
    data_path: str = "./data/conversation_dataset.jsonl"
    max_length: int = 384

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj", "k_proj")
    modules_to_save: tuple = ("embed_tokens", "lm_head")

    # 训练（双卡 3090：每卡 batch=12，梯度累积 2，有效 batch=12×2×2=48）
    per_device_train_batch_size: int = 12
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    num_train_epochs: int = 5
    warmup_ratio: float = 0.05  # 按总步数比例 warmup，比固定 steps 更合理
    max_grad_norm: float = 1.0

    # 日志与保存
    logging_steps: int = 10
    save_strategy: str = "epoch"
    dataloader_num_workers: int = 4


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config: TrainConfig):
    """加载基座模型与分词器，并应用 LoRA。"""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=list(config.modules_to_save),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 数据处理
# ---------------------------------------------------------------------------

def _normalize_role(msg: dict, prev_role: str | None) -> str:
    """将单条消息的角色规范为 user 或 assistant。"""
    role = msg.get("role", "unknown").lower()
    # 验证角色值
    if role in ("user", "assistant"):
        return role
    if prev_role is None or prev_role == "assistant":
        return "user"
    return "assistant"


def format_conversation(example: dict, tokenizer, max_length: int) -> dict | None:
    """带数据校验的对话格式处理"""
    messages = []
    prev_role = None

    for msg in example["conversations"]:
        content = msg.get("content", "")
        # 跳过无效消息
        if len(content.strip()) < 1:
            continue

        # 规范角色值
        role = _normalize_role(msg, prev_role)
        messages.append({"role": role, "content": content})
        prev_role = role

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        print(f"Template error: {e}")
        return None

    labels = []
    for msg in messages:
        content_ids = tokenizer.encode(
            msg["content"],
            add_special_tokens=False
        )
        if msg["role"] == "assistant":
            labels.extend(content_ids + [tokenizer.eos_token_id])
        else:
            labels.extend([-100] * (len(content_ids) + 1))

    return {"text": text, "labels": labels[:max_length]} if text else None


def load_dataset(path: str, tokenizer, max_length: int) -> Dataset:
    """从 JSONL 文件加载并格式化对话数据。"""
    data = []
    error_count = 0

    with open(path, "r", encoding="utf-8") as f:
        lines = list(f)

    for line_idx, line in enumerate(tqdm(lines, desc="Loading dataset")):
        try:
            # 原始数据加载
            raw_data = json.loads(line)

            # 处理修正后的对话
            formatted = format_conversation(raw_data, tokenizer, max_length)
            if formatted and len(formatted["text"]) > 10:
                data.append(formatted)
            else:
                print(f"跳过无效对话：第 {line_idx + 1} 行")

        except Exception as e:
            error_count += 1
            print(f"错误处理第 {line_idx + 1} 行：{e}")
            if error_count > 10:
                raise RuntimeError("发现过多错误，请先修正数据格式")

    print(f"成功加载 {len(data)} 条有效数据（跳过 {error_count} 条无效数据）")
    return Dataset.from_list(data)


def preprocess_function(examples: dict, tokenizer, max_length: int) -> dict:
    """批预处理：tokenize 并构造 labels。"""
    tokenized = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    labels = torch.full(
        (len(examples["text"]), max_length),
        fill_value=-100,
        dtype=torch.long,
    )
    for i, lbl in enumerate(examples["labels"]):
        labels[i, :len(lbl)] = torch.LongTensor(lbl[:max_length])

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------

def main():
    config = TrainConfig()

    model, tokenizer = load_model_and_tokenizer(config)

    dataset = load_dataset(config.data_path, tokenizer, config.max_length)
    processed_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_length),
        batched=True,
        batch_size=32,
        remove_columns=["text", "labels"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=True,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        report_to="none",
        output_dir=config.output_dir,
        save_safetensors=True,
        dataloader_num_workers=config.dataloader_num_workers,
        ddp_find_unused_parameters=False,  # PEFT 多卡时建议关闭，避免 DDP 报错
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    trainer.model.save_pretrained(
        config.output_dir,
        safe_serialization=True,
        save_embedding_layers=True,
    )
    print(f"模型已保存至 {config.output_dir}")


if __name__ == "__main__":
    main()

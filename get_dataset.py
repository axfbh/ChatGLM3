from datasets import load_dataset, Dataset

# 加载原始 JSON 格式的数据
raw_ds = load_dataset(
    "json",
    data_files={"train": "data/cat.json"},
    split="train"
)

# 使用 prompt + completion 格式（每条为对话列表），SFTTrainer 会根据 token 长度差自动生成 completion_mask，
prompts = [
    [{"role": "user", "content": item["instruction"]}]
    for item in raw_ds
]
completions = [
    [{"role": "assistant", "content": item["output"]}]
    for item in raw_ds
]

train_dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})

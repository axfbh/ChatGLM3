import torch
from peft import PeftModel
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
output_dir = "./qwen_lora_finetuned"
max_length = 384

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False  # 梯度检查点需要关闭cache
)

print("base_model load successful!")

model = PeftModel.from_pretrained(model, output_dir)

messages = [
    {"role": "user", "content": "要买一把茶刀看到有零食就拍了点"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=max_length,
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(responses)

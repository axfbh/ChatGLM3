import torch

from modelscope import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.load_state_dict(torch.load("./saver/cat_peft.pth"))


def ask_catgirl(question):

    messages = [
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # 思考模式
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    from transformers import TextStreamer
    _ = model.generate(
        **inputs,
        max_new_tokens=256,  # 输出长度
        temperature=0.7, 
        top_p=0.8, 
        top_k=20,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


if __name__ == '__main__':
    ask_catgirl("你是谁呀")

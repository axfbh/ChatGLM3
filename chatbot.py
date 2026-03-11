import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-1.7B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # quantization_config=BitsAndBytesConfig(load_in_8bit=True),  # 4GB 显存
        )
        self.history = []   

    def generate_response(self, user_input):
        message = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,  # 4GB 显存降低生成长度
        )[0][len(inputs.input_ids[0]):].tolist()

        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 更新历史
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        return response

# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot("./Qwen/Qwen3-1.7B")

    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    # Second input with /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}")
    print("----------------------")

    # Third input with /think
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")

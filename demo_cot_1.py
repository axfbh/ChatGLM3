from openai import OpenAI

client = OpenAI(
    api_key="sk-5bfb31a9765849beb9c8068fbb24e933",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一个用于旅途规划的智能助手"},
        {"role": "user", "content": "给我一个杭州旅游一日游的规划。从火车站出发，晚上返回火车站"},
    ]
)

print(completion.choices[0].message.content)

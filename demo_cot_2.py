from openai import OpenAI

# 创建系统提示 - 使用三重引号避免f-string问题（f-string问题是指在使用 Python的f-string（格式化字符串字面量）时遇到的语法错误、运行时错误或使用不当的情况）
system_prompt = f"""
**系统角色设定：旅游规划专家**

你是一个专业的旅游规划助手。核心任务是根据用户需求**分步骤**构建旅游规划方案，并**仅以JSON格式**输出规划步骤信息。
"""

client = OpenAI(
    api_key="sk-5bfb31a9765849beb9c8068fbb24e933",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "给我一个杭州旅游一日游的规划。从火车站出发，晚上返回火车站"},
    ]
)

print(completion.choices[0].message.content)

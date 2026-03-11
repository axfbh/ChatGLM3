from openai import OpenAI
import json
from datetime import datetime

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-5bfb31a9765849beb9c8068fbb24e933",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def get_current_weather(city):
    weather_data = {
        "北京市": {"temperature": "20°C", "condition": "晴天"},
        "上海市": {"temperature": "18°C", "condition": "多云"},
        "广州市": {"temperature": "25°C", "condition": "雨天"},
    }
    return weather_data.get(city, {"temperature": "未知", "condition": "未知"})


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [
    # 获取当前时刻的时间
    {
        "type": "function",
        "function": {
                "name": "get_current_time",
                "description": "当你想知道现在时间时非常有用。",
                "parameters": {}
        }
    },
    # 获取指定城市天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

messages = [{"role": "user", "content": input("请输入问题：")}]


completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-plus",
    messages=messages,
    extra_body={
        "enable_thinking": True,
    },
    tools=tools,
    parallel_tool_calls=True,
    stream=True,
)

reasoning_content = ""          # 定义完整思考过程
answer_content = ""             # 定义完整回复
tool_info = []                  # 存储工具调用信息
is_answering = False            # 判断是否结束思考过程并开始回复

print("="*20+"思考过程"+"="*20)
for chunk in completion:
    if not chunk.choices:
        # 处理用量统计信息
        print("\n"+"="*20+"Usage"+"="*20)
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # 处理 AI 的思考过程
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content
            print(delta.reasoning_content, end="", flush=True)
        # 处理最终回复内容
        else:
            if not is_answering:  # 首次进入回复阶段时打印
                is_answering = True
                print("\n"+"="*20+"回复内容"+"="*20)

            if delta.content is not None:
                answer_content += delta.content
                print(delta.content, end="", flush=True)  # 流式输出回复内容

            # 处理工具调用信息
            if delta.tool_calls is not None:
                for tool_call in delta.tool_calls:
                    index = tool_call.index

                    while len(tool_info) <= index:
                        tool_info.append({})

                    # 收集工具调用 ID （后续调用）
                    if tool_call.id:
                        tool_info[index]["id"] = tool_info[index].get(
                            'id', '') + tool_call.id

                    # 收集函数名 （后续路由到具体函数）
                    if tool_call.function and tool_call.function.name:
                        tool_info[index]["name"] = tool_info[index].get(
                            'name', '') + tool_call.function.name

                    # 收集函数参数  （JSON字符串格式，需要后续解析）
                    if tool_call.function and tool_call.function.arguments:
                        tool_info[index]["arguments"] = tool_info[index].get(
                            'arguments', '') + tool_call.function.arguments

print(f"\n"+"="*20+"工具调用信息"+"="*20)
if not tool_info:
    print("没有工具调用")
else:
    for info in tool_info:
        print(f"工具ID: {info.get('id', '')}")
        print(f"工具名称: {info.get('name', '')}")
        print(f"工具参数: {info.get('arguments', '')}")
        function_name = info.get('name', '')
        print(function_name)

        if function_name == "get_current_weather":
            city = json.loads(info.get('arguments', '{}'))
            function_result = get_current_weather(city['location'])
            print(f"工具结果: {function_result}")

        if function_name == "get_current_time":
            function_result = get_current_time()
            print(f"工具结果: {function_result}")

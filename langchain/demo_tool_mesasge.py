from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage

# 百炼 OpenAI 兼容接口；ChatOpenAI 默认读 OPENAI_API_KEY，这里统一支持 DASHSCOPE_API_KEY
_DASHSCOPE_COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "sk-5bfb31a9765849beb9c8068fbb24e933"


model = ChatOpenAI(
    model="qwen-plus",
    api_key=api_key,
    base_url=_DASHSCOPE_COMPAT_BASE,
    temperature=0.5,
    max_tokens=1000
)


model = create_agent(model=model)

# 模型进行工具调用后
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# 执行工具并创建结果消息
weather_result = "Sunny, 72°F"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # 必须匹配调用 ID
)

# 继续对话
messages = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,  # 模型的工具调用
    tool_message,  # 工具执行结果
]

response = model.invoke(messages)  # 模型处理结果

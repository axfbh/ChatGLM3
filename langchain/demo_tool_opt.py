from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

# 百炼 OpenAI 兼容接口；ChatOpenAI 默认读 OPENAI_API_KEY，这里统一支持 DASHSCOPE_API_KEY
_DASHSCOPE_COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "sk-5bfb31a9765849beb9c8068fbb24e933"


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"


@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str


@dataclass
class ResponseFormat:
    """代理的响应模式。"""
    # 带双关语的回应（始终必需）
    punny_response: str
    # 天气的任何有趣信息（如果有）
    weather_conditions: str | None = None


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。

你可以使用两个工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""

checkpointer = InMemorySaver()

model = ChatOpenAI(
    model="qwen-plus",
    api_key=api_key,
    base_url=_DASHSCOPE_COMPAT_BASE,
    temperature=0.5,
    max_tokens=1000
)

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather_for_location, get_user_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    out = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "外面的天气怎么样?"}
            ]
        },
        config=config,
        context=Context(user_id="1")
    )

    print(out['structured_response'])

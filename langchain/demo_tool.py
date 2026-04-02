from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 百炼 OpenAI 兼容接口；ChatOpenAI 默认读 OPENAI_API_KEY，这里统一支持 DASHSCOPE_API_KEY
_DASHSCOPE_COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "sk-5bfb31a9765849beb9c8068fbb24e933"


def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"

model = ChatOpenAI(
    model="qwen-plus",
    api_key=api_key,
    base_url=_DASHSCOPE_COMPAT_BASE,
    temperature=0.5,
    max_tokens=1000
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手",
)

if __name__ == "__main__":
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "旧金山的天气怎么样"}]}
    )
    print(out)

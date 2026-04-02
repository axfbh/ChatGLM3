from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  # [!code highlight]

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

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

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),  # [!code highlight]
)

out = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},  # [!code highlight]
)

print(out['messages'][-1])
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver  # [!code highlight]
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import SummarizationMiddleware

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
    tools=[],
    system_prompt="Please be concise and to the point.",
    middleware=[
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=4000,  # Trigger summarization at 4000 tokens
            messages_to_keep=20,  # Keep last 20 messages after summary
        )
    ],
    checkpointer=InMemorySaver(),  # [!code highlight]
)


config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
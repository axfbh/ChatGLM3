from typing import override
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

import uvicorn

class HelloWorldAgent:
    async def invoke(self) -> str:
        return 'Hello World!你好 世界'


class HelloWorldAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = HelloWorldAgent()

    # 处理预期响应事件或事件流的传入请求
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        执行 agent 的主要方法

        参数：
            context:  请求上下文对象，包含请求的详细信息
            event_queue:  事件队列，用于将执行结果放入队列

        返回：
            None
        """
        result = await self.agent.invoke()
        # 将结果包装成文本信息事件并异步放入事件队列
        await event_queue.enqueue_event(new_agent_text_message(result))

    # 处理取消正在进行任务的请求
    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception('cancel not supported')


class LoggingA2AStarletteApplication(A2AStarletteApplication):
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            method = scope["method"]
            print(f"Received request: {method} {path}")
        return await super().__call__(scope, receive, send)

# 创建技能和代理卡
skill = AgentSkill(
    id='hello_world',
    name='Returns hello world',
    description='just returns hello world',
    tags=['hello world'],

)

extended_skill = AgentSkill(
    id='super_hello_world',
    name='Returns a SUPER Hello World',
    description='A more enthusiastic greeting',
    tags=['hello world', 'super', 'extended'],
)

public_agent_card = AgentCard(
    name='Hello World Agent',
    description='Just a hello world agent',
    url='http://localhost:9000/',
    version='1.0.0',
    defaultInputModes=['text'],
    defaultOutputModes=['text'],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],
    supportsAuthenticatedExtendedCard=True,
)

specific_extended_agent_card = public_agent_card.model_copy(
    update={
        'name': 'Hello World Agent - Extended Edition',
        'description': 'Full-featured hello world agent',
        'version': '1.0.1',
        'skills': [skill, extended_skill],
    }
)

# 创建请求处理器和服务器
request_handler = DefaultRequestHandler(
    agent_executor=HelloWorldAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

server = LoggingA2AStarletteApplication(
    agent_card=public_agent_card,
    http_handler=request_handler,
    extended_agent_card=specific_extended_agent_card,
)

if __name__ == "__main__":
    print("Starting A2A server on http://0.0.0.0:9000")
    uvicorn.run(server.build(), host='0.0.0.0', port=9000)

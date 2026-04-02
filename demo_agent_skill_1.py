import uvicorn

from a2a.types import AgentSkill
from a2a.types import AgentCard, AgentCapabilities
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication


skill = AgentSkill(
    id="hello-world",
    name="Hello World Skill",
    description="Just a hello world skill",
    tags=["hello world"],
    examples=["hi", "hello world"]
)

agent_card = AgentCard(
    name="Hello World Agent",
    description="Just a hello world agent",
    url="http:/0.0.0.0:9000",
    version="0.0.1",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streams=True, tools=True),
    skills=[skill],
)

#构建服务器
server = A2AStarletteApplication(
    agent_card=agent_card ,http_handler=None
)

uvicorn.run(server.build(),host='0.0.0.0', port=9000)
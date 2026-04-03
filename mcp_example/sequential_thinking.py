import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
)

async def get_mcp_tools(server_params):
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 获取工具列表响应
                tools_response = await session.list_tools()
    finally:
        await asyncio.sleep(0.1)
    tools = tools_response.tools
    return tools

if __name__ == "__main__":
    tools = asyncio.run(get_mcp_tools(server_params))
    for tool in tools:
        print(tool)
        print("----------------------------")

import logging
from typing import Any
from uuid import uuid4
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)

async def main() -> None:
    # 使用localhost而非0.0.0.0
    base_url = 'http://localhost:9000'
    PUBLIC_AGENT_CARD_PATH = '/.well-known/agent.json'
    EXTENDED_AGENT_CARD_PATH = '/agent/authenticatedExtendedCard'

    async with httpx.AsyncClient() as httpx_client:
        # 初始化解析器
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        # 尝试获取代理卡
        _public_card = await resolver.get_agent_card()
        final_agent_card_to_use = _public_card
        print("------------------------")

        # 初始化客户端并发送消息
        client = A2AClient(httpx_client=httpx_client, agent_card=final_agent_card_to_use)

        request_id = uuid4().hex
        # 发送非流式消息
        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': 'hello'}
                ],
                'messageId': request_id,
            },
        }
        request = SendMessageRequest(
            id=request_id, params=MessageSendParams(**send_message_payload)
        )

        response = await client.send_message(request)
        print(response.model_dump(mode='json', exclude_none=True))

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

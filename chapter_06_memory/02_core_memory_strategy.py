import asyncio
from scratch_agents.agents.tool_calling_agent_ch6 import ToolCallingAgent
from scratch_agents.models.openai import OpenAILlm
from scratch_agents.sessions.in_memory_session_manager import InMemorySessionManager
from scratch_agents.memory.core_memory_strategy import CoreMemoryStrategy
from dotenv import load_dotenv
import os

load_dotenv()


async def test_core_memory_loading():
    user_id = "test_123"
    session_id = "test_session"
    session_manager = InMemorySessionManager()
    session = session_manager.get_or_create_session(session_id, user_id)
    session.core_memory["user"] = "User's name is Alice"
    
    agent = ToolCallingAgent(
        name="memory_agent",
        model=OpenAILlm(model="gpt-5-mini"),
        instructions="You are a helpful assistant",
        session_manager=session_manager,
        before_llm_callbacks=[CoreMemoryStrategy()]
    )
    
    response = await agent.run(
        "What's my name?",
        session_id=session_id,
        user_id=user_id
    )
    
    print(response)

asyncio.run(test_core_memory_loading())
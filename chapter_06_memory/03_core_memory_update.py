from scratch_agents.agents.tool_calling_agent_ch6 import ToolCallingAgent
from scratch_agents.models.openai import OpenAILlm
from scratch_agents.sessions.in_memory_session_manager import InMemorySessionManager
from scratch_agents.tools.core_memory_upsert import core_memory_upsert
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

user_id = "test_123"
session_id = "test_session"

async def test_automatic_memory_update():
    agent = ToolCallingAgent(
        name="learning_agent",
        model=OpenAILlm(model="gpt-5-mini"),
        instructions="Remember important user info with core_memory_upsert",
        tools=[core_memory_upsert],
        session_manager=InMemorySessionManager(),
    )
    
    await agent.run(
        "Hi! My name is Alice and I work as a data scientist.",
        session_id=session_id,
        user_id=user_id
    )
    
    session = agent.session_manager.get_session(session_id)
    print(session.core_memory['user'])

asyncio.run(test_automatic_memory_update())

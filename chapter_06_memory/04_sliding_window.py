import asyncio
from scratch_agents.agents.tool_calling_agent_ch6 import ToolCallingAgent
from scratch_agents.models.openai import OpenAILlm
from scratch_agents.sessions.in_memory_session_manager import InMemorySessionManager
from scratch_agents.memory.sliding_window_strategy import SlidingWindowStrategy
from scratch_agents.types.contents import Message
from scratch_agents.types.events import Event
from dotenv import load_dotenv
import os

load_dotenv()

user_id = "test_123"
session_id = "test_session"

async def test_sliding_window():
    
    session_manager = InMemorySessionManager()
    session = session_manager.create_session(session_id, user_id)
    
    session.events.append(Event(
        execution_id="exec1",
        author="user",
        content=[Message(role="user", content="My name is Alice"),
        Message(role="user", content="I live in Korea")]
    ))
    
    agent = ToolCallingAgent(
        name="window_agent",
        model=OpenAILlm(model="gpt-5-mini"),
        instructions="You are a helpful assistant",
        session_manager=session_manager,
        before_llm_callbacks=[SlidingWindowStrategy(max_messages=2)]  
    )
    
    response = await agent.run(
        "What's my name?",  
        session_id=session_id,
        user_id=user_id
    )
    print(response)

asyncio.run(test_sliding_window())
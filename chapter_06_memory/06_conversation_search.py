from scratch_agents.agents.tool_calling_agent_ch6 import ToolCallingAgent
from scratch_agents.models.openai import OpenAILlm
from scratch_agents.sessions.in_memory_session_manager import InMemorySessionManager
from scratch_agents.memory.sliding_window_strategy import SlidingWindowStrategy
from scratch_agents.tools.conversation_search import conversation_search
from scratch_agents.types.contents import Message
from scratch_agents.types.events import Event
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

user_id = "test_123"
session_id = "test_session"

async def test_search_with_sliding_window():
    """Demonstrate search recovering information lost to sliding window"""
    
    model = OpenAILlm(model="gpt-5-mini")
    session_manager = InMemorySessionManager()
    session = session_manager.create_session(session_id, user_id)
    
    conversation_history = [
        ("user", "My golden retriever puppy is named Max."),
        ("assistant", "Max is a lovely name for a golden retriever!"),
        ("user", "He loves playing fetch in the park."),
        ("assistant", "That's wonderful! Golden retrievers are great at fetch."),
    ]
    
    for role, content in conversation_history:
        event = Event(
            execution_id="pre_loaded",
            author=role,
            content=[Message(role=role, content=content)]
        )
        session.events.append(event)  
    
    agent = ToolCallingAgent(
        name="search_agent",
        model=model,
        instructions="""You are a helpful assistant. When asked about 
        information from earlier in our conversation, use the 
        conversation_search tool to find it.""",
        tools=[conversation_search],
        session_manager=session_manager,
        before_llm_callbacks=[
            SlidingWindowStrategy(max_messages=2)
        ]
    )
    
    response = await agent.run(
        "What was my puppy's name?",
        session_id=session_id,
        user_id=user_id
    )
    print(f"Agent: {response}\n")

asyncio.run(test_search_with_sliding_window())
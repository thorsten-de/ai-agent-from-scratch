from scratch_agents.agents.tool_calling_agent_ch6 import ToolCallingAgent
from scratch_agents.models.openai import OpenAILlm
from scratch_agents.sessions.in_memory_session_manager import InMemorySessionManager
from scratch_agents.memory.summarization_strategy import SummarizationStrategy
from scratch_agents.types.contents import Message
from scratch_agents.types.events import Event
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

user_id = "test_123"
session_id = "test_session"

async def test_summarization_strategy():
    """Demonstrate summarization strategy in action"""
    
    model = OpenAILlm(model="gpt-5-mini")
    session_manager = InMemorySessionManager()
    session = session_manager.create_session(session_id, user_id)
    
    messages = [
        Message(role="user", content="Hi, I'm Bob"),
        Message(role="assistant", content="Nice to meet you, Bob!"),
        Message(role="user", content="I work as a teacher"),
        Message(role="assistant", content="Wow! What subject?"),
        Message(role="user", content="I teach math"),
        Message(role="assistant", content="Math is important!"),
        Message(role="user", content="I have 30 students"),
        Message(role="assistant", content="That's a good class size"),
    ]
    
    for msg in messages:
        event = Event(
            execution_id="test_exec",
            author="test",
            content=[msg]
        )
        session.events.append(event)
    
    agent = ToolCallingAgent(
        name="summary_agent",
        model=model,
        instructions="You are a helpful assistant",
        session_manager=session_manager,
        before_llm_callbacks=[
            SummarizationStrategy(model=model, trigger_count=8, keep_recent=2)
        ]
    )
    
    response = await agent.run(
        "What subject do I teach?",
        session_id=session_id,
        user_id=user_id
    )
    
    if "conversation_summary" in session.state:
        print(f"Summary: {session.state['conversation_summary']}")
        print(f"Summary Index: {session.state['last_summarized_index']}")
    
    print(f"\nAgent response: {response}")

asyncio.run(test_summarization_strategy())

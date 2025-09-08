import asyncio
from scratch_agents.models.openai import OpenAILlm
from scratch_agents.agents.tool_calling_agent_ch6 import ToolCallingAgent
from scratch_agents.tools import calculator, search_web
from scratch_agents.sessions.in_memory_session_manager import InMemorySessionManager
from dotenv import load_dotenv
import os

load_dotenv()


async def main():
    """Demonstrate session memory functionality"""
    user_id = "test_123"
    # Initialize components
    model = OpenAILlm(model='gpt-5-mini')
    tools = [calculator, search_web]
    
    # Create agent with session manager
    agent = ToolCallingAgent(
        name="session_assistant",
        model=model,
        instructions="You are a helpful assistant that remembers our conversations.",
        tools=tools,
        session_manager=InMemorySessionManager()
    )
    
    # First interaction - session 1
    print("=== First Interaction (Session 1) ===")
    answer1 = await agent.run(
        "My name is Alice and I'm working on Project Alpha. What's 123 * 456?",
        session_id="session_1",
        user_id=user_id
    )
    print(f"Assistant: {answer1}\n")
    
    # Second interaction - continue session 1
    print("=== Second Interaction (Session 1) ===")
    answer2 = await agent.run(
        "What project am I working on and what was the result of the multiplication I asked about?",
        session_id="session_1",
        user_id=user_id
    )
    print(f"Assistant: {answer2}\n")
    
    # New session - session 2
    print("=== New Session (Session 2) ===")
    answer3 = await agent.run(
        "Do you remember my name?",
        session_id="session_2",
        user_id=user_id
    )
    print(f"Assistant: {answer3}\n")

if __name__ == "__main__":
    asyncio.run(main())
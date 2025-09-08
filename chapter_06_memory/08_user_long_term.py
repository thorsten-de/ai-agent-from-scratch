from scratch_agents.agents.execution_context_ch6 import ExecutionContext
from scratch_agents.agents.tool_calling_agent_ch6 import ToolCallingAgent
from scratch_agents.models.openai import OpenAILlm
from scratch_agents.sessions.user_cross_session_manager import UserCrossSessionManager
from scratch_agents.sessions.in_memory_session_manager import InMemorySessionManager
from scratch_agents.tools.base_tool import BaseTool
from scratch_agents.models.llm_request import LlmRequest
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

user_id = "test_user_123"

async def user_memory_save_callback(context: ExecutionContext):
    """Callback to save user memories after each interaction"""
    cross_session_manager = context.cross_session_manager
    session = context.session
    execution_id = context.execution_id
    
    await cross_session_manager.process_session(session=session, execution_id=execution_id)
    
class UserMemorySearchTool(BaseTool):
    """Tool to search and retrieve user memories"""
    async def execute(self, context, **kwargs):
        return None
    
    async def process_llm_request(self, request: LlmRequest, context: ExecutionContext):
        user_id = context.session.user_id
        # Get all existing memories for the user
        all_memories = await context.cross_session_manager.find_existing([], user_id)
        if all_memories:
            memory_contents = [mem['content'] for mem in all_memories]
            memory_text = "\n".join(f"- {content}" for content in memory_contents)
            request.add_instructions(f"You have the following memories about this user:\n{memory_text}\n\nUse these memories to personalize your responses.")


async def test_user_long_term_memory():
    """Test user long-term memory with location updates"""
    
    session_manager = InMemorySessionManager()
    model = OpenAILlm(model="gpt-4o-mini")
    cross_session_manager = UserCrossSessionManager(model=model)
    
    memory_search_tool = UserMemorySearchTool()
    
    agent = ToolCallingAgent(
        name="user_memory_agent",
        model=model,
        instructions="You are a helpful assistant that remembers information about the user. Have natural conversations and acknowledge what you know about the user when relevant.",
        tools=[memory_search_tool],
        session_manager=session_manager,
        cross_session_manager=cross_session_manager,
        after_run_callbacks=[user_memory_save_callback]
    )
    
    print("=== Testing User Long-term Memory ===\n")
    
    # Test conversation about location changes
    test_conversations = [
        "Hi! I'm living in New York City. I love the energy here!",
        "Actually, I just moved to Los Angeles last month. The weather is so much better here.",
        "What do you remember about where I live?"
    ]
    
    for i, message in enumerate(test_conversations, 1):
        print(f"\n--- Conversation {i} ---")
        print(f"User: {message}")
        session_id = f"user_session_{i}"
        
        response = await agent.run(
            message,
            session_id=session_id,
            user_id=user_id
        )
        print(f"Assistant: {response}")
        
        # Show current memories in the database with timestamps
        print("\n=> Current User Memories:")
        memories = await cross_session_manager.find_existing([], user_id)
        if memories:
            for mem in memories:
                created = mem.get('created_at', 'Unknown')[:19] if mem.get('created_at') != 'Unknown' else 'Unknown'
                updated = mem.get('updated_at', 'Unknown')[:19] if mem.get('updated_at') != 'Unknown' else 'Unknown'
                print(f"  - {mem['content']}")
                if created != updated:
                    print(f"    (Created: {created}, Updated: {updated})")
                else:
                    print(f"    (Created: {created})")
        else:
            print("  (No memories yet)")
        
        # Small delay to see the progression
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_user_long_term_memory())
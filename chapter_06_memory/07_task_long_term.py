from scratch_agents.agents.execution_context_ch6 import ExecutionContext
from scratch_agents.agents.tool_calling_agent_ch6 import ToolCallingAgent
from scratch_agents.models.openai import OpenAILlm
from scratch_agents.sessions.task_cross_session_manager import TaskCrossSessionManager
from scratch_agents.sessions.in_memory_session_manager import InMemorySessionManager
from scratch_agents.tools.base_tool import BaseTool
from scratch_agents.models.llm_request import LlmRequest
from scratch_agents.tools.search_web import search_web
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

user_id = "test_123"

async def long_term_memory_save_callback(context:ExecutionContext):
    cross_session_manager = context.cross_session_manager
    session = context.session
    execution_id = context.execution_id
    
    await cross_session_manager.process_session(session=session, execution_id=execution_id)
    
class MemorySearchTool(BaseTool):
    async def execute(self, context, **kwargs):
        return None
    
    async def process_llm_request(self, request: LlmRequest, context: ExecutionContext):
        user_input = context.user_input
        user_id = context.session.user_id
        results = await context.cross_session_manager.search(user_input, user_id)
        if results:
            request.add_instructions(f"Use the following task memory to answer the user's question: {results}")


async def test_long_term_memory_save():
    """Test long-term memory saving with a meaningful conversation"""
    
    session_manager = InMemorySessionManager()
    model = OpenAILlm(model="gpt-5-mini")
    cross_session_manager = TaskCrossSessionManager(model=model)
    
    memory_search_tool = MemorySearchTool()
    
    agent = ToolCallingAgent(
        name="memory_agent",
        model=model,
        instructions="You are a helpful assistant. Have a natural conversation and learn about the user's task. IMPORTANT: When the user asks about a specific term or technology, use the search results to provide a comprehensive answer. Do NOT ask for clarification if you find relevant search results. Only ask for clarification if search returns no results or the query is truly impossible to understand. If multiple meanings exist, provide information about the most common or relevant one based on the search results.",
        tools=[search_web, memory_search_tool],
        session_manager=session_manager,
        cross_session_manager=cross_session_manager,
        after_run_callbacks=[long_term_memory_save_callback]
    )
    
    print("=== Testing Long-term Memory Save ===\n")
    
    test_conversations = [
       "What is Mem0?",
       "How does mem0 work?"
    ]
    
    for i, message in enumerate(test_conversations, 1):
        print(f"User: {message}")
        session_id = f"test_session_{i}"
        
        response = await agent.run(
            message,
            session_id=session_id,
            user_id=user_id
        )
        print(response)
        # print(cross_session_manager.collection.peek())

if __name__ == "__main__":
    asyncio.run(test_long_term_memory_save())


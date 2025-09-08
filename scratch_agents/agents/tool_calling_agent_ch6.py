from typing import List, Dict, Any, Optional
from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..types.contents import Message, ToolCall
from ..types.events import Event
from .execution_context_ch6 import ExecutionContext
from ..tools.base_tool import BaseTool
from ..types.contents import ToolResult
from typing import Type
from pydantic import BaseModel
from ..tools.decorator import tool
import inspect
from ..sessions.base_session_manager import BaseSessionManager
from ..sessions.in_memory_session_manager import InMemorySessionManager
from ..sessions.base_cross_session_manager import BaseCrossSessionManager

class ToolCallingAgent:
    def __init__(self, name: str, model: BaseLlm, 
                 tools: List[BaseTool] = [], 
                 instructions: str = "", 
                 max_steps: int = 10, 
                 output_type: Optional[Type[BaseModel]] = None,
                 before_llm_callbacks = None,
                 after_llm_callbacks = None,
                 before_tool_callbacks = None,
                 after_tool_callbacks = None,
                 after_run_callbacks = None,
                 session_manager: BaseSessionManager = None,
                 cross_session_manager: BaseCrossSessionManager = None):
        self.name = name
        self.model = model
        self.max_steps = max_steps
        self.instructions = instructions
        self.output_type: Optional[Type[BaseModel]] = output_type
        self.output_tool: Optional[str] = None
        self.tools = self._setup_tools(tools)
        self.before_llm_callbacks = before_llm_callbacks or []
        self.after_llm_callbacks = after_llm_callbacks or []
        self.before_tool_callbacks = before_tool_callbacks or []
        self.after_tool_callbacks = after_tool_callbacks or []
        self.after_run_callbacks = after_run_callbacks or []
        self.session_manager = session_manager or InMemorySessionManager()  
        self.cross_session_manager = cross_session_manager
        
    def _setup_tools(self, tools: List[BaseTool]):
        if self.output_type is not None:
            @tool(name="final_answer", description="Return the final structured answer matching the required schema.")
            def final_answer(output: self.output_type) -> self.output_type:
                return output
            tools.append(final_answer)
            self.output_tool = final_answer.name
        return {t.name: t for t in tools}
        
    async def think(self, context: ExecutionContext, llm_request: LlmRequest):
        for callback in self.before_llm_callbacks:
            result = callback(context, llm_request)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        
        llm_response = await self.model.generate(llm_request)
        
        for callback in self.after_llm_callbacks:
            result = callback(context, llm_response)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result 
        
        return llm_response

    async def _execute_tool(self, context: ExecutionContext, tool_name: str, tool_input: dict) -> Any:
        """Execute a tool with context injection if needed"""
        tool = self.tools[tool_name]
        
        # All tools now handle context properly in their execute method
        return await tool.execute(context, **tool_input)
    
    async def act(self, context: ExecutionContext, tool_calls: List[ToolCall]):
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_input = tool_call.arguments
            print(f"  → Calling {tool_name} with {tool_input}")

            # Step 1: before_tool_callbacks - can skip tool execution
            tool_response = None
            for callback in self.before_tool_callbacks:
                result = callback(context, tool_call)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    tool_response = result
                    break
            
            # Step 2: Execute tool if no callback provided result
            status = "success"
            if tool_response is None:
                try:
                    tool_response = await self._execute_tool(context, tool_name, tool_input)
                except Exception as e:
                    tool_response = str(e)
                    status = "error"
            
                # Step 3: after_tool_callbacks - only after actual tool execution
                for callback in self.after_tool_callbacks:
                    result = callback(context, tool_response)
                    if inspect.isawaitable(result):
                        result = await result
                    if result is not None:
                        tool_response = result
                        break
            
            # Step 4: Wrap in ToolResult at the end
            if tool_response is not None:
                tool_result = ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status=status,
                    content=str(tool_response),
                )
                tool_results.append(tool_result)

        return tool_results
    
    async def step(self, context: ExecutionContext):
        print(f"[Step {context.current_step + 1}]")
        llm_request = await self._prepare_llm_request(context)
        llm_response = await self.think(context, llm_request)
        if llm_response.error_message:
            raise RuntimeError(f"LLM error: {llm_response.error_message}")
        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            required_output_tool=self.output_tool or None,
            **llm_response.model_dump(),
        )
        context.add_event(response_event)
        
        if tool_calls := response_event.get_tool_calls():
            tool_results = await self.act(context, tool_calls)
            tool_results_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                required_output_tool=self.output_tool or None,
                content=tool_results,
            )
            context.add_event(tool_results_event)
            
        context.increment_step()
        
    async def run(self, user_input: str, 
                  user_id: str = None,
                  session_id: str = None):
        session = self.session_manager.get_or_create_session(session_id, user_id)
        context = ExecutionContext(
            user_input=user_input,
            session=session,
            session_manager=self.session_manager,
            cross_session_manager=self.cross_session_manager,
        )
        
        user_input_event = Event(
            execution_id=context.execution_id,
            author="user",
            content=[
                Message(
                    role="user",
                    content=user_input,
                )
            ],
        )
        context.add_event(user_input_event)
        
        while not context.final_result and context.current_step < self.max_steps:
            await self.step(context)
            
            last_event = context.events[-1]
            if last_event.is_final_response():
                context.final_result = self._extract_final_result(last_event)
                
        for callback in self.after_run_callbacks:
            result = callback(context)
            if inspect.isawaitable(result):
                await result
            
        return context.final_result
            
    async def _prepare_llm_request(self, context: ExecutionContext):
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)
            
        llm_request = LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools_dict={tool.name:tool for tool in self.tools.values() if tool.tool_definition},
        )
        
        for tool in self.tools.values():
            await tool.process_llm_request(llm_request, context)
            
        if self.output_tool:
            llm_request.tool_choice = "required"
        elif llm_request.tools_dict:
            llm_request.tool_choice = "auto"
        else:
            llm_request.tool_choice = None
            
        return llm_request
    
    def _extract_final_result(self, event: Event):
        if event.required_output_tool:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.status == "success"
                    and item.name == event.required_output_tool
                    and item.content
                ):
                    return item.content[0]
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
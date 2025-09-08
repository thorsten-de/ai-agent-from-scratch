from typing import Any, Dict, Type, Union, Optional
from abc import ABC, abstractmethod
import json
from .schema_utils import format_tool_definition
from ..agents.execution_context_ch6 import ExecutionContext
from ..models.llm_request import LlmRequest


class BaseTool(ABC):
    
    def __init__(
        self, 
        name: str = None, 
        description: str = None, 
        tool_definition: Optional[Union[Dict[str, Any], str]] = None,
        pydantic_input_model: Type = None,
        output_type: str = "str"
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self.pydantic_input_model = pydantic_input_model
        self.output_type = output_type
        
        if isinstance(tool_definition, str):
            self._tool_definition = json.loads(tool_definition)
        elif tool_definition is not None:
            self._tool_definition = tool_definition
        else:
            self._tool_definition = None  

    @property
    def tool_definition(self):
        if self._tool_definition is None:
            self._tool_definition = self._generate_definition()
        return self._tool_definition
    
    def _generate_definition(self):
        if self.pydantic_input_model:
            try:
                from pydantic import BaseModel
                if issubclass(self.pydantic_input_model, BaseModel):
                    parameters = self.pydantic_input_model.model_json_schema()
                    return format_tool_definition(
                        self.name, self.description, parameters
                    )
            except ImportError:
                pass
        else:
            return None
    
    async def __call__(self, **kwargs) -> Any:
        return await self.execute(**kwargs)
    
    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the execute method"
        )
        
    async def process_llm_request(self, request: LlmRequest, context: ExecutionContext):
        return None
    
    def to_code_prompt(self) -> str:
        """Generate tool description for code execution environment"""
        params_desc = ""
        if self._tool_definition and "function" in self._tool_definition:
            func_def = self._tool_definition["function"]
            if "parameters" in func_def and "properties" in func_def["parameters"]:
                params = []
                for param_name, param_info in func_def["parameters"]["properties"].items():
                    param_type = param_info.get("type", "Any")
                    param_desc = param_info.get("description", "")
                    required = param_name in func_def["parameters"].get("required", [])
                    req_str = " (required)" if required else " (optional)"
                    params.append(f"    - {param_name}: {param_type}{req_str} - {param_desc}")
                if params:
                    params_desc = "\n  Parameters:\n" + "\n".join(params)
        
        return f"""Tool: {self.name}
  Description: {self.description}
  Output Type: {self.output_type}{params_desc}"""
from .base_memory_strategy import MemoryStrategy
from ..models.llm_request import LlmRequest
from ..types.contents import Message

class SummarizationStrategy(MemoryStrategy):
    """Summarize old messages to preserve information while reducing tokens"""
    
    def __init__(self, model, trigger_count: int = 10, keep_recent: int = 3):
        self.model = model
        self.trigger_count = trigger_count  #A
        self.keep_recent = keep_recent  #B

    async def _generate_summary(self, messages_text: str):
        request = LlmRequest(
            instructions=[  #A
                "Summarize the following conversation concisely.",  #A
                "Preserve key facts, decisions, and important context.",  #A
                "Keep the summary under 200 words."  #A
            ],
            contents=[Message(role="user", content=messages_text)]  #B
        )
        
        response = await self.model.generate(request)  #C
        
        for item in response.content:  #D
            if isinstance(item, Message) and item.role == "assistant":  #D
                return item.content  #D
        
        return "Summary generation failed"  #E

    async def apply(self, context, llm_request):
        """Apply summarization when new messages since last summary exceed threshold"""
        contents = llm_request.contents
        
        messages_only = [item for item in contents if isinstance(item, Message)]  #A
        last_summarized = context.state.get("last_summarized_index", 0)
        
        total_messages = len(messages_only)  #B
        new_messages_count = total_messages - last_summarized  #B
        
        if new_messages_count < self.trigger_count:
            return None
        
        summarize_until = total_messages - self.keep_recent  #C
        to_summarize = messages_only[last_summarized:summarize_until]  #C
        to_keep = contents[-self.keep_recent:] if len(contents) >= self.keep_recent else contents  #C
        
        if not to_summarize:
            return None
        
        existing_summary = context.state.get("conversation_summary")
        
        summary_input = []  #D
        if existing_summary:  #D
            summary_input.append(f"Previous Summary:\n{existing_summary}\n")  #D
        
        summary_input.append("New Messages to Summarize:\n")  #D
        for msg in to_summarize:  #D
            summary_input.append(f"{msg.role}: {msg.content}")  #D
        
        messages_text = "\n".join(summary_input)  #D
        
        new_summary = await self._generate_summary(messages_text)  #E
        
        context.state["conversation_summary"] = new_summary
        context.state["last_summarized_index"] = summarize_until
        
        if new_summary:
            summary_instruction = f"[Previous Conversation Summary]\n{new_summary}"
            llm_request.add_instructions([summary_instruction])  #F
        
        llm_request.contents = to_keep  #G
        
        print(f"Compressed {len(to_summarize)} messages")
        print(f"Keeping {len(to_keep)} recent items")
        
        return None

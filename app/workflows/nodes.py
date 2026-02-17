"""LangGraph workflow node functions extracted from graph.py."""
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import os
from app.core.config import settings
from app.workflows.prompt_loader import (
    get_system_message,
    get_prompt,
    get_retriever_tool_config,
    get_settings
)

# Ensure OPENAI_API_KEY is set in environment
# Some langchain components check os.environ directly
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

# Load settings from prompts.json
_settings = get_settings()


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def create_workflow_nodes(
    retriever_tool: BaseTool,
    all_tools: Optional[List[BaseTool]] = None,
    model_name: str = None
):
    """Create workflow node functions with shared models.

    Args:
        retriever_tool: The retriever tool instance (used for document grading)
        all_tools: All available tools (retriever + custom tools). Defaults to [retriever_tool].
        model_name: Model name (defaults to value from prompts.json)
    """
    if all_tools is None:
        all_tools = [retriever_tool]

    # Use model from prompts.json if not specified
    if model_name is None:
        model_name = _settings.get('default_model', 'gpt-4o-mini')

    temperature = _settings.get('default_temperature', 0)
    streaming_enabled = _settings.get('streaming_enabled', True)

    # Initialize models with settings from prompts.json
    response_model = init_chat_model(
        model_name,
        temperature=temperature,
        streaming=streaming_enabled
    )
    grader_model = init_chat_model(model_name, temperature=temperature, streaming=False)
    
    def _get_latest_user_question(messages, exclude_last: bool = False):
        """Helper to find the most recent user message in the conversation.
        
        With checkpointing, messages[0] might be from a previous conversation turn.
        We need to find the actual current question being asked.
        
        Args:
            messages: List of messages from the state
            exclude_last: If True, exclude the last message (e.g., when it's a tool response)
        
        Returns:
            The content of the most recent user message
        """
        search_messages = messages[:-1] if exclude_last else messages
        
        for msg in reversed(search_messages):
            # Check for LangChain message objects
            if hasattr(msg, 'type') and msg.type == "human":
                return msg.content
            # Check for dict-style messages
            elif isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        
        # Fallback to first message if no user message found
        return messages[0].content if messages else ""
    
    def generate_query_or_respond(state: MessagesState):
        """Call the model to generate a response based on the current state.
        
        CRITICAL: In multi-turn conversations with checkpointing, we must ensure
        the LLM focuses on the LATEST user question, not previous ones in the thread.
        
        Also handles incomplete tool call sequences from corrupted checkpoints.
        """
        messages = state["messages"]
        
        # CRITICAL: Clean up ALL incomplete tool call sequences from corrupted checkpoints
        # OpenAI requires: AI message with tool_calls → MUST be followed by tool messages
        # Strategy: Skip any AI message with tool_calls that isn't immediately followed
        # by tool messages (or has tool messages but not for all tool_call_ids)
        
        cleaned_messages = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            msg_type = getattr(msg, 'type', None) or (msg.get('type') if isinstance(msg, dict) else None)
            msg_role = getattr(msg, 'role', None) or (msg.get('role') if isinstance(msg, dict) else None)
            
            # Check if this is an AI/assistant message
            is_ai = (msg_type == "ai") or (msg_role == "assistant")
            
            if is_ai:
                # Check if it has tool_calls
                tool_calls = None
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls = msg.tool_calls
                elif isinstance(msg, dict) and msg.get('tool_calls'):
                    tool_calls = msg.get('tool_calls')
                
                if tool_calls:
                    # Extract ALL tool_call_ids from this AI message
                    tool_call_ids = set()
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tc_id = tc.get('id')
                            if tc_id:
                                tool_call_ids.add(tc_id)
                        elif hasattr(tc, 'id'):
                            tool_call_ids.add(tc.id)
                    
                    # Safety check: If we couldn't extract tool_call_ids, skip to be safe
                    if not tool_call_ids:
                        # Can't verify - skip this message to prevent errors
                        i += 1
                        continue
                    
                    # Check if ALL tool_call_ids have corresponding tool responses
                    # Look ahead through remaining messages to find tool responses
                    found_responses = set()
                    for j in range(i + 1, len(messages)):
                        next_msg = messages[j]
                        next_type = getattr(next_msg, 'type', None) or (next_msg.get('type') if isinstance(next_msg, dict) else None)
                        next_role = getattr(next_msg, 'role', None) or (next_msg.get('role') if isinstance(next_msg, dict) else None)
                        
                        # Check if this is a tool message
                        is_tool = (next_type == "tool") or (next_role == "tool")
                        if is_tool:
                            # Extract tool_call_id
                            tool_call_id = None
                            if hasattr(next_msg, 'tool_call_id'):
                                tool_call_id = next_msg.tool_call_id
                            elif isinstance(next_msg, dict):
                                tool_call_id = next_msg.get('tool_call_id')
                            
                            if tool_call_id and tool_call_id in tool_call_ids:
                                found_responses.add(tool_call_id)
                        
                        # Stop looking if we hit another user message (new turn)
                        if (next_type == "human") or (next_role == "user"):
                            break
                    
                    # If we didn't find responses for ALL tool_call_ids, skip this incomplete AI message
                    if found_responses != tool_call_ids:
                        # Skip this incomplete AI message
                        i += 1
                        continue
            
            # Message is safe to include
            cleaned_messages.append(msg)
            i += 1
        
        # Get the latest user question - this is what we're answering NOW
        latest_question = _get_latest_user_question(cleaned_messages, exclude_last=False)
        
        # Build context with explicit focus on current question
        # Load system message from prompts.json
        system_msg = get_system_message(
            "generate_query_or_respond",
            current_question=latest_question
        )
        
        messages_with_context = [
            {"role": "system", "content": system_msg}
        ] + list(cleaned_messages)
        
        response = (
            response_model
            .bind_tools(all_tools).invoke(messages_with_context)
        )
        return {"messages": [response]}

    def grade_documents(
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question.
        
        CRITICAL: Prevents infinite loops by limiting rewrites to 1 attempt.
        After one rewrite, always generates answer even if docs aren't perfect.
        """
        messages = state["messages"]
        
        # CRITICAL: Prevent infinite loops by limiting rewrites to 1
        # Check recent messages (last 6) for rewrite pattern:
        # Pattern: user → ai (tool_call) → tool → user (rewritten) → ai (tool_call) → tool
        # If we see 2 tool responses in recent messages, we've already rewritten once
        
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        tool_responses_in_recent = sum(
            1 for msg in recent_messages
            if (hasattr(msg, 'type') and msg.type == "tool") or
               (isinstance(msg, dict) and msg.get("type") == "tool") or
               (isinstance(msg, dict) and msg.get("name"))  # Tool messages have 'name' field
        )
        
        # If we have 2+ tool responses in recent messages, we've already rewritten once
        # Break the loop by going to generate_answer
        if tool_responses_in_recent >= 2:
            return "generate_answer"
        
        # Get the current question (exclude last message which should be the tool response)
        question = _get_latest_user_question(messages, exclude_last=True)
        
        # CRITICAL: Find the actual tool response message (not just assume it's last)
        # Tool responses come after AI messages with tool_calls
        context = ""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            msg_type = getattr(msg, 'type', None) or (msg.get('type') if isinstance(msg, dict) else None)
            msg_role = getattr(msg, 'role', None) or (msg.get('role') if isinstance(msg, dict) else None)
            
            # Check if this is a tool message
            if (msg_type == "tool") or (msg_role == "tool"):
                # This is the tool response - extract content
                if hasattr(msg, 'content'):
                    context = msg.content
                elif isinstance(msg, dict):
                    context = msg.get('content', '')
                break
        
        # If no tool response found, try last message as fallback
        if not context and messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                context = last_msg.content
            elif isinstance(last_msg, dict):
                context = last_msg.get('content', '')
        
        # Load prompt from prompts.json
        prompt = get_prompt("grade_documents", question=question, context=context)
        response = (
            grader_model
            .with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": prompt}]
            )
        )
        score = response.binary_score
        
        if score == "yes":
            return "generate_answer"
        else:
            # First time grading as "no" - allow one rewrite
            return "rewrite_question"
    
    def rewrite_question(state: MessagesState):
        """Rewrite the original user question."""
        messages = state["messages"]
        
        # Get the current question
        question = _get_latest_user_question(messages, exclude_last=False)
        
        # Load prompt from prompts.json
        prompt = get_prompt("rewrite_question", question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}
    
    def generate_answer(state: MessagesState):
        """Generate an answer from retrieved context."""
        messages = state["messages"]
        
        # Get the current question (exclude last message which should be the tool response)
        question = _get_latest_user_question(messages, exclude_last=True)
        
        # CRITICAL: Find the actual tool response message (not just assume it's last)
        # Tool responses come after AI messages with tool_calls
        context = ""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            msg_type = getattr(msg, 'type', None) or (msg.get('type') if isinstance(msg, dict) else None)
            msg_role = getattr(msg, 'role', None) or (msg.get('role') if isinstance(msg, dict) else None)
            
            # Check if this is a tool message
            if (msg_type == "tool") or (msg_role == "tool"):
                # This is the tool response - extract content
                if hasattr(msg, 'content'):
                    context = msg.content
                elif isinstance(msg, dict):
                    context = msg.get('content', '')
                break
        
        # If no tool response found, try last message as fallback
        if not context and messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                context = last_msg.content
            elif isinstance(last_msg, dict):
                context = last_msg.get('content', '')
        
        # Handle empty context case
        if not context or context.strip() == "":
            # No context retrieved - provide helpful response
            response_content = "I don't have enough information in the provided documents to answer this question. Please ensure relevant documents have been ingested."
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=response_content)]}
        
        # Load prompt from prompts.json
        prompt = get_prompt("generate_answer", question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    def route_after_tools(
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        """Route after tool execution based on which tool was called.

        If retriever tool was called: grade documents and route accordingly.
        If other tools were called: go to generate_answer (tool result as context).
        """
        messages = state["messages"]

        # Find the most recent tool response
        last_tool_name = None
        for msg in reversed(messages):
            msg_type = getattr(msg, 'type', None) or (msg.get('type') if isinstance(msg, dict) else None)
            msg_role = getattr(msg, 'role', None) or (msg.get('role') if isinstance(msg, dict) else None)

            if (msg_type == "tool") or (msg_role == "tool"):
                if hasattr(msg, 'name'):
                    last_tool_name = msg.name
                elif isinstance(msg, dict):
                    last_tool_name = msg.get('name')
                break

        # If retriever tool was called, grade the documents
        retriever_name = retriever_tool.name
        if last_tool_name == retriever_name:
            return grade_documents(state)

        # For other tools, go to generate_answer to formulate a natural response
        return "generate_answer"

    return {
        "generate_query_or_respond": generate_query_or_respond,
        "grade_documents": grade_documents,
        "route_after_tools": route_after_tools,
        "rewrite_question": rewrite_question,
        "generate_answer": generate_answer,
    }


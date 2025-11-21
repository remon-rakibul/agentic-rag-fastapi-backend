"""Chat service for RAG workflow orchestration."""
from typing import AsyncGenerator, Optional
import uuid
from langchain_core.messages import convert_to_messages
from app.services.vector_store_service import get_vector_store_service
from app.workflows.rag_graph import build_rag_graph, get_checkpointer
from app.core.config import settings
from app.utils.retrieval_logger import get_retrieval_logger


class ChatService:
    """Service for managing chat interactions with RAG workflow."""
    
    def __init__(self):
        self.vector_store_service = get_vector_store_service()
    
    def get_graph_for_user(self, user_id: Optional[int] = None, checkpointer=None):
        """Get a RAG graph configured for a specific user with checkpointer.
        
        Creates a RAG workflow that retrieves only from the user's documents.
        
        Args:
            user_id: User ID for scoping retriever (filters documents by user)
            checkpointer: PostgresSaver checkpointer instance (required)
            
        Returns:
            Compiled LangGraph workflow with user-isolated retrieval
        """
        # Get user-scoped retriever with top-k=5 filtering
        # This will:
        # 1. Filter documents WHERE user_id = {user_id}
        # 2. Rank filtered docs by cosine similarity to query
        # 3. Return top 5 most relevant documents from user's collection
        retriever = self.vector_store_service.get_retriever(
            user_id=user_id,
            search_type="similarity",  # Cosine similarity search
            search_kwargs={"k": 5}      # Retrieve top 5 most similar docs
        )
        
        # Build graph with user's retriever AND checkpointer
        graph = build_rag_graph(
            retriever=retriever,
            checkpointer=checkpointer,
            tool_name="retrieve_documents",
            tool_description="Search and return information from your ingested documents."
        )
        
        return graph
    
    async def stream_chat(
        self,
        message: str,
        user_id: Optional[int] = None,
        thread_id: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """Stream chat response from RAG workflow with token-by-token streaming.
        
        Yields:
            Dict with keys: type ('token', 'done', 'error'), content, thread_id
        """
        try:
            # Prepare config with thread_id
            if thread_id is None:
                thread_id = str(uuid.uuid4())
            
            # Get AsyncPostgresSaver for true async streaming support
            checkpointer = await get_checkpointer()
            
            # Build graph WITH checkpointer
            graph = self.get_graph_for_user(user_id=user_id, checkpointer=checkpointer)
            
            # Prepare input
            input_messages = convert_to_messages([{"role": "user", "content": message}])
            
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 50  # Prevent infinite loops (default is 25)
            }
            
            # Use astream_events for TRUE token-by-token streaming
            # This works with AsyncPostgresSaver!
            full_content = ""
            
            # Buffer for direct responses from generate_query_or_respond
            # We buffer because tokens stream BEFORE we know if retrieval will happen
            direct_response_buffer = []
            will_retrieve = False
            
            # Set context for retrieval logging
            logger = get_retrieval_logger()
            logger.set_context(thread_id=thread_id, original_question=message)
            
            async for event in graph.astream_events(
                {"messages": input_messages},
                config=config,
                version="v1"  # Use v1 as in the working example
            ):
                metadata = event.get("metadata", {})
                node_name = metadata.get("langgraph_node", "")
                
                # Detect if retrieval will happen (before tokens are streamed)
                if event["event"] == "on_chain_start" and node_name == "retrieve":
                    will_retrieve = True
                    # Clear buffer since we're going retrieval path
                    direct_response_buffer = []
                
                # Listen for LLM token streaming events
                if event["event"] == "on_chat_model_stream":
                    # Stream from generate_answer (after retrieval)
                    if node_name == "generate_answer":
                        content = event["data"]["chunk"].content
                        if content:
                            full_content += content
                            yield {
                                "type": "token",
                                "content": content,
                                "thread_id": thread_id
                            }
                    
                    # Buffer tokens from generate_query_or_respond
                    # We'll stream them only if retrieval doesn't happen
                    elif node_name == "generate_query_or_respond":
                        chunk = event["data"].get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            direct_response_buffer.append(chunk.content)
                
                # When generate_query_or_respond ends, check if we should stream buffered content
                if event["event"] == "on_chain_end" and node_name == "generate_query_or_respond":
                    # If we didn't go to retrieve and have buffered content, stream it
                    if not will_retrieve and direct_response_buffer:
                        buffered_content = "".join(direct_response_buffer)
                        full_content += buffered_content
                        yield {
                            "type": "token",
                            "content": buffered_content,
                            "thread_id": thread_id
                        }
                        direct_response_buffer = []
                    
                    # Reset flag for next iteration (if generate_query_or_respond is called again)
                    will_retrieve = False
            
            # Final check: Stream any remaining buffered direct response
            # (in case workflow ended without going through generate_answer)
            if direct_response_buffer and not will_retrieve:
                buffered_content = "".join(direct_response_buffer)
                full_content += buffered_content
                yield {
                    "type": "token",
                    "content": buffered_content,
                    "thread_id": thread_id
                }
            
            # Yield completion
            yield {
                "type": "done",
                "content": full_content,
                "thread_id": thread_id
            }
            
        except Exception as e:
            import traceback
            error_str = str(e)
            
            # Check for incomplete tool call sequence error
            if "tool_calls" in error_str and "tool_call_id" in error_str:
                error_detail = (
                    f"Checkpoint contains incomplete tool call sequence. "
                    f"This usually happens when a conversation was interrupted.\n\n"
                    f"SOLUTION: Use a new thread_id (leave empty or generate a new UUID) "
                    f"or clear the checkpoint for thread_id: {thread_id}\n\n"
                    f"Original error: {error_str}\n{traceback.format_exc()}"
                )
            else:
                error_detail = f"{error_str}\n{traceback.format_exc()}"
            
            yield {
                "type": "error",
                "content": error_detail,
                "thread_id": thread_id if 'thread_id' in locals() else None
            }


def get_chat_service() -> ChatService:
    """Get chat service instance."""
    return ChatService()


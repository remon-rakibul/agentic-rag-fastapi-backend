"""Utility to load prompts and system messages from JSON configuration."""
import json
import os
from pathlib import Path
from typing import Dict, Any

# Path to prompts.json (relative to this file)
PROMPTS_FILE = Path(__file__).parent / "prompts.json"


class PromptLoader:
    """Loads and manages prompts from JSON configuration."""
    
    _prompts_data: Dict[str, Any] = None
    _last_modified: float = 0
    
    @classmethod
    def _load_prompts(cls) -> Dict[str, Any]:
        """Load prompts from JSON file with caching."""
        if cls._prompts_data is None or cls._should_reload():
            try:
                with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                    cls._prompts_data = json.load(f)
                    cls._last_modified = os.path.getmtime(PROMPTS_FILE)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Prompts file not found: {PROMPTS_FILE}\n"
                    f"Please create {PROMPTS_FILE} with prompt configurations."
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in prompts file {PROMPTS_FILE}: {e}\n"
                    f"Please check the JSON syntax."
                )
        return cls._prompts_data
    
    @classmethod
    def _should_reload(cls) -> bool:
        """Check if prompts file has been modified."""
        try:
            current_modified = os.path.getmtime(PROMPTS_FILE)
            return current_modified > cls._last_modified
        except OSError:
            return False
    
    @classmethod
    def get_system_message(cls, node_name: str, **kwargs) -> str:
        """Get system message for a specific node.
        
        Args:
            node_name: Name of the node (e.g., 'generate_query_or_respond')
            **kwargs: Variables to format into the template
            
        Returns:
            Formatted system message string
        """
        data = cls._load_prompts()
        system_messages = data.get('system_messages', {})
        
        if node_name not in system_messages:
            raise KeyError(
                f"System message not found for node '{node_name}'.\n"
                f"Available nodes: {list(system_messages.keys())}"
            )
        
        template = system_messages[node_name]['template']
        return template.format(**kwargs)
    
    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs) -> str:
        """Get prompt template for a specific use case.
        
        Args:
            prompt_name: Name of the prompt (e.g., 'grade_documents')
            **kwargs: Variables to format into the template
            
        Returns:
            Formatted prompt string
        """
        data = cls._load_prompts()
        prompts = data.get('prompts', {})
        
        if prompt_name not in prompts:
            raise KeyError(
                f"Prompt not found: '{prompt_name}'.\n"
                f"Available prompts: {list(prompts.keys())}"
            )
        
        template = prompts[prompt_name]['template']
        return template.format(**kwargs)
    
    @classmethod
    def get_retriever_tool_config(cls) -> Dict[str, str]:
        """Get retriever tool configuration.
        
        Returns:
            Dict with 'name' and 'description' keys
        """
        data = cls._load_prompts()
        return data.get('retriever_tool', {
            'name': 'retrieve_documents',
            'description': 'Search and return information from ingested documents.'
        })
    
    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        """Get default settings.
        
        Returns:
            Dict with model, temperature, k, streaming settings
        """
        data = cls._load_prompts()
        return data.get('settings', {
            'default_model': 'gpt-4o-mini',
            'default_temperature': 0,
            'default_k': 5,
            'streaming_enabled': True
        })
    
    @classmethod
    def reload(cls) -> None:
        """Force reload prompts from file (useful for testing)."""
        cls._prompts_data = None
        cls._last_modified = 0


# Convenience functions
def get_system_message(node_name: str, **kwargs) -> str:
    """Get system message for a node."""
    return PromptLoader.get_system_message(node_name, **kwargs)


def get_prompt(prompt_name: str, **kwargs) -> str:
    """Get prompt template."""
    return PromptLoader.get_prompt(prompt_name, **kwargs)


def get_retriever_tool_config() -> Dict[str, str]:
    """Get retriever tool configuration."""
    return PromptLoader.get_retriever_tool_config()


def get_settings() -> Dict[str, Any]:
    """Get default settings."""
    return PromptLoader.get_settings()


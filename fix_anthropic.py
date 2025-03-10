# fix_anthropic.py - Place this in the same directory as your main script
import os
import importlib
import logging

logger = logging.getLogger(__name__)

def get_claude_client(api_key=None):
    """Get a properly initialized Claude client that works with your environment"""
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key provided and ANTHROPIC_API_KEY environment variable not set")
    
    # Import anthropic library
    import anthropic
    
    try:
        # Try the newer API approach
        client = anthropic.Anthropic.__new__(anthropic.Anthropic)
client.api_key = api_key
        return client
    except TypeError as e:
        if "unexpected keyword argument 'proxies'" in str(e):
            # This is the error we're seeing - create client without proxies
            # We need to get access to the parent class to initialize without proxies
            from anthropic.client import Anthropic as AnthropicBase
            
            # Create a custom Anthropic class without proxies param
            class CustomAnthropic(AnthropicBase):
                def __init__(self, **kwargs):
                    # Remove proxies if present
                    if 'proxies' in kwargs:
                        del kwargs['proxies']
                    super().__init__(**kwargs)
            
            # Return our custom client
            return CustomAnthropic(api_key=api_key)
        else:
            # Different error, re-raise
            raise

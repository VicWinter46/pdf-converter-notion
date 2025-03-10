import sys
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkey patch the anthropic module
def patch_anthropic():
    try:
        import anthropic
        
        # Store original __init__ methods
        if hasattr(anthropic, 'Anthropic'):
            original_init = anthropic.Anthropic.__init__
            
            # Create a patched init that strips out proxies
            def patched_init(self, *args, **kwargs):
                if 'proxies' in kwargs:
                    logger.info(f"Removing proxies from Anthropic init: {kwargs['proxies']}")
                    del kwargs['proxies']
                return original_init(self, *args, **kwargs)
            
            # Apply our patch
            anthropic.Anthropic.__init__ = patched_init
            logger.info("Successfully patched Anthropic class")
            
        if hasattr(anthropic, 'Client'):
            original_client_init = anthropic.Client.__init__
            
            def patched_client_init(self, *args, **kwargs):
                if 'proxies' in kwargs:
                    logger.info(f"Removing proxies from Client init: {kwargs['proxies']}")
                    del kwargs['proxies']
                return original_client_init(self, *args, **kwargs)
                
            anthropic.Client.__init__ = patched_client_init
            logger.info("Successfully patched Client class")
            
        logger.info("Anthropic module successfully patched")
        return True
    except Exception as e:
        logger.error(f"Error patching anthropic: {str(e)}")
        return False

# Patch anthropic when this module is imported
success = patch_anthropic()

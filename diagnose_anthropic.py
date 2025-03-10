import sys
import os
import logging
import inspect

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check environment variables for proxy settings
logger.info("Checking environment variables for proxy settings:")
for key, value in os.environ.items():
    if "proxy" in key.lower():
        logger.info(f"Found proxy environment variable: {key}={value}")

# Check if requests library is patching things
logger.info("Checking if requests library is installed:")
try:
    import requests
    logger.info(f"Requests version: {requests.__version__}")
    logger.info("Checking requests session defaults:")
    session = requests.Session()
    logger.info(f"Session proxies: {session.proxies}")
except ImportError:
    logger.info("Requests library not installed")

# Try to import anthropic and inspect it
logger.info("Trying to import anthropic:")
try:
    import anthropic
    logger.info(f"Anthropic version: {anthropic.__version__ if hasattr(anthropic, '__version__') else 'unknown'}")
    
    # Check if Anthropic or Client class exists
    if hasattr(anthropic, 'Anthropic'):
        logger.info("Found Anthropic class")
        # Print the source code of the __init__ method
        try:
            logger.info(f"Anthropic.__init__ source:\n{inspect.getsource(anthropic.Anthropic.__init__)}")
        except Exception as e:
            logger.info(f"Could not get Anthropic.__init__ source: {e}")
    
    if hasattr(anthropic, 'Client'):
        logger.info("Found Client class")
        # Print the source code of the __init__ method
        try:
            logger.info(f"Client.__init__ source:\n{inspect.getsource(anthropic.Client.__init__)}")
        except Exception as e:
            logger.info(f"Could not get Client.__init__ source: {e}")
            
    # Check if HTTPClient is being used
    if hasattr(anthropic, 'Client') and hasattr(anthropic.Client, 'httpclient'):
        logger.info("Checking HTTPClient class")
        httpclient = anthropic.Client.httpclient
        logger.info(f"HTTPClient class: {httpclient}")
    
except ImportError:
    logger.info("Anthropic library not installed")
except Exception as e:
    logger.error(f"Error inspecting anthropic: {e}")

# Try a simple test creation without proxies
logger.info("Trying to create Anthropic client with minimal args:")
try:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or "fake_key_for_testing"
    logger.info("Creating client with just api_key")
    if hasattr(anthropic, 'Anthropic'):
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("Successfully created Anthropic client")
    elif hasattr(anthropic, 'Client'):
        client = anthropic.Client(api_key=api_key)
        logger.info("Successfully created Client instance")
    else:
        logger.info("Could not find appropriate client class")
except Exception as e:
    logger.error(f"Error creating client: {e}")

logger.info("Diagnosis complete")

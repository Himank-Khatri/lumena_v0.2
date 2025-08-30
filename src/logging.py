import logging
import os
from datetime import datetime
from src.config import config

def setup_logging(log_dir=config.get("general.log_dir")):
    """
    Sets up logging to output debug messages to a file and info/warning to console.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime("chatbot_%Y%m%d_%H%M%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    # Create logger
    logger = logging.getLogger("chatbot_logger")
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs debug messages
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create console handler with a higher log level (e.g., CRITICAL) to suppress errors from CLI
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL) # Changed to CRITICAL to suppress ERROR from console
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# Initialize logger
logger = setup_logging()

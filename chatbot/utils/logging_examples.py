"""
Example usage of global logger in different modules
"""
from utils.logger import get_logger

# Get module-specific logger
logger = get_logger(__name__)

def example_function():
    """Example function showing logging usage"""
    
    logger.info("This is an info message")
    logger.debug("This is a debug message (only visible when debug mode is enabled)")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    try:
        # Some operation that might fail
        result = 10 / 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Full exception details:")  # This includes stack trace
    
    return "Function completed"

# Usage in different scenarios
class ExampleClass:
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ExampleClass initialized")
    
    def process_data(self, data):
        self.logger.info(f"Processing data: {len(data) if data else 'None'}")
        self.logger.debug(f"Data content: {data}")
        
        if not data:
            self.logger.warning("Empty data received")
            return None
        
        # Processing logic here
        self.logger.info("Data processed successfully")
        return data

# Different logging patterns
def log_performance(func):
    """Decorator for performance logging"""
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        
        logger.debug(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
        return result
    
    return wrapper

@log_performance
def slow_function():
    """Example function with performance logging"""
    import time
    time.sleep(1)  # Simulate slow operation
    return "Operation completed"

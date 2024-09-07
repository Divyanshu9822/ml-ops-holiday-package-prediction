from src.logger import logger

logger.info("Hello, World!")

def some_function():
    try:
        logger.info("This is an info log from some_function")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    logger.info("Starting the script")
    some_function()
    logger.info("Ending the script")
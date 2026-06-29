from multi_doc_chat.logger.custom_logger import CustomLogger

# Shared, process-wide structured logger used across the package.
GLOBAL_LOGGER = CustomLogger().get_logger("multi_doc_chat")

__all__ = ["CustomLogger", "GLOBAL_LOGGER"]

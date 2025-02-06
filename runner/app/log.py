import contextvars
import logging

# Create a context variable to store request_id per request
request_id_var = contextvars.ContextVar("request_id", default="")

class ContextFilter(logging.Filter):
    """Logging filter to add request_id from contextvars."""
    def filter(self, record):
        record.__dict__.setdefault("request_id", request_id_var.get())
        return True

def config_logging():
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - request_id=%(request_id)s %(message)s",
        defaults={"request_id": ""},
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addFilter(ContextFilter())  # Attach the filter to all loggers
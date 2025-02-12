import logging

def config_logging(log_level: int = logging.INFO, request_id: str = "", stream_id: str = ""):
    formatter = logging.Formatter(
        "timestamp=%(asctime)s level=%(levelname)s logger=%(name)s location=%(filename)s:%(lineno)d:%(funcName)s request_id=%(request_id)s stream_id=%(stream_id)s message=%(message)s",
        defaults={"request_id": request_id, "stream_id": stream_id},
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    logger = logging.getLogger()  # Root logger
    logger.setLevel(log_level)
    logger.addHandler(handler)
    return logger

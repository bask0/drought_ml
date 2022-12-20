
import logging 


class LoggerAction(object):
    def __init__(self, level: int = logging.DEBUG, logger_name: str = 'pytorch_lightning'):
        self.level = level
        self.logger_name = logger_name

    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        if self.level >= logger.level:
            return logger
        else:
            None

    def __exit__(self, type, value, traceback):
        pass

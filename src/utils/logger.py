import logging


def get_logger(log_level="INFO", filename=None):
    level = getattr(logging, log_level)
    format = "%(asctime)s [%(levelname)s] %(message)s"
    dateformat = "%d/%m/%y %H:%M:%S"
    if filename:
        logging.basicConfig(
            level=level,
            filename=filename,
            filemode="w",
            format=format,
            datefmt=dateformat,
        )
    else:
        logging.basicConfig(
            level=level,
            format=format,
            datefmt=dateformat,
        )
    return logging

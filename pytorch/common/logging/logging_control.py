import os
import logging


def create_logger(
    output_folder, log_name, console_level=logging.ERROR, file_level=logging.WARNING
):
    # create logger
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_filename = os.path.join(output_folder, log_name)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    # file logging
    fileHandler = logging.FileHandler(log_filename)
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(file_level)
    rootLogger.addHandler(fileHandler)
    # and console for several types of messages
    # create console handler and set level to debug
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(console_level)
    rootLogger.addHandler(consoleHandler)

import logging
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s      %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('train.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info('Hello')
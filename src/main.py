#from experiment_remote import run
from experiment_8 import run
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


run()
print(1)

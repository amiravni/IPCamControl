import logging.config
import os.path
LOGGING_CONF = os.path.join(os.path.dirname(__file__), "logging.ini")
logging.config.fileConfig(LOGGING_CONF)
import logging
LOGGER = logging.getLogger('main')
from cfg.config import *
from cfg.version import *
from cfg.queues import *

from multiprocessing import Queue as mpQueue #TODO: change to Queue
import traceback

def try_except_dec(default_output=''):
    def decorate(f):
        def applicator(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as ex:
                LOGGER.exception("*** EXCEPTION at " + str(f.__name__) + " from " + str(f.__globals__['__name__']) +
                                 " ***:\n " + str(traceback.format_exc()))
                return default_output
        return applicator
    return decorate


class MyQueue:
    def __init__(self, name='', maxsize=128):
        self.name = name
        self._maxsize = maxsize
        self.queue = mpQueue(maxsize=self._maxsize)

    def restart(self):
        self.queue = mpQueue(maxsize=self._maxsize)

    def check_queue(self, warning=2):
        if self.queue.qsize() > GENERAL['queue_warning_length']:
            LOGGER.warning('{}: Frame Queue size is {}'.format(str(self.name), str(self.queue.qsize())))

    def put(self, var, block=True):
        self.check_queue()
        self.queue.put(var)

    def get(self, block=True, timeout=0):
        return self.queue.get(block=block, timeout=timeout)

    def full(self):
        return self.queue.full()

    def qsize(self):
        return self.queue.qsize()

capture_queue = MyQueue('Capture Queue', maxsize=500)
image_queue = MyQueue('Image Queue', maxsize=128)
file_queue = MyQueue('File Queue', maxsize=20)
object_detection_queue = MyQueue('OD Queue', maxsize=100)
alert_queue = MyQueue('Alert Queue', maxsize=100)

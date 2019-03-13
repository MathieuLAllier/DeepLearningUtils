import re
import tensorflow as tf


def name_scope(f):
    """ https://engineering.taboola.com/tensorflow-scope-software-engineering/ """
    def func(*args, **kwargs):
        name = f.__name__[re.search(r'[^_]', f.__name__).start():]
        with tf.name_scope(name):
            return f(*args, **kwargs)
    return func

import time


def sagemaker_timestamp():
    """
    Return a timestamp with millisecond precision.
    As implemented in sagemaker.utils.sagemaker_timestamp
    """
    moment = time.time()
    moment_ms = repr(moment).split('.')[1][:3]
    return time.strftime("%Y-%m-%d-%H-%M-%S-{}".format(moment_ms), time.gmtime(moment))

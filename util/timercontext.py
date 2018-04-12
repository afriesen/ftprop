from contextlib import contextmanager
from timeit import default_timer as timer


@contextmanager
def timer_context(name, timers_dict):
    start_time = timer()

    yield

    if name not in timers_dict:
        timers_dict[name] = 0.0
    timers_dict[name] += timer() - start_time

from contextlib import contextmanager
import time

@contextmanager
def timer(enabled=True, label=""):
    if enabled:
        start = time.time()
        yield
        end = time.time()
        print(f"{label} took {end - start:.3f} seconds")
    else:
        yield

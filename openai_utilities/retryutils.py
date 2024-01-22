import random
import time
from icecream import ic

def retry_after_random_wait(
    min_wait :int, 
    max_wait: int, 
    retry_count: int,
    errors: tuple
):
    def decorator(func):    
        def wrapper(*args, **kwargs):
            try_counter = 0
            while try_counter < retry_count:
                try:
                    return func(*args, **kwargs)
                except errors as err:
                    try_counter += 1
                    delay = random.randint(min_wait, max_wait)
                    ic(f"Hit Rate Limit. Going to sleep for {delay}s")
                    # TODO: this is for debugging only so remove it later
                    # print("[ERROR] Hit error: %s | Waiting %ds | RETRY number: %d" % (err.status_code, delay, try_counter))                    
                    time.sleep(delay)
                except Exception as e:
                    raise e
            raise Exception(f"maximum retry of {retry_count} reached")            
        return wrapper
    return decorator

def retry_after_func_wait(max_retries: int, errors: tuple, wait_time_func):
    def decorator(func):    
        def wrapper(*args, **kwargs):
            try_counter = 0
            while try_counter < max_retries:
                try:
                    return func(*args, **kwargs)
                except errors as err:
                    try_counter += 1
                    print("Hit error: %s, RETRY number: %d" % (err, try_counter))
                    delay = wait_time_func(err)
                    time.sleep(delay)
                except Exception as e:
                    raise e
            raise Exception("maximum retry of %d reached" % max_retries)            
        return wrapper
    return decorator

import time
import threading
from collections import defaultdict, deque
from config import MAX_CONCURRENT_CALLS, RATE_LIMIT_WINDOW_MINUTES, RATE_LIMIT_CALLS_PER_WINDOW

class RateLimiter:
    def __init__(self):
        self.active_calls = 0
        self.call_timestamps = defaultdict(lambda: deque())
        self.lock = threading.Lock()
    
    def can_start_call(self, caller_id="default"):
        with self.lock:
            # check concurrent call limit
            if self.active_calls >= MAX_CONCURRENT_CALLS:
                return False
            
            # check per-caller rate limit
            now = time.time()
            window_start = now - (RATE_LIMIT_WINDOW_MINUTES * 60)
            
            timestamps = self.call_timestamps[caller_id]
            while timestamps and timestamps[0] < window_start:
                timestamps.popleft()
            
            if len(timestamps) >= RATE_LIMIT_CALLS_PER_WINDOW:
                return False
            
            self.active_calls += 1
            timestamps.append(now)
            return True
    
    def end_call(self):
        with self.lock:
            if self.active_calls > 0:
                self.active_calls -= 1

rate_limiter = RateLimiter()

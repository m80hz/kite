import time, sys

class Timer:
    def __init__(self, name: str, logger=print):
        self.name = name
        self.logger = logger
    def __enter__(self):
        self.t0 = time.time()
        self.logger(f"[TIMER] start: {self.name}")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        self.logger(f"[TIMER] end:   {self.name} ({dt:.3f}s)")
        self.elapsed = dt


def warn(msg: str):
    """Print an orange-colored warning message."""
    ORANGE = "\033[38;5;208m"
    RESET = "\033[0m"
    try:
        print(f"{ORANGE}[WARN]{RESET} {msg}")
    except Exception:
        # Fallback without color
        print(f"[WARN] {msg}")

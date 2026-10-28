VERBOSE_XRAY = True 

def log_xray(message: str):
    if VERBOSE_XRAY:
        print(f"[X-RAY] {message}")
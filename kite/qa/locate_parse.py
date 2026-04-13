import re

def parse_locate_time_conf(text: str):
    if not text:
        return None, 0.0
    t = text.lower()
    m = re.search(r'(?:t\s*=\s*)?([0-9]+\.?[0-9]*)\s*(?:s|sec|seconds?)', t)
    if m:
        try:
            return float(m.group(1)), 0.9
        except Exception:
            pass
    m = re.search(r'frame\s*([0-9]+)', t)
    if m:
        try:
            f = int(m.group(1)); return f, 0.6
        except Exception:
            pass
    m = re.search(r'([0-9]+\.?[0-9]*)', t)
    if m:
        try:
            return float(m.group(1)), 0.4
        except Exception:
            pass
    return None, 0.0

import numpy as np
from .segmenter import get_fps
from .optflow import compute_flow_magnitude_sequence

def select_event_times(video_path: str, stride:int=2, topk:int=5):
    mags, idxs = compute_flow_magnitude_sequence(video_path, stride=stride)
    if len(mags)==0: return []
    m = (mags - mags.min()) / (mags.max() - mags.min() + 1e-6)
    order = np.argsort(-m).tolist()
    chosen = []
    for i in order:
        if len(chosen)>=topk: break
        if all(abs(i-j)>5 for j in chosen):
            chosen.append(i)
    fps = get_fps(video_path)
    return [idxs[i]/fps for i in chosen]

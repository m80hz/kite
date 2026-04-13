import cv2
import numpy as np
from .optflow import compute_flow_magnitude_sequence, propose_segments_by_flow

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps

def segment_video(video_path, flow_stride=2, window_sec=2.0, overlap_sec=0.5, max_segments=8, min_flow_mag=0.5, transcode_if_needed=False):
    """Segment a video by optical-flow magnitude.

    Args:
        video_path: path to video file
        flow_stride: frame stride when computing flow
        window_sec, overlap_sec, max_segments, min_flow_mag: as before
        transcode_if_needed: if True, attempt an ffmpeg transcode when OpenCV cannot decode the input.
    """
    # try to get fps; if it fails and transcode requested, try a transcode
    try:
        fps = get_fps(video_path)
    except RuntimeError:
        if transcode_if_needed:
            # avoid circular import at module import time
            from .optflow import _ffmpeg_transcode_to_h264
            import os
            tpath = _ffmpeg_transcode_to_h264(video_path)
            try:
                fps = get_fps(tpath)
            finally:
                try:
                    os.unlink(tpath)
                except Exception:
                    pass
        else:
            # fall back to a safe default fps so downstream math doesn't crash
            fps = 30.0

    mags, idxs = compute_flow_magnitude_sequence(video_path, stride=flow_stride, transcode_if_needed=transcode_if_needed)
    segs = propose_segments_by_flow(mags, idxs, fps, window_sec, overlap_sec, max_segments, min_flow_mag)
    # convert frame indices to seconds
    return [(s/fps, e/fps) for (s,e) in segs], fps

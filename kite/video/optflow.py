import cv2
import numpy as np
import subprocess
import tempfile
import shutil
import os

def _ffmpeg_transcode_to_h264(src_path):
    """Transcode video to an H.264 mp4 in a temp file and return its path.
    Requires ffmpeg on PATH. Caller is responsible for deleting the file.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; please install ffmpeg to transcode AV1 videos.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    dst = tmp.name
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        dst
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        os.unlink(dst)
        raise RuntimeError(f"ffmpeg failed to transcode video: {e}")
    return dst


def compute_flow_magnitude_sequence(video_path, stride=2, resize=(512,512), transcode_if_needed=False):
    """Compute per-frame mean optical-flow magnitude sequence.

    If OpenCV cannot decode the input (common with AV1 builds of ffmpeg missing decoders),
    set transcode_if_needed=True to attempt an ffmpeg transcode to H.264 and retry.
    """
    def _open_cap(path):
        cap = cv2.VideoCapture(path)
        return cap

    cap = _open_cap(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # quick sanity check: try to grab one frame to ensure decoding works
    ok, frame = cap.read()
    if not ok:
        cap.release()
        if transcode_if_needed:
            try:
                tpath = _ffmpeg_transcode_to_h264(video_path)
            except Exception as e:
                raise RuntimeError(f"Failed to decode video and ffmpeg transcode failed: {e}")
            cap = _open_cap(tpath)
            if not cap.isOpened():
                os.unlink(tpath)
                raise RuntimeError(f"Cannot open transcoded video: {tpath}")
            # continue and make sure we cleanup the temp file at the end
            cleanup_transcoded = tpath
            # we've already read one frame attempt failed previously; re-read from start
        else:
            raise RuntimeError(
                "Video cannot be decoded by OpenCV/FFmpeg (see stderr for ffmpeg messages). "
                "If the file uses AV1/other codecs try installing ffmpeg or re-encoding the video to H.264."
            )
    else:
        # we got a frame; put it back by seeking to 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cleanup_transcoded = None

    prev = None
    mags = []
    frames = []
    idxs = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        if resize is not None:
            try:
                frame = cv2.resize(frame, resize)
            except Exception:
                # if frame is None or invalid, skip
                idx += 1
                continue
        # color conversion can fail if frame is None or 1-channel
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            idx += 1
            continue
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mags.append(float(np.mean(mag)))
            frames.append(frame)
            idxs.append(idx)
        prev = gray
        idx += 1
    cap.release()
    if cleanup_transcoded is not None:
        try:
            os.unlink(cleanup_transcoded)
        except Exception:
            pass
    return np.array(mags, dtype=np.float32), idxs

def propose_segments_by_flow(mags, idxs, fps, window_sec=2.0, overlap_sec=0.5, max_segments=8, min_flow_mag=0.5):
    if len(mags)==0:
        return []
    # normalize and pick top peaks
    import numpy as np
    m = (mags - mags.min()) / (mags.max() - mags.min() + 1e-6)
    # candidate indices sorted by score
    cand = np.argsort(-m)
    used = np.zeros_like(m, dtype=bool)
    segs = []
    w = int(window_sec*fps)
    o = int(overlap_sec*fps)
    for c in cand:
        if len(segs)>=max_segments:
            break
        if m[c] < min_flow_mag:
            break
        # convert idxs[c] (frame index) to center; compute [start,end)
        center = idxs[c]
        start = max(0, center - w//2)
        end = center + w//2
        # enforce non-overlap beyond overlap_sec
        conflict = False
        for (s,e) in segs:
            if not (end <= s+o or start >= e-o):
                conflict = True
                break
        if conflict:
            continue
        segs.append((start, end))
    # sort by start
    segs.sort(key=lambda x: x[0])
    return segs

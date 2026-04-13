from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter/union

def _center(b):
    return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

def _dist(a, b):
    ax, ay = a; bx, by = b
    return ((ax-bx)**2 + (ay-by)**2) ** 0.5

def infer_contact_and_bimanual(frames: List, detections_per_frame: List[List[Dict[str,Any]]]) -> Tuple[str, List[str]]:
    """Return ([CONTACT:GAIN|LOSS|STABLE], [BIMANUAL tokens]).
    Heuristics:
    - Pick 'gripper' boxes (or 'robot gripper', 'end effector') each frame.
    - Pick primary object as the non-gripper with highest mean IoU/nearest center to gripper across frames.
    - Contact trend = IOU(last) - IOU(first).
    - Bimanual: if two grippers detected and both approach object (or nearest gripper identity switches) add [BIMANUAL] / [HANDOVER].
    - Coord: if two grippers both near object (< distance thresh) add [COORD:left↔right].
    """
    gripper_aliases = {'gripper','robot gripper','end effector','end-effector','claw'}
    contact_series = []
    bimanual_tokens: List[str] = []

    # collect grippers and objects per frame
    grippers_f = []
    objects_f = []
    for dets in detections_per_frame:
        gs = [d for d in dets if any(alias in d['name'].lower() for alias in gripper_aliases)]
        os = [d for d in dets if d not in gs and 'robot' not in d['name'].lower() and 'arm' not in d['name'].lower()]
        grippers_f.append(gs[:2])  # at most two
        objects_f.append(os)

    # choose primary object by proximity to any gripper over frames
    primary_obj = None
    best_score = 1e9
    for t, (gs, os) in enumerate(zip(grippers_f, objects_f)):
        for o in os:
            oc = _center(o['bbox'])
            if gs:
                dmin = min(_dist(oc, _center(g['bbox'])) for g in gs)
                if dmin < best_score:
                    best_score = dmin; primary_obj = o

    if primary_obj is None:
        contact_token = "[CONTACT:STABLE]"
        return contact_token, bimanual_tokens

    # IOU series with nearest gripper
    nearest_ids = []
    for t, gs in enumerate(grippers_f):
        if not gs:
            contact_series.append(0.0); nearest_ids.append(None); continue
        dlist = [(_dist(_center(primary_obj['bbox']), _center(g['bbox'])), i) for i,g in enumerate(gs)]
        dlist.sort()
        idx = dlist[0][1]
        iou = _iou(primary_obj['bbox'], gs[idx]['bbox'])
        contact_series.append(iou)
        nearest_ids.append(idx)

    if len(contact_series)>=2:
        delta = contact_series[-1] - contact_series[0]
    else:
        delta = 0.0
    if delta > 0.10:
        contact_token = "[CONTACT:GAIN]"
    elif delta < -0.10:
        contact_token = "[CONTACT:LOSS]"
    else:
        contact_token = "[CONTACT:STABLE]"

    # BIMANUAL/HANDOVER/COORD
    two_grippers_somewhere = any(len(gs)>=2 for gs in grippers_f)
    if two_grippers_somewhere:
        # handover if nearest gripper id changes over time
        if any(nearest_ids[i] != nearest_ids[0] and nearest_ids[i] is not None for i in range(1,len(nearest_ids))):
            bimanual_tokens.append("[HANDOVER]")
        # coord if at any frame, both grippers within a small band around object
        for gs in grippers_f:
            if len(gs)>=2:
                d1 = _dist(_center(primary_obj['bbox']), _center(gs[0]['bbox']))
                d2 = _dist(_center(primary_obj['bbox']), _center(gs[1]['bbox']))
                if d1 < 80 and d2 < 80:  # pixel threshold; scene-scale dependent
                    bimanual_tokens.append("[COORD:left↔right]")
                    break
        bimanual_tokens.append("[BIMANUAL]")

    return contact_token, list(dict.fromkeys(bimanual_tokens))  # unique-order

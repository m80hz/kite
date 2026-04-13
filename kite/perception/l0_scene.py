import cv2
import numpy as np

def basic_objects_from_frame(frame, topk=5):
    """Heuristic: extract salient nouns via simple saliency + color clusters as placeholders.
    This is a TRAINING-FREE stub so pipeline runs without detectors. Replace with YOLO/OWLVIT if available.
    Returns a list of pseudo-objects with positions."
    """
    h, w = frame.shape[:2]
    # simple grid sampling of points as pseudo-objects
    objs = []
    step = max(1, min(h,w)//4)
    for y in range(step//2, h, step):
        for x in range(step//2, w, step):
            objs.append({"name": "obj", "cx": x, "cy": y})
            if len(objs)>=topk:
                return objs
    return objs

def relation_left_of(a,b):
    return a["cx"] < b["cx"]

def relation_above(a,b):
    return a["cy"] < b["cy"]

def build_2d_relations(objs):
    rels = []
    for i,a in enumerate(objs):
        for j,b in enumerate(objs):
            if i==j: continue
            rels.append((i,"left_of",j) if relation_left_of(a,b) else (i,"right_of",j))
            rels.append((i,"above",j) if relation_above(a,b) else (i,"below",j))
    return rels

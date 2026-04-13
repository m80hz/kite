from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import json
from .camera import CameraIntrinsics
from .depth3d import project_bbox_to_3d_centroid, box_depth_stats

@dataclass
class Node3D:
    id: int
    name: str
    bbox: List[float]
    point3d: Tuple[float,float,float]
    depth_var: float
    score: float

@dataclass
class Edge3D:
    subj: int
    pred: str
    obj: int
    weight: float = 1.0

@dataclass
class SceneGraph3D:
    nodes: List[Node3D] = field(default_factory=list)
    edges: List[Edge3D] = field(default_factory=list)
    t_sec: Optional[float] = None
    frame_idx: Optional[int] = None

def _rel_left_of(pa, pb): return pa[0] < pb[0]
def _rel_above(pa, pb):   return pa[1] < pb[1]
def _rel_infront_of(pa, pb): return pa[2] < pb[2]
"""
Ordering relations follow the camera frame convention:
    +x = right, +y = down, +z = forward (away from the camera).
Thus:
    left_of  → pa.x < pb.x
    above    → pa.y < pb.y  (smaller y is visually higher when +y points down)
    in_front → pa.z < pb.z  (closer to the camera has smaller z)
"""

def build_local_graph3d(detections: List[Dict[str,Any]], depth_map, cam: CameraIntrinsics, t_sec: float=None, frame_idx: int=None) -> SceneGraph3D:
    nodes = []
    for i, d in enumerate(detections):
        p3 = project_bbox_to_3d_centroid(d['bbox'], depth_map, cam)
        _, var = box_depth_stats(depth_map, d['bbox'])
        nodes.append(Node3D(id=i, name=d['name'], bbox=d['bbox'], point3d=p3, depth_var=var, score=d.get('score',0.0)))
    edges = []
    for i,a in enumerate(nodes):
        for j,b in enumerate(nodes):
            if i==j: continue
            if _rel_left_of(a.point3d, b.point3d):
                edges.append(Edge3D(subj=i, pred="left_of", obj=j))
            if _rel_above(a.point3d, b.point3d):
                edges.append(Edge3D(subj=i, pred="above", obj=j))
            if _rel_infront_of(a.point3d, b.point3d):
                edges.append(Edge3D(subj=i, pred="in_front_of", obj=j))
    return SceneGraph3D(nodes=nodes, edges=edges, t_sec=t_sec, frame_idx=frame_idx)

def graph3d_to_text(g: SceneGraph3D, max_nodes:int=8, max_rels:int=12) -> str:
    names = [f"{i}:{n.name}" for i,n in enumerate(g.nodes[:max_nodes])]
    rels = []
    for r in g.edges[:max_rels]:
        a = g.nodes[r.subj].name
        b = g.nodes[r.obj].name
        rels.append(f"{a}->{r.pred}->{b}")
    ttag = []
    if g.t_sec is not None:
        ttag.append(f"[SCENEGRAPH_TIMESTAMP:{g.t_sec:.2f}s]")
    if g.frame_idx is not None:
        ttag.append(f"[SCENEGRAPH_FRAME:{g.frame_idx}]")
    return " ".join(ttag + [f"[SCENE3D:NODES] {'; '.join(names)}", f"[SCENE3D:RELS] {'; '.join(rels)}\n"])

def graph3d_to_text_with_prefix(g: SceneGraph3D, prefix: str="", **kwargs) -> str:
    base = graph3d_to_text(g, **kwargs)
    return f"{prefix}{base}" if prefix else base

# ---------------- JSON serialization helpers -----------------
def graph3d_to_json(g: SceneGraph3D, max_nodes:int=16, max_rels:int=24) -> Dict[str, Any]:
    """Return a JSON-serializable dict representing the local 3D scene graph.

    Schema (example):
    {
      "frame_time": 12.40,
      "frame_idx": 123,
      "objects": [
         {"id":0, "cls":"mug", "bbox":[x1,y1,x2,y2], "pos":[x,y,z], "depth_var":0.0021, "score":0.86}
      ],
      "relations": [
         {"subj":0, "rel":"left_of", "obj":1},
         ...
      ]
    }
    """
    obj_list = []
    for n in g.nodes[:max_nodes]:
        obj_list.append({
            "id": n.id,
            "cls": n.name,
            "bbox": [round(float(v),2) for v in n.bbox],
            "pos": [round(float(p),2) for p in n.point3d],
            "depth_var": round(float(n.depth_var),2),
            "score": round(float(n.score),2)
        })
    rel_list = []
    for e in g.edges[:max_rels]:
        rel_list.append({"subj": e.subj, "rel": e.pred, "obj": e.obj})
    return {
        "frame_time": (round(float(g.t_sec),2) if g.t_sec is not None else None),
        "frame_idx": int(g.frame_idx) if g.frame_idx is not None else None,
        "objects": obj_list,
        "relations": rel_list
    }

def graph3d_to_json_str(g: SceneGraph3D, **kwargs) -> str:
    return json.dumps(graph3d_to_json(g, **kwargs), ensure_ascii=False)

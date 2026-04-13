from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np
import json

@dataclass
class Track3D:
    id: int
    name: str
    points: List[Tuple[float,float,float]] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    last_bbox: List[float] = field(default_factory=list)
    count: int = 0

@dataclass
class CameraExtrinsics:
    R: np.ndarray  # 3x3
    t: np.ndarray  # 3,
    @staticmethod
    def identity():
        return CameraExtrinsics(R=np.eye(3), t=np.zeros(3))

def _transform_point(p, extr: CameraExtrinsics):
    return (extr.R @ np.array(p).reshape(3,)) + extr.t

class GlobalSceneGraphAggregator:
    def __init__(self, max_dist: float = 0.2):
        self.max_dist = max_dist
        self.tracks: List[Track3D] = []
        self.next_id = 0

    def _assign(self, nodes_local, extr: CameraExtrinsics, t_sec: float=None):
        for n in nodes_local:
            pw = _transform_point(n.point3d, extr)
            best_id = -1; best_d = 1e9
            for t in self.tracks:
                if t.name != n.name or len(t.points)==0: 
                    continue
                d = np.linalg.norm(np.array(t.points[-1]) - pw)
                if d < best_d:
                    best_d = d; best_id = t.id
            if best_d < self.max_dist:
                tr = next(t for t in self.tracks if t.id==best_id)
                tr.points.append(tuple(pw)); tr.last_bbox = n.bbox; tr.count += 1
                if t_sec is not None:
                    tr.times.append(float(t_sec))
            else:
                tr = Track3D(id=self.next_id, name=n.name, points=[tuple(pw)], times=[float(t_sec)] if t_sec is not None else [], last_bbox=n.bbox, count=1)
                self.tracks.append(tr); self.next_id += 1

    def update_from_local(self, local_graph, extrinsics: CameraExtrinsics):
        self._assign(local_graph.nodes, extrinsics, getattr(local_graph, 't_sec', None))

    def summary(self, max_tracks:int=10) -> str:
        lines = []
        for t in self.tracks[:max_tracks]:
            if len(t.points)==0: continue
            p = np.mean(np.array(t.points), axis=0)
            lines.append(f"{t.name}@({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")
        return "[GLOBAL3D: OBJECT TRACKS] " + "; ".join(lines)

    def to_text_relations(self, max_pairs:int=12) -> str:
        """
        Ordering relations adhere to camera frame convention:
          +x right, +y down, +z forward.
        We omit absolute "near" due to non-metric depth scale.
        """
        arr = [(t, np.mean(np.array(t.points), axis=0)) for t in self.tracks if len(t.points)>0]
        rels = []
        for i in range(len(arr)):
            for j in range(len(arr)):
                if i==j: continue
                ti,pi = arr[i]; tj,pj = arr[j]
                if pi[0] < pj[0]:
                    rels.append(f"{ti.name}->left_of->{tj.name}")
                if pi[1] < pj[1]:
                    rels.append(f"{ti.name}->above->{tj.name}")
                if pi[2] < pj[2]:
                    rels.append(f"{ti.name}->in_front_of->{tj.name}")
        return "[GLOBAL3D: OBJECT RELATIONS] " + "; ".join(rels[:max_pairs])

    def track_legend(self, max_tracks:int=12) -> List[str]:
        lines = []
        for t in self.tracks[:max_tracks]:
            if t.times:
                t0 = min(t.times); t1 = max(t.times)
                lines.append(f"Track{t.id}:{t.name}[{t0:.2f}-{t1:.2f}s]")
            else:
                lines.append(f"Track{t.id}:{t.name}")
        return lines

    # ---------------- JSON serialization ----------------
    def to_json(self, max_tracks:int=20, max_rels:int=40) -> Dict[str, Any]:
        tracks_out = []
        for t in self.tracks[:max_tracks]:
            if len(t.points)==0:
                continue
            pts = np.array(t.points)
            meanp = pts.mean(axis=0)
            tracks_out.append({
                "track_id": t.id,
                "cls": t.name,
                "mean_pos": [round(float(v),2) for v in meanp],
                "samples": len(t.points),
                "time_span": [round(float(min(t.times)),2) if t.times else None, round(float(max(t.times)),2) if t.times else None]
            })
        # Derive pairwise relations as in to_text_relations
        rels_out = []
        arr = [(t, np.mean(np.array(t.points), axis=0)) for t in self.tracks if len(t.points)>0]
        for i in range(len(arr)):
            for j in range(len(arr)):
                if i==j: continue
                ti,pi = arr[i]; tj,pj = arr[j]
                if pi[0] < pj[0]:
                    rels_out.append({"subj": ti.name, "rel": "left_of", "obj": tj.name})
                if pi[1] < pj[1]:
                    rels_out.append({"subj": ti.name, "rel": "above", "obj": tj.name})
                if pi[2] < pj[2]:
                    rels_out.append({"subj": ti.name, "rel": "in_front_of", "obj": tj.name})
                if len(rels_out) >= max_rels:
                    break
            if len(rels_out) >= max_rels:
                break
        # Include track legend strings for temporal span readability
        legend = []
        for t in self.tracks[:max_tracks]:
            if len(t.points)==0:
                continue
            if t.times:
                legend.append(f"Track{t.id}:{t.name}[{min(t.times):.2f}-{max(t.times):.2f}s]")
            else:
                legend.append(f"Track{t.id}:{t.name}")
        return {"tracks": tracks_out, "relations": rels_out, "track_legend": legend}

    def to_json_str(self, **kwargs) -> str:
        return json.dumps(self.to_json(**kwargs), ensure_ascii=False)

    # -------------- Velocity utilities (XZ plane) --------------
    def velocities_xz(self) -> Dict[int, Tuple[float,float]]:
        """Return recent (vx, vz) per track using last 2 samples in camera frame.

        vx along +x (right), vz along +z (forward away from camera).
        Missing or insufficient samples omitted.
        """
        out: Dict[int, Tuple[float,float]] = {}
        for tr in self.tracks:
            if len(tr.points) < 2 or len(tr.times) < 2:
                continue
            p0, p1 = tr.points[-2], tr.points[-1]
            t0, t1 = tr.times[-2], tr.times[-1]
            dt = max(1e-6, t1 - t0)
            vx = (p1[0]-p0[0]) / dt
            vz = (p1[2]-p0[2]) / dt
            out[tr.id] = (vx, vz)
        return out

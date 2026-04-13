"""BEV (Bird's Eye View) raster rendering utilities.

This module produces a compact 256x256 top-down raster summarizing the 3D scene:
  - Occupancy / semantic dots for tracked objects (projected X,Z from camera frame)
  - Color per class (hash → deterministic RGB)
    - Short class label text (object name only, small font)
    - Movement direction arrows (fixed pixel length) only for robot-related objects (robot, gripper, arm, end-effector)
  - Highlight of objects present in the current local keyframe (larger circle border)

Coordinate convention inherited from camera frame used elsewhere:
   +x = right, +y = down, +z = forward (away from camera)
In the BEV map:
   Horizontal axis (columns)  → +x (right)
   Vertical axis (rows)       → +z (forward); smaller z (closer) near the bottom, farther objects toward the top.

Design goals:
 - Pure function: render_bev(local_graph, global_agg, size=256) → np.ndarray (BGR uint8)
 - No heavy dependencies beyond cv2 / numpy
 - Safe if graphs empty (returns blank white canvas with legend line)

The resulting image can be appended to VLM multi-image inputs directly after the RGB frame
from which the local graph was extracted. A short prompt note should describe semantics
 (handled by caller). 
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2

try:  # type hints only; modules imported at runtime by caller
    from .scene_graph3d import SceneGraph3D
    from .global_scene_graph import GlobalSceneGraphAggregator, Track3D
except Exception:  # pragma: no cover
    SceneGraph3D = Any  # type: ignore
    GlobalSceneGraphAggregator = Any  # type: ignore
    Track3D = Any  # type: ignore


def _color_for_class(cls: str) -> Tuple[int,int,int]:
    """Generate a deterministic bright color (BGR) for a class string."""
    h = (hash(cls) & 0xFFFFFF) ^ 0x5A5A5A
    # Emphasize different channels; ensure not too dark
    b = 64 + (h & 0xFF) * 191 // 255
    g = 64 + ((h >> 8) & 0xFF) * 191 // 255
    r = 64 + ((h >> 16) & 0xFF) * 191 // 255
    return int(b), int(g), int(r)


def _compute_extents_local(local_graph: SceneGraph3D) -> Optional[Tuple[float,float,float,float]]:
    xs, zs = [], []
    for n in getattr(local_graph, 'nodes', []) or []:
        try:
            xs.append(float(n.point3d[0])); zs.append(float(n.point3d[2]))
        except Exception:
            continue
    if not xs or not zs:
        return None
    xmin, xmax = min(xs), max(xs)
    zmin, zmax = min(zs), max(zs)
    # pad slightly to avoid edge clipping
    pad_x = max(1e-6, 0.05*(xmax-xmin+1e-6))
    pad_z = max(1e-6, 0.05*(zmax-zmin+1e-6))
    return (xmin-pad_x, xmax+pad_x, zmin-pad_z, zmax+pad_z)


def _track_velocity(track: Track3D) -> Optional[Tuple[float,float]]:
    # This helper remains for compatibility; prefer aggregator.velocities_xz when available.
    try:
        if getattr(track, 'points', None) is None or getattr(track, 'times', None) is None:
            return None
        if len(track.points) < 2 or len(track.times) < 2:
            return None
        p0, p1 = track.points[-2], track.points[-1]
        t0, t1 = track.times[-2], track.times[-1]
        dt = max(1e-6, float(t1) - float(t0))
        vx = (p1[0]-p0[0]) / dt
        vz = (p1[2]-p0[2]) / dt
        return float(vx), float(vz)
    except Exception:
        return None


def _is_robot_related(name: str) -> bool:
    s = (name or "").lower()
    # Conservative set of keywords to identify robot/gripper-related tracks
    keywords = (
        "robot", "gripper", "end effector", "end_effector", "ee", "wrist", "manipulator", "arm"
    )
    return any(k in s for k in keywords)


def render_bev(local_graph: SceneGraph3D, global_agg: GlobalSceneGraphAggregator, size: int = 256) -> np.ndarray:
    """Render a BEV raster.

    Args:
        local_graph: SceneGraph3D for current keyframe (to highlight current objects).
        global_agg:  GlobalSceneGraphAggregator with accumulated tracks (for positions + velocity).
        size:        Output square size in pixels.

    Returns:
        (H,W,3) uint8 BGR image.
    """
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    if local_graph is None or not getattr(local_graph, 'nodes', None):
        cv2.putText(canvas, 'EMPTY_SCENE', (8, size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        return canvas
    # Compute extents strictly from local scene (no aggregation)
    ext = _compute_extents_local(local_graph)
    if ext is None:
        cv2.putText(canvas, 'EMPTY_SCENE', (8, size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        return canvas
    xmin, xmax, zmin, zmax = ext
    scale_x = (size-1) / max(1e-6, (xmax - xmin))
    scale_z = (size-1) / max(1e-6, (zmax - zmin))

    # Draw grid (optional light lines)
    for frac in np.linspace(0,1,5):
        xg = int(frac*(size-1))
        zg = int(frac*(size-1))
        cv2.line(canvas, (xg,0), (xg,size-1), (230,230,230), 1, cv2.LINE_AA)
        cv2.line(canvas, (0,zg), (size-1,zg), (230,230,230), 1, cv2.LINE_AA)

    # Optional: prepare velocity map from aggregator (used only for matching robot nodes)
    vel_map: Dict[int, Tuple[float,float]] = {}
    if global_agg is not None and hasattr(global_agg, 'velocities_xz'):
        try:
            vm = global_agg.velocities_xz()
            if isinstance(vm, dict):
                vel_map = vm
        except Exception:
            vel_map = {}

    # For matching local nodes to tracks (for velocity only), build a helper over agg tracks if provided
    agg_tracks = list(getattr(global_agg, 'tracks', [])) if global_agg is not None else []

    # Draw local detections only
    for n in getattr(local_graph, 'nodes', []) or []:
        try:
            px, _, pz = n.point3d
            u = int((px - xmin) * scale_x)
            v = int((pz - zmin) * scale_z)
            v = (size-1) - v
            u = max(0, min(size-1, u))
            v = max(0, min(size-1, v))
            color = _color_for_class(getattr(n, 'name', ''))
            conf = float(getattr(n, 'score', 0.2) or 0.2)
            radius = int(3 + 5 * max(0.0, min(1.0, conf)))   # submission version
            # radius = int(5 + 8 * max(0.0, min(1.0, conf)))
            cv2.circle(canvas, (u, v), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        except Exception:
            continue
    # Direction arrows for robot-related objects present in this local frame only
    # Fixed pixel length, only direction from last motion (aggregator track displacement or velocity)
    arrow_pix_len = 22
    for n in getattr(local_graph, 'nodes', []) or []:
        try:
            if not _is_robot_related(getattr(n, 'name', '')):
                continue
            # Find nearest matching track (name + proximity) to query velocity; optional
            best_tr = None; best_d = 1e9
            for tr in agg_tracks:
                if getattr(tr, 'name', None) != getattr(n, 'name', None) or not getattr(tr, 'points', None):
                    continue
                d = np.linalg.norm(np.array(tr.points[-1]) - np.array(n.point3d))
                if d < best_d:
                    best_d = d; best_tr = tr
            if best_tr is None or best_d >= 0.4:
                continue
            # Prefer direction from last displacement (p1 - p0); fallback to velocity direction
            dx = dz = None
            try:
                if hasattr(best_tr, 'points') and best_tr.points and len(best_tr.points) >= 2:
                    p0 = np.array(best_tr.points[-2], dtype=float)
                    p1 = np.array(best_tr.points[-1], dtype=float)
                    disp = p1 - p0
                    dx = float(disp[0]); dz = float(disp[2])
                else:
                    vx_vz = vel_map.get(getattr(best_tr, 'id', -1)) or _track_velocity(best_tr)
                    if vx_vz is not None:
                        dx, dz = float(vx_vz[0]), float(vx_vz[1])
            except Exception:
                dx = dz = None
            if dx is None or dz is None:
                continue
            # Anchor at this node's position
            px, _, pz = n.point3d
            u = int((px - xmin) * scale_x)
            v = int((pz - zmin) * scale_z)
            v = (size-1) - v
            u = max(0, min(size-1, u))
            v = max(0, min(size-1, v))
            # Convert world direction (dx,dz) into pixel direction and normalize to fixed length
            dir_u = dx * scale_x
            dir_v = -dz * scale_z  # minus because v axis is flipped
            norm = float(np.hypot(dir_u, dir_v))
            if not np.isfinite(norm) or norm < 1e-6:
                continue
            du = int(np.clip((dir_u / norm) * arrow_pix_len, -size, size))
            dv = int(np.clip((dir_v / norm) * arrow_pix_len, -size, size))
            end = (max(0, min(size-1, u + du)), max(0, min(size-1, v + dv)))
            color = _color_for_class(getattr(n, 'name', ''))
            cv2.arrowedLine(canvas, (u, v), end, (0,0,0), thickness=3, lineType=cv2.LINE_AA, tipLength=0.25)
            cv2.arrowedLine(canvas, (u, v), end, color, thickness=2, lineType=cv2.LINE_AA, tipLength=0.25)
        except Exception:
            continue
    # Add object name labels (no numeric id) based on local_graph nodes for clarity.
    if local_graph is not None:
        for n in getattr(local_graph, 'nodes', []):
            # project node position (x,z)
            try:
                px, _, pz = n.point3d
                u = int((px - xmin) * scale_x)
                v = int((pz - zmin) * scale_z)
                v = (size-1) - v
                u = max(0, min(size-1, u))
                v = max(0, min(size-1, v))
                name_str = str(getattr(n, 'name', '') or '')
                parts = name_str.split()
                label = parts[0][:8] if parts else ''
                cv2.putText(canvas, label, (u+5, v-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(canvas, label, (u+5, v-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
            except Exception:
                continue

    # Legend axes
    cv2.putText(canvas, 'X-> (right)', (4, size-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40,40,40), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Z^ (forward)', (size-110, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40,40,40), 1, cv2.LINE_AA)
    if local_graph is not None and getattr(local_graph, 't_sec', None) is not None:
        cv2.putText(canvas, f"t={local_graph.t_sec:.2f}s", (4,14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
    return canvas

__all__ = ["render_bev"]

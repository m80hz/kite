"""Unified keyframe selection & representation.

This module introduces:
 - Keyframe dataclass: holds frame index, time (sec), motion metrics, and lazily-computed
   perception artifacts (detections, depth, local 3D scene graph, point cloud) plus helpers to
   visualize RGB, depth, or fused point-cloud with object centroids and relations.
 - KeyframeSelector: strategies ('motion','uniform','mixed') for selecting up to N keyframes
   using optical flow magnitude (Farneback) to score motion significance.
 - Motion summary helpers producing textual tokens for HTATC context.

All downstream evaluation (QA, narratives) can consume the returned keyframes directly.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
import cv2, math, numpy as np

from .optflow import compute_flow_magnitude_sequence
from .segmenter import get_fps
from ..perception.camera import CameraIntrinsics
from ..perception.depth3d import DepthEstimator
from ..perception.detector_openvocab import OpenVocabDetector
from ..perception.scene_graph3d import build_local_graph3d, graph3d_to_text_with_prefix

MotionLevel = Literal['low','medium','high']

@dataclass
class Keyframe:
    video_path: str
    frame_idx: int
    time_sec: float
    motion_score: float  # normalized 0..1 within selected keyframes
    motion_level: MotionLevel
    ovd_backend: str = 'auto'
    yolo_weights: Optional[str] = None
    image: Optional[np.ndarray] = None
    detections: Optional[List[Dict[str,Any]]] = None
    depth: Optional[np.ndarray] = None
    local_graph: Optional[Any] = None
    _cam: Optional[CameraIntrinsics] = None
    bev_image: Optional[np.ndarray] = None  # stored per-keyframe BEV (top-down) once computed

    # -------- Lazy resources --------
    def load_image(self) -> np.ndarray:
        if self.image is not None:
            return self.image
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        ok, fr = cap.read(); cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read frame {self.frame_idx} from {self.video_path}")
        self.image = fr
        return fr

    def camera(self) -> CameraIntrinsics:
        if self._cam is None:
            fr = self.load_image()
            H,W = fr.shape[:2]
            self._cam = CameraIntrinsics.from_image_size(W, H, fov_deg=180.0/(math.pi))
        return self._cam

    def ensure_detections(self, text_queries: Optional[List[str]]=None) -> List[Dict[str,Any]]:
        if self.detections is not None:
            return self.detections
        fr = self.load_image()
        det = OpenVocabDetector(backend=self.ovd_backend, yolo_weights=self.yolo_weights)
        self.detections = det.detect(fr, text_queries=text_queries or [
            "robot arm .","gripper .","end effector .","tool .","object .","cup .","mug .","drawer .","microwave .","bottle ."
        ])
        return self.detections

    def ensure_depth(self, depth_estimator: Optional[DepthEstimator]=None) -> np.ndarray:
        if self.depth is not None:
            return self.depth
        if depth_estimator is None:
            depth_estimator = DepthEstimator(pred_is_inverse=True)
        self.depth = depth_estimator.predict(self.load_image())
        return self.depth

    def ensure_local_graph(self, depth_estimator: Optional[DepthEstimator]=None):
        if self.local_graph is not None:
            return self.local_graph
        dets = self.ensure_detections()
        depth = self.ensure_depth(depth_estimator)
        cam = self.camera()
        self.local_graph = build_local_graph3d(dets, depth, cam, t_sec=self.time_sec, frame_idx=self.frame_idx)
        return self.local_graph

    def local_graph_text(self, depth_estimator: Optional[DepthEstimator]=None) -> str:
        g = self.ensure_local_graph(depth_estimator)
        return graph3d_to_text_with_prefix(g, prefix=f"[LOCAL_SCENE t={self.time_sec:.2f}s] ")

    # -------- Visualization helpers --------
    def pointcloud(self, depth_estimator: Optional[DepthEstimator]=None):
        depth = self.ensure_depth(depth_estimator)
        fr = self.load_image()
        cam = self.camera()
        H,W = depth.shape
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        Z = depth
        X = (us - cam.cx) * Z / cam.fx
        Y = (vs - cam.cy) * Z / cam.fy
        pts = np.stack([X,Y,Z], axis=-1).reshape(-1,3).astype(np.float32)
        cols = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).reshape(-1,3).astype(np.float32)/255.0
        valid = Z.reshape(-1) > 1e-6
        return pts[valid], cols[valid]

    def visualize_pointcloud(self, save_png: Optional[str]=None, show: bool=False):
        pts, cols = self.pointcloud()
        g = self.ensure_local_graph()
        node_centers = []
        for n in g.nodes:
            cx,cy,cz = n.point3d
            node_centers.append([float(cx), float(cy), float(cz)])
        edges = [(e.subj, e.obj, e.pred) for e in g.edges]
        try:
            import pyvista as pv
            plotter = pv.Plotter(off_screen=not show)
            pc = pv.PolyData(pts)
            pc['RGB'] = (cols*255).astype(np.uint8)
            plotter.add_mesh(pc, scalars='RGB', rgb=True, point_size=2.0, render_points_as_spheres=True)
            for c in node_centers:
                plotter.add_mesh(pv.Sphere(radius=0.05, center=c), color='white')
            for (a,b,pred) in edges:
                p0 = np.array(node_centers[a]); p1 = np.array(node_centers[b])
                line = pv.Line(p0, p1)
                plotter.add_mesh(line, color=('red' if pred in ('in_front_of') else 'green'), line_width=3)
            if save_png:
                plotter.show(screenshot=save_png, auto_close=not show)
            elif show:
                plotter.show()
            return
        except Exception:
            pass
        # fallback to vedo
        try:
            import vedo as vd
            vd.settings.default_backend='vtk'
            actors=[vd.Points(pts, r=3).pointColors(cols)]
            for c in node_centers:
                actors.append(vd.Sphere(pos=c, r=0.02, c='white'))
            for (a,b,pred) in edges:
                actors.append(vd.Line(node_centers[a], node_centers[b]).lw(3).c('red' if pred in ('in_front_of') else 'green'))
            plt = vd.Plotter(size=(1280,720), bg='white')
            plt.show(actors, interactive=False)
            if save_png:
                plt.screenshot(save_png)
            if show:
                plt.interactive().close()
            else:
                plt.close()
        except Exception:
            raise RuntimeError('No visualization backend available (pyvista/vedo)')

    def rgb(self):
        return self.load_image()

    def depth_normalized(self, depth_estimator: Optional[DepthEstimator]=None):
        d = self.ensure_depth(depth_estimator)
        mn, mx = float(d.min()), float(d.max())
        if mx - mn < 1e-6:
            return np.zeros_like(d)
        return (d - mn)/(mx - mn)

    def rgbd_concat(self, depth_estimator: Optional[DepthEstimator]=None):
        rgb = self.rgb()
        dn = self.depth_normalized(depth_estimator)
        dn_u8 = (np.clip(dn*255.0,0,255)).astype(np.uint8)
        dn_color = cv2.applyColorMap(dn_u8, cv2.COLORMAP_INFERNO)
        return cv2.hconcat([rgb, dn_color])


class KeyframeSelector:
    def __init__(self, strategy: Literal['motion','uniform','mixed']='motion', max_keyframes:int=5, stride:int=2):
        self.strategy = strategy
        self.max_keyframes = max_keyframes
        self.stride = stride

    def _uniform_indices(self, count: int) -> List[int]:
        if count <= 0: return []
        k = min(self.max_keyframes, count)
        return sorted({int(i) for i in np.linspace(0, count-1, k)})

    def select(self, video_path: str) -> List[Keyframe]:
        mags, idxs = compute_flow_magnitude_sequence(video_path, stride=self.stride)
        if len(idxs)==0:
            return []
        fps = get_fps(video_path) or 30.0
        # Normalize motion scores
        if len(mags)==0:
            mnorm = np.zeros((len(idxs),), dtype=np.float32)
        else:
            mnorm = (mags - mags.min())/(mags.max()-mags.min()+1e-6)
        order_motion = list(np.argsort(-mnorm))
        chosen: List[int] = []
        if self.strategy in ('motion','mixed'):
            for i in order_motion:
                if len(chosen) >= self.max_keyframes: break
                if all(abs(i-j)>5 for j in chosen):
                    chosen.append(i)
        if self.strategy=='uniform' or (self.strategy=='mixed' and len(chosen)<self.max_keyframes):
            need = self.max_keyframes - len(chosen)
            if need>0:
                for u in self._uniform_indices(len(idxs)):
                    if len(chosen) >= self.max_keyframes: break
                    if u not in chosen:
                        chosen.append(u)
        chosen = sorted(chosen)
        if not chosen:
            return []
        sel_scores = mnorm[chosen]
        low_th = np.quantile(sel_scores, 0.33)
        high_th = np.quantile(sel_scores, 0.66)
        kfs: List[Keyframe] = []
        for ci in chosen:
            sc = float(mnorm[ci])
            if sc <= low_th: lvl='low'
            elif sc >= high_th: lvl='high'
            else: lvl='medium'
            fi = idxs[ci]
            kfs.append(Keyframe(video_path, fi, fi/fps, sc, lvl))
        return kfs

    @staticmethod
    def motion_summary(kfs: List[Keyframe]) -> str:
        if not kfs:
            return 'motion_summary: none'
        counts={'low':0,'medium':0,'high':0}
        for k in kfs: counts[k.motion_level]+=1
        avg = sum(k.motion_score for k in kfs)/len(kfs)
        return f"motion_summary: avg={avg:.2f} counts(low={counts['low']},med={counts['medium']},high={counts['high']})"

    @staticmethod
    def motion_line(kfs: List[Keyframe]) -> str:
        return ' '.join([f"t={k.time_sec:.2f}:{k.motion_level}" for k in kfs])

# Convenience function to preserve external dependency names used elsewhere
def extract_frame_at_time(video_path: str, t: float):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, t*fps)))
    ok, fr = cap.read(); cap.release()
    if not ok:
        return np.zeros((512,512,3), dtype=np.uint8)
    return fr

def montage_1xN(frames: List[np.ndarray], labels: List[str]=None) -> np.ndarray:
    if labels is None:
        labels = [f"k{i}" for i in range(len(frames))]
    if not frames:
        return np.zeros((512,512,3), dtype=np.uint8)
    # Normalize label count to number of frames (avoid IndexError if caller passed fewer)
    if len(labels) < len(frames):
        labels = labels + [f"k{i}" for i in range(len(labels), len(frames))]
    elif len(labels) > len(frames):
        labels = labels[:len(frames)]
    H = max(fr.shape[0] for fr in frames)
    W = max(fr.shape[1] for fr in frames)
    frames_r = [cv2.resize(fr, (W,H)) if (fr.shape[0]!=H or fr.shape[1]!=W) else fr for fr in frames]
    canvas = np.zeros((H, W*len(frames_r), 3), dtype=np.uint8)
    for i, fr in enumerate(frames_r):
        canvas[:, i*W:(i+1)*W, :] = fr
        if i < len(labels):
            cv2.putText(canvas, labels[i][:18], (i*W+8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    return canvas

def save_montage_image(img, out_path: str):
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path

# ---------------- BEV montage helpers ----------------
def bev_montage_1xN(bev_frames: List[np.ndarray], labels: List[str]=None) -> np.ndarray:
    """Create a horizontal montage of BEV (bird's-eye) frames.

    Mirrors montage_1xN but kept separate for clarity and possible future styling
    differences (e.g., background, legend). Accepts any list of 2D/3-channel images.
    """
    if labels is None:
        labels = [f"bev{i}" for i in range(len(bev_frames))]
    if not bev_frames:
        return np.zeros((256,256,3), dtype=np.uint8)
    # Normalize to 3-channel BGR uint8
    norm_frames = []
    for fr in bev_frames:
        if fr is None:
            continue
        f = fr
        if f.ndim == 2:  # grayscale
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        if f.dtype != np.uint8:
            f = np.clip(f,0,255).astype(np.uint8)
        norm_frames.append(f)
    if not norm_frames:
        return np.zeros((256,256,3), dtype=np.uint8)
    H = max(fr.shape[0] for fr in norm_frames)
    W = max(fr.shape[1] for fr in norm_frames)
    frames_r = [cv2.resize(fr, (W,H)) if (fr.shape[0]!=H or fr.shape[1]!=W) else fr for fr in norm_frames]
    canvas = np.full((H, W*len(frames_r), 3), 255, dtype=np.uint8)
    for i, fr in enumerate(frames_r):
        x0 = i*W; x1 = (i+1)*W
        canvas[:, x0:x1, :] = fr
        # Draw black border rectangle around tile
        cv2.rectangle(canvas, (x0,0), (x1-1,H-1), (0,0,0), 2, cv2.LINE_AA)
        if i < len(labels):
            cv2.putText(canvas, labels[i][:16], (x0+6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(canvas, labels[i][:16], (x0+6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return canvas

def save_bev_montage_image(img, out_path: str):
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path

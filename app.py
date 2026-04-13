#!/usr/bin/env python3
"""Interactive Gradio demo for the evaluation pipeline (single-video showcase).
"""
from typing import List, Dict, Any, Tuple, Optional
import os, math, time, asyncio, tempfile
import cv2
import numpy as np
import gradio as gr
import matplotlib
import plotly.graph_objects as go

# Import project modules (assumes running from repo root)
from kite.qa import get_adapter
from kite.qa.prompt_augment import augment_question
from kite.qa.parse import parse_mcq_label
from kite.qa.locate_parse import parse_locate_time_conf
from kite.qa.failure_locating import build_failure_locating_prompt, parse_failure_candidates
from kite.context.robot_profile import RobotProfile
from kite.context.htatc import build_htatc
from kite.perception.camera import CameraIntrinsics
from kite.perception.depth3d import DepthEstimator
from kite.perception.detector_openvocab import OpenVocabDetector
from kite.perception.scene_graph3d import build_local_graph3d, graph3d_to_text, graph3d_to_json
from kite.perception.global_scene_graph import GlobalSceneGraphAggregator, CameraExtrinsics
from kite.video.keyframes import extract_frame_at_time, montage_1xN, KeyframeSelector, bev_montage_1xN
from kite.narrative.final_summarizer import generate_final_narrative

try:
    # Optional BEV renderer (if available)
    from kite.perception.bev import render_bev  # type: ignore
except Exception:
    render_bev = None  # type: ignore

# -------------------- Logging --------------------
def log(msg: str):
    ts = time.strftime('%H:%M:%S')
    print(f"[APP {ts}] {msg}", flush=True)

# -------------------- Config --------------------
SAMPLE_VIDEOS = []  # will be populated from datasets/* recursively (mp4)
for root,dirs,files in os.walk('datasets'):
    for fn in files:
        if fn.lower().endswith(('.mp4','.mov','.mkv')):
            SAMPLE_VIDEOS.append(os.path.join(root, fn))
SAMPLE_VIDEOS = sorted(SAMPLE_VIDEOS)[:12]

DEFAULT_MODEL_NAME = os.environ.get('KITE_MODEL_NAME', 'Qwen/Qwen2.5-VL-7B-Instruct')
# If user does not provide an explicit API base URL, fall back to local vLLM default
_fallback_api = 'http://127.0.0.1:8000/v1'
DEFAULT_MODEL_URL = os.environ.get('KITE_MODEL_URL', _fallback_api)
DEFAULT_ROBOT_PROFILE = os.environ.get('KITE_ROBOT_PROFILE', 'examples/robot_profiles/dart_dual_arm.json')

STANDARD_QUESTIONS = [
    'Task identification',
    'Task planning',
    'Failure detection',
    'Failure identification',
    'Failure locating',
    'Failure explanation',
    'High-level correction',
    'Low-level correction'
]

# -------------------- Helpers --------------------

FAST_DET_LIMIT = 12  # draw up to N boxes per keyframe


def _detect_and_depth_with_keyframes(
    video_path: str,
    enable_3d: bool = True,
    max_keyframes: int = 5,
    stride: int = 2,
    ovd_backend: str = 'groundingdino',
    yolo_weights: Optional[str] = None,
    enable_bev_maps: bool = True,
    progress: Optional[gr.Progress] = None
):
    """Runs detection and optional depth/3D aggregation on selected keyframes.

    Returns:
    - keyframe_times: List[float]
    - det_gallery: List[(rgb_bgr, caption)]
    - local_graph_texts: List[str]
    - global_scene_text: str (JSON)
    - pc_payload: dict(depth, rgb, g_local, cam)
    - bev_images: List[(t, bev_img or None)]
    - depth_images: List[(t, depth_colored or None)]
    - flow_images: List[(t, flow_overlay or None)]
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, frame0 = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError('Failed to read video')
    H, W = frame0.shape[:2]
    cam = CameraIntrinsics.from_image_size(W, H, fov_deg=180.0/(math.pi))

    # Unified keyframe selection
    selector = KeyframeSelector(strategy='motion', max_keyframes=max_keyframes, stride=stride)
    keyframes = selector.select(video_path)
    if not keyframes:
        # fallback: center frame
        nframes = 0
        try:
            cap = cv2.VideoCapture(video_path)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            pass
        keyframes = [type('KFPlaceholder',(object,),{'time_sec':1.0, 'frame_idx':0})()]
    keyframe_times = [float(kf.time_sec) for kf in keyframes if hasattr(kf, 'time_sec')]

    # Perception modules
    ovdet = OpenVocabDetector(backend=ovd_backend, yolo_weights=yolo_weights)
    depth_est = DepthEstimator(pred_is_inverse=True) if enable_3d else None
    g_agg = GlobalSceneGraphAggregator()

    det_gallery: List[Tuple[np.ndarray, str]] = []
    local_graph_texts: List[str] = []
    bev_images: List[Tuple[float, Optional[np.ndarray]]] = []
    depth_images: List[Tuple[float, Optional[np.ndarray]]] = []  # colored depth for UI per keyframe
    flow_images: List[Tuple[float, Optional[np.ndarray]]] = []   # optical flow overlay per keyframe (vs previous)
    last_depth_map = None
    last_rgb_frame = None
    last_g_local = None
    prev_for_flow: Optional[np.ndarray] = None

    for i, t in enumerate(keyframe_times[:max_keyframes]):
        if progress:
            progress(((i+1)/max(1, len(keyframe_times))), desc=f"Perception keyframe {i+1}/{len(keyframe_times)} @ {t:.2f}s")
        fr = extract_frame_at_time(video_path, t)
        # Optical flow overlay vs previous keyframe (if any)
        try:
            if prev_for_flow is None:
                flow_images.append((t, None))
            else:
                prev_gray = cv2.cvtColor(prev_for_flow, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(fr)
                hsv[..., 1] = 255
                # angle to hue (0..180)
                hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
                # magnitude to value (0..255)
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                overlay = cv2.addWeighted(fr, 0.4, flow_bgr, 0.6, 0)
                flow_images.append((t, overlay))
        except Exception as e:
            log(f"Flow overlay error @ t={t:.2f}s: {e}")
            flow_images.append((t, None))
        prev_for_flow = fr
        # Match full_eval query set (note trailing periods as in dataset prompts)
        # dets = ovdet.detect(fr, text_queries=[
        #     "robot arm .","gripper .","tool .","cup .","mug .","bottle .","knife .","can .","fork .","spoon .",
        #     "microwave .","cube .","cylinder ."
        # ])
        dets = ovdet.detect(fr, text_queries=[
            "robot arm .","gripper .","bottle .","mug .","ball .","knife .","fork .","spoon .","fruit .",
            "microwave .","cube .","box ."
        ])
        # Draw detections
        det_vis = fr.copy()
        for d in dets[:FAST_DET_LIMIT]:
            x1, y1, x2, y2 = map(int, d['bbox'])
            cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(det_vis, d['name'], (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        det_gallery.append((det_vis[:, :, ::-1], f"t={t:.2f}s"))  # RGB for UI

        # 3D / BEV
        if enable_3d and depth_est is not None:
            try:
                dmap = depth_est.predict(fr)
                dmin, dmax, dmean = float(np.min(dmap)), float(np.max(dmap)), float(np.mean(dmap))
                frame_idx = int(t * fps)
                g_local = build_local_graph3d(dets, dmap, cam, t_sec=t, frame_idx=frame_idx)
                local_graph_texts.append(f"[LOCAL_SCENE t={t:.2f}s] " + graph3d_to_text(g_local))
                g_agg.update_from_local(g_local, CameraExtrinsics.identity())
                last_depth_map = dmap
                last_rgb_frame = fr
                last_g_local = g_local
                log(f"Depth OK t={t:.2f}s dmin={dmin:.3f} dmean={dmean:.3f} dmax={dmax:.3f} nodes={len(g_local.nodes)}")
                # Depth visualization (colorized)
                try:
                    d0, d1 = float(np.min(dmap)), float(np.max(dmap))
                    if not np.isfinite(d0) or not np.isfinite(d1) or d1 <= d0:
                        dnorm = np.zeros_like(dmap, dtype=np.float32)
                    else:
                        dnorm = (dmap - d0) / max(1e-6, (d1 - d0))
                    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
                    colored = (cmap((dnorm * 255.0).astype(np.uint8))[:, :, :3] * 255).astype(np.uint8)
                    depth_images.append((t, colored))
                except Exception:
                    depth_images.append((t, None))
                # BEV render if available
                bev_img = None
                if enable_bev_maps and render_bev is not None:
                    try:
                        bev_img = render_bev(g_local, g_agg, size=256)
                    except Exception as e:
                        log(f"BEV render error @ t={t:.2f}s: {e}")
                bev_images.append((t, bev_img))
            except Exception as e:
                log(f"3D step failed at t={t:.2f}s: {e}")
                local_graph_texts.append(f"[LOCAL_SCENE t={t:.2f}s] [SCENE2D ONLY] " + \
                    ", ".join(d['name'] for d in dets[:FAST_DET_LIMIT]))
                bev_images.append((t, None))
                depth_images.append((t, None))
        else:
            # 2D-only textual summary
            local_graph_texts.append(f"[LOCAL_SCENE t={t:.2f}s] [SCENE2D ONLY] " + \
                ", ".join(d['name'] for d in dets[:FAST_DET_LIMIT]))
            bev_images.append((t, None))
            depth_images.append((t, None))

    # Global scene JSON string
    try:
        global_scene_text = g_agg.to_json_str(max_tracks=15, max_rels=30)
    except Exception:
        global_scene_text = graph3d_to_json({'nodes': [], 'relations': []})  # minimal fallback

    pc_payload = {'depth': last_depth_map, 'rgb': last_rgb_frame, 'g_local': last_g_local, 'cam': cam}
    return keyframe_times, det_gallery, local_graph_texts, global_scene_text, pc_payload, bev_images, depth_images, flow_images, g_agg


def _build_context(robot_prof: RobotProfile, plan_steps: List[str], global_scene_text: str, keyframe_times: List[float], enable_groups: Optional[Dict[str,bool]] = None):
    # Build structured HTATC aligned with full_eval
    motion_line = 'motion=' + (';'.join([f"{i}:n/a" for i in range(len(keyframe_times))]) if keyframe_times else 'n/a')
    return build_htatc(
        robot_prof.as_prompt_text(),
        plan_steps,
        global_scene_text,
        keyframe_times,
        motion_contact_line=motion_line,
        enable_groups=(enable_groups or {k: True for k in ['ROBOT','PLAN','SCENE3D','KEYFRAMES']}),
        structured=True
    )

async def _ask(adapter, model_name, model_url, frames, question, context):
    return await adapter.qa_with_images_and_context(model_name, model_url, frames, question, context)

# -------------------- Core processing --------------------

def process_video(video_path: str, model_name: str, model_url: str, robot_profile_path: Optional[str], enable_3d: bool, enable_bev_maps: bool, progress: Optional[gr.Progress]=None):
    log(f"Process start video={video_path} model={model_name} enable_3d={enable_3d}")
    t0 = time.time()
    if model_url and not model_url.startswith('http'):
        log(f"Model URL '{model_url}' doesn't look like HTTP; using {_fallback_api}.")
        model_url = _fallback_api
    adapter = get_adapter(model_name)
    robot_prof = RobotProfile.load(robot_profile_path)

    if progress:
        progress(0.05, desc="Selecting keyframes")
    keyframe_times, det_gallery, local_graph_texts, global_scene_text, pc_payload, bev_images, depth_images, flow_images, g_agg = _detect_and_depth_with_keyframes(
        video_path, enable_3d=enable_3d, max_keyframes=5, stride=2, ovd_backend='groundingdino', yolo_weights=None, enable_bev_maps=enable_bev_maps, progress=progress)

    if progress:
        progress(0.55, desc="Task planning")
    # Use all keyframes for planning prompt visuals
    frames_tp = [extract_frame_at_time(video_path, t) for t in keyframe_times]
    steps: List[str] = []
    raw_planning_q = "<image>\nIn the video, the robotic arm executes a task. Describe the task steps clearly. Provide a numbered list of concise high-level steps."
    tp_q = augment_question(model_name, 'Task planning', raw_planning_q)
    try:
        async def _plan():
            return await _ask(adapter, model_name, model_url, frames_tp, tp_q, robot_prof.as_prompt_text())
        resp = asyncio.run(_plan())
        lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
        steps = [ln.split('.',1)[1].strip() for ln in lines if ln[:2].isdigit() and '.' in ln[:3]]
        if not steps and resp:
            steps = [resp[:200]]
    except Exception as e:
        steps = [f'(planning error: {e})']

    if progress:
        progress(0.75, desc="Building context")
    htatc = _build_context(robot_prof, steps, global_scene_text, keyframe_times)

    if progress:
        progress(0.90, desc="Preparing montages")
    # RGB montage of all keyframes (as in full_eval narrative inputs)
    all_rgb_frames = [extract_frame_at_time(video_path, t) for t in keyframe_times]
    rgb_labels = [f"t={t:.2f}" for t in keyframe_times]
    montage_rgb = None
    try:
        if all_rgb_frames:
            montage_rgb = montage_1xN(all_rgb_frames, labels=rgb_labels)[:, :, ::-1]
    except Exception as e:
        log(f"RGB montage error: {e}")

    # BEV montage in the same order as keyframes
    bev_montage_img = None
    if enable_bev_maps and any(im is not None for _, im in bev_images):
        try:
            bev_ordered = [im for _, im in bev_images if im is not None]
            if bev_ordered:
                bev_labels = [f"BEV t={t:.2f}" for t, im in bev_images if im is not None]
                bev_montage_img = bev_montage_1xN(bev_ordered, labels=bev_labels)
                if bev_montage_img is not None:
                    bev_montage_img = bev_montage_img[:, :, ::-1]
        except Exception as e:
            log(f"BEV montage error: {e}")

    # Optical Flow montage (same keyframe order; skip Nones)
    flow_montage_img = None
    try:
        flow_ordered = [im for _, im in flow_images if im is not None]
        if flow_ordered:
            flow_labels = [f"FLOW t={t:.2f}" for t, im in flow_images if im is not None]
            flow_montage_img = montage_1xN(flow_ordered, labels=flow_labels)
            if flow_montage_img is not None:
                flow_montage_img = flow_montage_img[:, :, ::-1]
    except Exception as e:
        log(f"Flow montage error: {e}")

    # Depth montage (same keyframe order; skip Nones)
    depth_montage_img = None
    try:
        # depth_images contain RGB images; convert to BGR for montage utility, then back to RGB for UI
        depth_ordered_bgr = [im[:, :, ::-1] for _, im in depth_images if im is not None]
        if depth_ordered_bgr:
            depth_labels = [f"DEPTH t={t:.2f}" for t, im in depth_images if im is not None]
            depth_montage_img = montage_1xN(depth_ordered_bgr, labels=depth_labels)
            # if depth_montage_img is not None:
            #     depth_montage_img = depth_montage_img[:, :, ::-1]
    except Exception as e:
        log(f"Depth montage error: {e}")

    # Detections montage (use det_gallery images; they are RGB in UI)
    det_montage_img = None
    try:
        det_ordered_bgr = [img[:, :, ::-1] for (img, cap) in det_gallery if img is not None]
        if det_ordered_bgr:
            det_labels = [f"DET {cap}" for (_img, cap) in det_gallery if _img is not None]
            det_montage_img = montage_1xN(det_ordered_bgr, labels=det_labels)
            if det_montage_img is not None:
                det_montage_img = det_montage_img[:, :, ::-1]
    except Exception as e:
        log(f"Detections montage error: {e}")

    # Depth gallery (per keyframe)
    depth_gallery = []
    try:
        depth_gallery = [((im[:, :, ::-1] if im is not None else None), f"t={t:.2f}s") for t, im in depth_images if im is not None]
    except Exception:
        depth_gallery = []

    # Optical Flow gallery (per keyframe)
    flow_gallery = []
    try:
        flow_gallery = [((im[:, :, ::-1] if im is not None else None), f"t={t:.2f}s") for t, im in flow_images if im is not None]
    except Exception:
        flow_gallery = []

    # Build an interactive point cloud via Plotly (no mesh)
    pc_plot = None
    try:
        dmap = pc_payload.get('depth'); rgb0 = pc_payload.get('rgb'); cam = pc_payload.get('cam')
        if dmap is not None and rgb0 is not None and cam is not None:
            Hc, Wc = dmap.shape
            # Subsample points to keep interactive plot light
            max_points = 20000
            stride_pc = max(1, int(np.sqrt((Hc * Wc) / max_points)))
            ys, xs = np.mgrid[0:Hc:stride_pc, 0:Wc:stride_pc]
            zs = dmap[::stride_pc, ::stride_pc]
            # backproject to camera coordinates
            X = (xs - cam.cx) * zs / cam.fx
            Y = (ys - cam.cy) * zs / cam.fy
            Z = zs
            # colors from RGB
            cols = rgb0[::stride_pc, ::stride_pc]
            cols_hex = [f"rgb({int(r)},{int(g)},{int(b)})" for b,g,r in cols.reshape(-1,3)]
            pc_trace = go.Scatter3d(
                x=X.reshape(-1), y=Y.reshape(-1), z=Z.reshape(-1),
                mode='markers', marker=dict(size=1.5, color=cols_hex), hoverinfo='none'
            )
            fig = go.Figure(data=[pc_trace])
            fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0))
            pc_plot = fig
    except Exception:
        pc_plot = None

    total_time = time.time() - t0
    log(f"Process complete in {total_time:.2f}s")

    return {
        'keyframe_times': keyframe_times,
        'detections_gallery': det_gallery,
    'bev_gallery': [((im[:, :, ::-1] if im is not None else None), f"t={t:.2f}s") for t, im in bev_images if im is not None],
    'depth_gallery': depth_gallery,
    'flow_gallery': flow_gallery,
    'bev_frames_rgb': [im for _, im in bev_images if im is not None],
        'montage_rgb': montage_rgb,
    'montage_det': det_montage_img,
    'montage_depth': depth_montage_img,
    'montage_bev': bev_montage_img,
    'montage_flow': flow_montage_img,
        'local_graph_texts': local_graph_texts,
        'global_scene_text': global_scene_text,
        'steps': steps,
        'htatc': htatc,
        'robot_profile_text': robot_prof.as_prompt_text(),
        'video_path': video_path,
        'model_name': model_name,
        'model_url': model_url,
        'elapsed_sec': total_time,
        'pc_depth': pc_payload.get('depth'),
        'pc_rgb': pc_payload.get('rgb'),
        'pc_g_local': pc_payload.get('g_local'),
    'pc_cam': pc_payload.get('cam'),
    'pc_plot': pc_plot
    }

# -------------------- QA interaction --------------------

def compose_question(state: Dict[str,Any], question_type: str) -> str:
    if not state:
        return '(no state)'
    base_q = {
        'Task identification': "<image>\nWhat is the robot doing in the video? Please describe its task.",
        'Task planning': "<image>\nIn the video, the robotic arm executes a task. Please break down its execution into a sequence of subtasks.",
        'Failure detection': "<image>\nBased on the video of the robotic arm executing a task, did it finish the task successfully?",
        'Failure identification': "<image>\nFrom the video of the robotic arm performing a task, what kind of error can be observed during the task? (Your answer should choose one of the following options: ['Orientation deviation.', 'Grasping error.', 'Position deviation.'])",
        # For locating, we show a lightweight preview; the actual question sent will be build_failure_locating_prompt()+legend
        'Failure locating': "<image>\nYou will see all keyframes (RGB), then BEV diagrams. Identify up to 3 candidate failure frames (0..N-1). Return JSON {\"candidates\":[{\"frame_num\": int, \"confidence\": float}]}",
        'Failure explanation': "<image>\nThis is a video of a robotic arm performing a task, please explain in detail the reason for the task failure.",
        'High-level correction': "<image>\nThis is a video of a robotic arm performing a task, an error occurred during execution. Provide high-level corrective instructions to help the robot recover and complete the task successfully.",
        'Low-level correction': "<image>\nBased on the video, an error happened while the robot was executing a task, give detailed low-level instructions to correct the issue and allow the task to be finished."
    }.get(question_type, question_type)
    if question_type.lower().startswith('failure locating'):
        legend = ""
        try:
            if state and 'keyframe_times' in state and state['keyframe_times']:
                legend = "Candidates: " + ", ".join([f"{i}:{t:.2f}s" for i, t in enumerate(state['keyframe_times'])])
        except Exception:
            legend = ""
        return build_failure_locating_prompt() + ("\n" + legend if legend else "")
    else:
        return augment_question(state['model_name'], question_type, base_q)

def run_question(state: Dict[str,Any], question_type: str):
    if not state:
        return '(no state)', state, '(no state)'
    adapter = get_adapter(state['model_name'])
    qtext = compose_question(state, question_type)
    # Build frames: all RGB keyframes then BEV diagrams (if any) for ALL QAs
    frames_rgb = [extract_frame_at_time(state['video_path'], t) for t in state['keyframe_times']]
    bev_frames = []
    try:
        # Prefer raw BEV frames list if present
        if 'bev_frames_rgb' in state and state['bev_frames_rgb']:
            bev_frames = [im for im in state['bev_frames_rgb'] if im is not None]
        elif 'bev_gallery' in state and state['bev_gallery']:
            # bev_gallery is [(rgb, caption)] — extract images
            bev_frames = [img for (img, _cap) in state['bev_gallery'] if img is not None]
        else:
            bev_frames = []
    except Exception:
        bev_frames = []
    frames = frames_rgb + bev_frames

    # Context: optional BEV note + HTATC for ALL QAs
    bev_note = ""
    if bev_frames:
        bev_note = (
            "[pseudo Bird's Eye View layout: After ALL RGB keyframes, matching bev schematic top-down layout diagrams follow in the same order. "
            "Each BEV plots objects in ground plane (X right, Z forward). Dot color = class. Circle radius is proportional to object detection confidence (larger = higher confidence)."
            "Use RGB first, then BEV for relative spatial layout. It is not to scale.]\n"
        )
    context = ((bev_note + state['htatc']) if state.get('htatc') else bev_note)
    
    try:
        qtext = qtext + " (Refer to the keyframes, and BEV diagrams in failure explanation and locating.)"
        resp = asyncio.run(_ask(adapter, state['model_name'], state['model_url'], frames, qtext, context))
    except Exception as e:
        resp = f'(error: {e})'

    # Keep label extraction for Failure identification
    if question_type.lower().startswith('failure identification'):
        lab, _ = parse_mcq_label(resp)
        if lab:
            resp = f'label={lab}; raw={resp}'

    # For Failure locating, also parse candidates (now BEV included too)
    if question_type.lower().startswith('failure locating'):
        try:
            cands = parse_failure_candidates(resp)
            resp = (resp + (f"\nParsed candidates: {cands}" if cands is not None else "\nParsed candidates: None"))
        except Exception:
            pass

    return resp, state, qtext
    pts = []
    cols = []
    for y in ys:
        for x in xs:
            Z = float(depth[y, x])
            if not np.isfinite(Z) or Z <= 0:
                continue
            X = (x - cam.cx) * Z / cam.fx
            Y = (y - cam.cy) * Z / cam.fy
            pts.append((X, Y, Z))
            b, g, r = rgb[y, x]
            cols.append((int(r), int(g), int(b)))
            if len(pts) >= max_points:
                break
        if len(pts) >= max_points:
            break
    if not pts:
        return None, None
    return np.array(pts, np.float32), np.array(cols, np.uint8)

def run_arbitrary(state: Dict[str, Any], question_text: str):
    """Ask an arbitrary question using all RGB keyframes followed by BEV frames (if any).

    Returns: (answer_text, state)
    """
    if not state:
        return '(no state)', state
    adapter = get_adapter(state['model_name'])
    # Frames: RGB keyframes then BEV diagrams
    frames_rgb = [extract_frame_at_time(state['video_path'], t) for t in state.get('keyframe_times', [])]
    bev_frames = []
    try:
        if state.get('bev_frames_rgb'):
            bev_frames = [im for im in state['bev_frames_rgb'] if im is not None]
        elif state.get('bev_gallery'):
            bev_frames = [img for (img, _cap) in state['bev_gallery'] if img is not None]
    except Exception:
        bev_frames = []
    frames = frames_rgb + bev_frames

    # Context: BEV note + HTATC if available
    bev_note = ""
    if bev_frames:
        bev_note = (
            "[pseudo Bird's Eye View layout: After ALL RGB keyframes, matching bev schematic top-down layout diagrams follow in the same order. "
            "Each BEV plots objects in ground plane (X right, Z forward). Dot color = class. Circle radius is proportional to object detection confidence (larger = higher confidence)."
            "Use RGB first, then BEV for relative spatial layout. It is not to scale.]\n"
        )
    context = ((bev_note + state['htatc']) if state.get('htatc') else bev_note)

    qtext = question_text or ''
    try:
        resp = asyncio.run(_ask(adapter, state['model_name'], state['model_url'], frames, qtext, context))
    except Exception as e:
        resp = f'(error: {e})'
    return resp, state

def _build_pointcloud_arrays(depth: np.ndarray, rgb: np.ndarray, cam: CameraIntrinsics, max_points: int = 200_000):
    H, W = depth.shape
    # Subsample adaptively
    stride = max(1, int(max(H, W) / 360))
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    pts = []
    cols = []
    for y in ys:
        for x in xs:
            Z = float(depth[y, x])
            if not np.isfinite(Z) or Z <= 0:
                continue
            X = (x - cam.cx) * Z / cam.fx
            Y = (y - cam.cy) * Z / cam.fy
            pts.append((X, Y, Z))
            b, g, r = rgb[y, x]
            cols.append((int(r), int(g), int(b)))
            if len(pts) >= max_points:
                break
        if len(pts) >= max_points:
            break
    if not pts:
        return None, None
    return np.array(pts, np.float32), np.array(cols, np.uint8)

def _write_ply_ascii(path: str, points: np.ndarray, colors: np.ndarray):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def _export_pointcloud_ply(state: Dict[str,Any]) -> Optional[str]:
    depth = state.get('pc_depth'); rgb = state.get('pc_rgb'); cam = state.get('pc_cam')
    if depth is None or rgb is None or cam is None:
        return None
    pts, cols = _build_pointcloud_arrays(depth, rgb, cam)
    if pts is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.ply')
    tmp.close()
    # Try PyVista save first for robustness; fallback to manual writer
    try:
        import pyvista as pv
        cloud = pv.PolyData(pts)
        cloud['RGB'] = cols
        cloud.save(tmp.name)
    except Exception:
        _write_ply_ascii(tmp.name, pts, cols)
    return tmp.name

# -------------------- Gradio UI --------------------

def build_ui():
    with gr.Blocks(title='KITE Evaluation Demo', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# KITE Evaluation Demo')
        gr.Markdown('Select a video, run the pipeline, then explore detections, BEV, 3D scene context, and ask questions. Keyframes drive all steps.')

        with gr.Row():
            video_dd = gr.Dropdown(label='Sample Video', choices=SAMPLE_VIDEOS, value=SAMPLE_VIDEOS[0] if SAMPLE_VIDEOS else None)
            upload = gr.File(label='Or Upload Video', file_types=['.mp4','.mov','.mkv'])
            model_name_in = gr.Textbox(label='Model Name', value=DEFAULT_MODEL_NAME)
            model_url_in = gr.Textbox(label='Model URL', value=DEFAULT_MODEL_URL)
            robot_prof_in = gr.Textbox(label='Robot Profile Path (optional)', value=DEFAULT_ROBOT_PROFILE)
            enable_3d_ck = gr.Checkbox(label='Enable 3D (depth + 3D SG)', value=True)
            enable_bev_ck = gr.Checkbox(label='Enable BEV maps', value=True)
            process_btn = gr.Button('Process Video', variant='primary')

        state = gr.State()

        with gr.Row():
            keyframe_times_out = gr.Textbox(label='Keyframe Times (s)', interactive=False)
            det_gallery = gr.Gallery(label='Detections (RGB with boxes)', columns=3, height='auto')
            flow_gallery = gr.Gallery(label='Optical Flow (per keyframe)', columns=3, height='auto')
            bev_gallery = gr.Gallery(label='BEV (per keyframe)', columns=3, height='auto')
            depth_gallery = gr.Gallery(label='Depth (per keyframe)', columns=3, height='auto')
            local_graph_txt = gr.Textbox(label='Local Scene Graphs (text)', lines=8)
            global_graph_txt = gr.Textbox(label='Global Scene (JSON)', lines=6)
            steps_txt = gr.Textbox(label='Plan Steps', lines=4)
            htatc_txt = gr.Textbox(label='HTATC Context', lines=16)
            status_txt = gr.Textbox(label='Status', value='Idle', interactive=False)

        with gr.Row():
            montage_rgb_img = gr.Image(label='All Keyframes (RGB montage)', type='numpy')
            montage_flow_img = gr.Image(label='All Keyframes (Flow montage)', type='numpy')
            montage_det_img = gr.Image(label='All Keyframes (Detections montage)', type='numpy')

        with gr.Row():
            montage_depth_img = gr.Image(label='All Keyframes (Depth montage)', type='numpy')
            montage_bev_img = gr.Image(label='All Keyframes (BEV montage)', type='numpy')

        # 3D point cloud (interactive)
        with gr.Row():
            pc_export_btn = gr.Button('Export PLY (Point Cloud)')
            pc_model = gr.Model3D(label='PLY Preview (download to view)')
            pc_plot = gr.Plot(label='Interactive Point Cloud (Plotly)')

        # Narrative and QA
        with gr.Row():
            narrative_btn = gr.Button('Generate Narrative')
            narrative_out = gr.Textbox(label='Narrative Summary', lines=10)

        with gr.Row():
            qa_dropdown = gr.Dropdown(label='Question Type', choices=STANDARD_QUESTIONS, value=STANDARD_QUESTIONS[0])
            ask_btn = gr.Button('Ask Selected QA')
        qa_question_preview = gr.Textbox(label='Composed Question (sent to model)', lines=8)
        qa_answer = gr.Textbox(label='QA Answer', lines=6)

        arbitrary_q = gr.Textbox(label='Arbitrary Question')
        arbitrary_btn = gr.Button('Ask Arbitrary')
        arbitrary_answer = gr.Textbox(label='Arbitrary Answer', lines=6)

        def _on_process(sample_video, upload_file, model_name, model_url, robot_profile_path, enable_3d_flag, enable_bev_flag, progress=gr.Progress(track_tqdm=False)):
            path = sample_video
            if upload_file is not None:
                path = upload_file.name
            if not path:
                return None, 'No video selected.'
            data = process_video(path, model_name, model_url, robot_profile_path or None, enable_3d_flag, enable_bev_flag, progress=progress)
            return data, f"Done: keyframes={len(data['keyframe_times'])} steps={len(data['steps'])}"

        def _unpack(result):
            data, status = result
            if not data:
                return ("", [], [], [], [], "", "", "", "", None, None, None, None, None, None, status, None)
            return (
                ', '.join(f"{t:.2f}" for t in data['keyframe_times']),
                data['detections_gallery'],
                data['bev_gallery'],
                data['depth_gallery'],
                data.get('flow_gallery', []),
                '\n'.join(data['local_graph_texts']),
                data['global_scene_text'],
                '\n'.join(data['steps']),
                data['htatc'],
                data['montage_rgb'],
                data.get('montage_det', None),
                data.get('montage_depth', None),
                data['montage_bev'],
                data.get('montage_flow', None),
                data.get('pc_plot', None),
                status,
                data
            )

        process_btn.click(
            lambda *args: _unpack(_on_process(*args)),
            inputs=[video_dd, upload, model_name_in, model_url_in, robot_prof_in, enable_3d_ck, enable_bev_ck],
            outputs=[keyframe_times_out, det_gallery, bev_gallery, depth_gallery, flow_gallery, local_graph_txt, global_graph_txt, steps_txt, htatc_txt, montage_rgb_img, montage_det_img, montage_depth_img, montage_bev_img, montage_flow_img, pc_plot, status_txt, state]
        )

        # Export interactive point cloud
        pc_export_btn.click(lambda st: _export_pointcloud_ply(st), inputs=[state], outputs=[pc_model])

        # Narrative generation using all RGB keyframes montage context
        def _gen_narrative(st):
            if not st:
                return '(no state)', st
            try:
                adapter = get_adapter(st['model_name'])
                frames = [extract_frame_at_time(st['video_path'], t) for t in st['keyframe_times']]
                # Minimal call; extended args optional
                txt = asyncio.run(
                    generate_final_narrative(
                        adapter,
                        st['model_name'],
                        st['model_url'],
                        frames,
                        None,
                        st['robot_profile_text'],
                        st['steps'],
                        st['htatc'],
                        intended_task='',
                        failure_detection='',
                        failure_identification='',
                        failure_candidates=None,
                        extra_context='',
                    )
                )
                return txt, st
            except Exception as e:
                return f"(narrative error: {e})", st

        narrative_btn.click(_gen_narrative, inputs=[state], outputs=[narrative_out, state])

        # Update composed question when changing dropdown
        qa_dropdown.change(compose_question, inputs=[state, qa_dropdown], outputs=[qa_question_preview])
        # Ask selected QA (also refresh composed question shown)
        ask_btn.click(run_question, inputs=[state, qa_dropdown], outputs=[qa_answer, state, qa_question_preview])
        # Arbitrary QA
        arbitrary_btn.click(run_arbitrary, inputs=[state, arbitrary_q], outputs=[arbitrary_answer, state])

    return demo

if __name__ == '__main__':
    ui = build_ui()
    ui.launch(server_name='0.0.0.0', server_port=7860, share=False)
    
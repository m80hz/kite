import re, os, math, json, asyncio, random, cv2, numpy as np, torch 
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

# ----------------- Reproducibility helpers -----------------
def set_global_seed(seed: int):
    """Best-effort attempt to make evaluation deterministic.
    Note: Model generation (VLM) may remain stochastic depending on adapter settings (temperature, sampling).
    """
    if seed is None:
        return
    try:
        import random, numpy as np
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Deterministic flags
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore
        except Exception:
            pass
        try:
            import cv2
            cv2.setRNGSeed(seed)
        except Exception:
            pass
    except Exception as e:
        print(f"[SEED] warning: failed to set full determinism: {e}")

from ..qa import get_adapter
from ..qa.prompt_augment import augment_question
from ..qa.parse import parse_mcq_label
from ..video.segmenter import get_fps
from ..video.keyframes import extract_frame_at_time, montage_1xN, save_montage_image, KeyframeSelector, bev_montage_1xN, save_bev_montage_image, Keyframe
from ..perception.camera import CameraIntrinsics
from ..perception.depth3d import DepthEstimator
from ..perception.detector_openvocab import OpenVocabDetector
from ..perception.scene_graph3d import build_local_graph3d, graph3d_to_json, graph3d_to_json_str
from ..perception.global_scene_graph import GlobalSceneGraphAggregator, CameraExtrinsics
from ..narrative.final_summarizer import generate_final_narrative
from ..context.robot_profile import RobotProfile
from ..context.htatc import build_htatc
from ..qa.failure_locating import build_failure_locating_prompt, parse_failure_candidates
from ..utils.logger import Timer

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _clean_q(q: str) -> str:
    # strip and remove image tokens
    return (q or '').replace('<image>', '').strip()

def _normalize_text(s: str) -> str:
    """Normalize text for comparison and metrics.

    Steps:
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse internal whitespace
    - Remove punctuation characters (keep alphanumerics, space, and periods that are part of decimal numbers)
    - Remove standalone periods / commas / exclamation / question marks
    """
    if s is None:
        return ''
    s = s.lower().strip()
    # Preserve decimal points inside numbers; temporarily protect them
    s = re.sub(r"(\d)\.(\d)", r"\1<DECIMAL>\2", s)
    # Remove punctuation (except placeholder)
    s = re.sub(r"[^a-z0-9\s<DECIMAL>]", " ", s)
    # Restore decimal points
    s = s.replace('<decimal>', '.').replace('<DECIMAL>', '.')
    # Collapse whitespace
    s = " ".join(s.split())
    return s

# Helper to fetch first token (for MCQ lenient match)
_WORD_RE = re.compile(r"[a-z0-9']+")

def _first_token(s: str) -> str:
    toks = _WORD_RE.findall(s)
    return toks[0] if toks else ''

def _extract_first_float(s: str) -> Optional[float]:
    import re
    if not s:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

# -------------------------
"""
Text similarity metrics for descriptive QA evaluation (non-LLM):
- Exact Match (normalized)
- Token-level F1 (SQuAD-style unigram overlap)
- ROUGE-L (requires rouge-score package; optional)
- BLEU and chrF (requires sacrebleu; optional)

All metrics are computed per sample and averaged per question type.
"""
try:
    from rouge_score import rouge_scorer as _rouge_scorer
except Exception:
    _rouge_scorer = None

try:
    import sacrebleu as _sacrebleu
except Exception:
    _sacrebleu = None

# Optional: SBERT cosine similarity for descriptive QA
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer  # type: ignore
except Exception:
    _SentenceTransformer = None

import re

_WORD_RE = re.compile(r"[\w']+")

def _simple_tokenize(s: str) -> List[str]:
    if not s:
        return []
    return _WORD_RE.findall(s.lower())

def _exact_match(a: str, b: str) -> float:
    return 1.0 if _normalize_text(a) == _normalize_text(b) and _normalize_text(a) != '' else 0.0

def _token_f1(a: str, b: str) -> float:
    ta, tb = _simple_tokenize(a), _simple_tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, len(ta))
    rec = overlap / max(1, len(tb))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def _rouge_l(a: str, b: str) -> Optional[float]:
    if _rouge_scorer is None:
        return None
    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    try:
        scores = scorer.score(b, a)  # reference, hypothesis order for rouge-score
        return float(scores["rougeL"].fmeasure)
    except Exception:
        return None

def _bleu(a: str, b: str) -> Optional[float]:
    # a: hypothesis/pred, b: reference
    if _sacrebleu is None:
        return None
    try:
        return float(_sacrebleu.sentence_bleu(a, [b]).score) / 100.0
    except Exception:
        return None

def _chrf(a: str, b: str) -> Optional[float]:
    if _sacrebleu is None:
        return None
    try:
        return float(_sacrebleu.sentence_chrf(a, [b]).score) / 100.0
    except Exception:
        return None

# SBERT cosine similarity (scaled to [0,1]); lazy-load model on CPU
_SBERT_MODEL = None

def _get_sbert_model():
    global _SBERT_MODEL
    if _SentenceTransformer is None:
        return None
    if _SBERT_MODEL is None:
        name = os.environ.get('SBERT_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        try:
            _SBERT_MODEL = _SentenceTransformer(name, device='cpu')
        except Exception:
            return None
    return _SBERT_MODEL

def _sbert_cosine(a: str, b: str) -> Optional[float]:
    model = _get_sbert_model()
    if model is None:
        return None
    a = a or ''
    b = b or ''
    if not a and not b:
        return 1.0
    try:
        emb = model.encode([a.strip(), b.strip()], normalize_embeddings=True, convert_to_numpy=True)
        import numpy as np  # local import to avoid hard dep if not used
        sim = float((emb[0] * emb[1]).sum())  # cosine since normalized
        sim01 = 0.5 * (sim + 1.0)  # map [-1,1] → [0,1]
        return max(0.0, min(1.0, sim01))
    except Exception:
        return None

# Modify compute_text_metrics to apply normalization before metric tokenization
old_compute_text_metrics = compute_text_metrics if 'compute_text_metrics' in globals() else None

def compute_text_metrics(pred: str, ref: str) -> Dict[str, Optional[float]]:  # type: ignore[override]
    pred_n = _normalize_text(pred)
    ref_n = _normalize_text(ref)
    return {
        'exact_match': _exact_match(pred_n, ref_n),
        'token_f1': _token_f1(pred_n, ref_n),
        'rougeL_f1': _rouge_l(pred_n, ref_n),
        'bleu': _bleu(pred_n, ref_n),
        'chrf': _chrf(pred_n, ref_n),
        'sbert_sim': _sbert_cosine(pred_n, ref_n),
    }


def evaluate_split(
    dataset_folder: str,
    split_json: str,
    out_dir: str,
    model_name: str,
    model_url: str,
    enable_llm_eval: bool = False,
    llm_url: str = None,
    llm_model_name: str = None,
    robot_profile: Optional[str] = None,
    yolo_weights: Optional[str] = None,
    ovd_backend: str = "auto",
    enable_3d_graph: bool = True,
    enable_final_narrative: bool = True,
    enable_tatc: bool = True,
    enable_bev_maps: bool = True,
    dump_htatc: bool = False,
    ablate: Optional[List[str]] = None,
    force_bimanual_tokens: bool = False,
    narrative_focus_locating: bool = True,
    seed: Optional[int] = None
):
    # Seed everything early (includes keyframe selection randomness etc.)
    set_global_seed(seed)  # best-effort
    _ensure_dir(out_dir)
    adapter = get_adapter(model_name)
    print(f"Using adapter for model {model_name}")
    robot_prof = RobotProfile.load(robot_profile)
    print(f"Using robot profile: {robot_prof.as_prompt_text()[:80]} ...")
    with open(split_json, 'r') as f:
        annos_per_video = json.load(f)
    print(f"[SPLIT] {os.path.basename(split_json)} | videos: {len(annos_per_video)}")  # annos schema

    enable_groups = {k: True for k in ['ROBOT','PLAN','SCENE3D','KEYFRAMES']}
    if ablate:
        for k in ablate:
            k = k.strip().upper()
            if k in enable_groups:
                enable_groups[k] = False

    stats_data: List[Dict[str,Any]] = []
    results_data: Dict[str, Dict[str, List[Dict[str,Any]]]] = {}
    htatc_dump_rows: List[Dict[str,Any]] = []

    for video_id, video_dict in tqdm(annos_per_video.items(), desc="videos", unit="vid"):
        vpath = os.path.join(dataset_folder, video_dict['video'])
        task_name = video_dict.get('task', 'unknown')
        annos = video_dict['annos']

        with Timer(f"open:{video_id}"):
            import cv2
            cap = cv2.VideoCapture(vpath); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); ret, frame0 = cap.read(); cap.release()
            H,W = (frame0.shape[0], frame0.shape[1]) if ret else (512,512)

        # Unified keyframe selection (replaces event/evidence distinction)
        with Timer(f"keyframes:{video_id}"):
            selector = KeyframeSelector(strategy='motion', max_keyframes=5, stride=2)
            keyframes = selector.select(vpath)
        if not keyframes:
            # fallback: center frame
            keyframes = [
                # simple placeholder when selection fails
                type('KFPlaceholder',(object,),{'time_sec':(nframes/(2*fps)), 'frame_idx':int(nframes/2)})()
            ]
        keyframe_times = [kf.time_sec for kf in keyframes if hasattr(kf,'time_sec')]
        print(f"[VIDEO] {video_id} | task: {task_name} | fps: {fps:.1f} | frames: {nframes} | keyframes: {keyframe_times}")

        cam = CameraIntrinsics.from_image_size(W, H, fov_deg=180.0/(math.pi))
        ovdet = OpenVocabDetector(backend=ovd_backend, yolo_weights=yolo_weights)

        # Use DepthEstimator with defaults; callers can modify this function signature if they need to pass
        # pred_is_inverse or scale_pred. By default we treat predictions as inverse and use scale=1.0.
        depth_est = DepthEstimator(pred_is_inverse=True) if enable_3d_graph else None

        g_agg = GlobalSceneGraphAggregator()
        local_graph_json: List[Dict[str,Any]] = []

        bev_images_per_keyframe = []  # parallel list aligned with keyframe_times
        with Timer(f"3dsg:{video_id}"):
            for t in keyframe_times[:5]:
                fr = extract_frame_at_time(vpath, t)
                dets = ovdet.detect(fr, text_queries=[
                    "robot arm .","gripper .","tool .","cup .","mug .","bottle .","knife .","can .","fork .","spoon .",
                    "microwave .","cube .","cylinder ."
                ])
                if enable_3d_graph and depth_est is not None:
                    d = depth_est.predict(fr)
                    dmin, dmax, dmean = float(d.min()), float(d.max()), float(d.mean())
                    frame_idx = int(t * fps)
                    g_local = build_local_graph3d(dets, d, cam, t_sec=t, frame_idx=frame_idx)
                    local_graph_json.append(graph3d_to_json(g_local))
                    g_agg.update_from_local(g_local, CameraExtrinsics.identity())
                    # Render BEV map for this keyframe
                    if enable_bev_maps:
                        try:
                            from ..perception.bev import render_bev
                            bev_img = render_bev(g_local, g_agg, size=256)
                            # Attach to matching Keyframe object (by time proximity)
                            for kf in keyframes:
                                if abs(kf.time_sec - t) < 1e-3 and hasattr(kf, 'bev_image'):
                                    kf.bev_image = bev_img
                                    break
                            # # Save individual BEV image for traceability
                            # save_path_single = os.path.join(out_dir, f"{video_id}_bev_t{t:.2f}.png")
                            # try:
                            #     import cv2 as _cv_single
                            #     _cv_single.imwrite(save_path_single, bev_img)
                            # except Exception:
                            #     pass
                        except Exception as e:
                            print(f"[BEV] error rendering BEV: {e}")
                            bev_img = None
                        bev_images_per_keyframe.append(bev_img)
                else:
                    # Fallback 2D-only JSON representation (rounded floats)
                    objs = []
                    for i,dv in enumerate(dets[:12]):
                        x1,y1,x2,y2 = dv['bbox']
                        objs.append({
                            "id": i,
                            "cls": dv['name'],
                            "bbox": [round(float(x1),2),round(float(y1),2),round(float(x2),2),round(float(y2),2)],
                            "score": round(float(dv.get('score',0.0)),2)
                        })
                    local_graph_json.append({
                        "frame_time": round(float(t),2),
                        "frame_idx": int(t*fps),
                        "objects": objs,
                        "relations": []
                    })
                    bev_images_per_keyframe.append(None)

            global_scene_json_str = g_agg.to_json_str(max_tracks=15, max_rels=30)
            global_scene_text = global_scene_json_str  # pass JSON string into HTATC
            print(f"[3DSG] {video_id} | local_json: {len(local_graph_json)} | global tracks={len(g_agg.tracks)}")

        plan_steps = []
        if 'Task planning' in annos:
            tp_q = annos['Task planning'][0]['value']
            with Timer(f"task_planning:{video_id}"):
                frames_tp = [extract_frame_at_time(vpath, t) for t in keyframe_times]
                resp = asyncio.run(adapter.qa_with_images_and_context(model_name, model_url, frames_tp, tp_q, robot_prof.as_prompt_text()))
                lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
                # Extract numbered steps like "1. do X"; fallback to truncation if none
                plan_steps = [ln.split('.',1)[1].strip() for ln in lines if ln[:2].isdigit() and '.' in ln[:3]]
                if not plan_steps and resp:
                    plan_steps = [resp[:200]]

        # Build HTATC context (structured)
        htatc = ""
        bev_note = ""
        if enable_tatc:
            # Build base HTATC for QA
            htatc = build_htatc(
                robot_prof.as_prompt_text(),
                plan_steps,
                global_scene_text,
                keyframe_times,
                motion_contact_line=("motion=" + (';'.join([f"{i}:{'H' if hasattr(kf,'motion_level') and kf.motion_level=='high' else ('M' if hasattr(kf,'motion_level') and kf.motion_level=='medium' else 'L')}" for i,kf in enumerate(keyframes)]) if keyframes and hasattr(keyframes[0],'motion_level') else 'n/a')),
                enable_groups=enable_groups,
                structured=True
            )
            # Prepare a BEV note for narrative 
            if enable_bev_maps and any(img is not None for img in bev_images_per_keyframe):
                bev_note = (
                    "[pseudo Bird's Eye View layout: After ALL RGB keyframes, matching bev schematic top-down layout diagrams follow in the same order. "
                    "Each BEV plots objects in ground plane (X right, Z forward). Dot color = class. Circle radius is proportional to object detection confidence (larger = higher confidence)."
                    "Use RGB first, then BEV for relative spatial layout. It is not to scale.]\n"
                )

        preds_per_q = {}
        for qt, qa in annos.items():
            base_q = qa[0]['value']
            qtext = augment_question(model_name, qt, base_q)
            # Treat Failure locating like standard MCQ for scoring; for storyboard we run a separate locating pass later
            
            # Use only RGB frames for QA prompts
            rgb_frames = []
            for t in keyframe_times:
                # Prefer reusing Keyframe stored image if available
                kf_match = None
                for kf in keyframes:
                    if abs(kf.time_sec - t) < 1e-3:
                        kf_match = kf; break
                if kf_match and getattr(kf_match, 'image', None) is not None:
                    fr_rgb = kf_match.image
                else:
                    fr_rgb = extract_frame_at_time(vpath, t)
                rgb_frames.append(fr_rgb)
            # Append BEV frames after RGB frames for ALL QA tasks
            bev_frames = []
            try:
                bev_frames = [kf.bev_image for kf in keyframes if hasattr(kf, 'bev_image') and kf.bev_image is not None]
            except Exception:
                bev_frames = []
            frames = rgb_frames + bev_frames
            # frames = [extract_frame_at_time(vpath, t) for t in evidence_times]
            # if len(event_times)>=2:
            #     frames += [extract_frame_at_time(vpath, event_times[0]), extract_frame_at_time(vpath, event_times[-1])]
            # Add a BEV note to context when BEV frames exist
            bev_note = ""
            if bev_frames:
                bev_note = (
                    "[pseudo Bird's Eye View layout: After ALL RGB keyframes, matching bev schematic top-down layout diagrams follow in the same order. "
                    "Each BEV plots objects in ground plane (X right, Z forward). Dot color = class. Circle radius is proportional to object detection confidence (larger = higher confidence)."
                    "Use RGB first, then BEV for relative spatial layout. It is not to scale.]\n"
                )
            ctx = (bev_note + htatc) if bev_note else htatc
            with Timer(f"qa:{qt}:{video_id}"):
                resp = asyncio.run(adapter.qa_with_images_and_context(model_name, model_url, frames, qtext, ctx))
            qt_lower = qt.lower()
            if qt_lower.startswith("failure identification") or qt_lower.startswith("failure locating"):
                lab, _ = parse_mcq_label(resp)
                preds_per_q[qt] = lab or resp
            else:
                preds_per_q[qt] = resp
            if dump_htatc:
                # store context length for later consolidation
                htatc_dump_rows.append({
                    'video_id': video_id,
                    'question_type': qt,
                    'htatc': htatc,
                    'groups': enable_groups,
                    'htatc_len_chars': len(htatc),
                    'htatc_len_words': len(htatc.split())
                })

        # Separate locating pass for storyboard/narrative: ask for up to 3 candidate frames among keyframe_times
        # Build a locating prompt referencing indices 0..len(keyframe_times)-1
        if enable_final_narrative and keyframe_times:
            loc_q = build_failure_locating_prompt()
            # Provide a compact context: HTATC plus a small legend of candidate indices and times
            legend = "Candidates: " + ", ".join([f"{i}:{t:.2f}s" for i,t in enumerate(keyframe_times)])
            # Locating pass also uses sequential ordering: all RGB then BEV diagrams
            frames_loc_rgb = [extract_frame_at_time(vpath, t) for t in keyframe_times]
            frames_loc_bev = []
            if enable_bev_maps:
                for bev_im in bev_images_per_keyframe:
                    if bev_im is not None:
                        frames_loc_bev.append(bev_im)
            frames_loc = frames_loc_rgb + frames_loc_bev
            if bev_note:
                locating_context = bev_note + htatc
            else:
                locating_context = htatc
            with Timer(f"qa:locating_storyboard:{video_id}"):
                loc_resp = asyncio.run(adapter.qa_with_images_and_context(model_name, model_url, frames_loc, loc_q + "\n" + legend, locating_context))
            cands = parse_failure_candidates(loc_resp)
            # Map candidate indices to times and confidences; take top up to 3
            cand_times: List[Tuple[float,float]] = []  # (time, conf)
            cand_list_records: List[Dict[str, float]] = []
            for c in cands[:3]:
                idx = max(0, min(int(c.get('frame_num', 0)), len(keyframe_times)-1))
                t_c = float(keyframe_times[idx])
                conf_c = float(c.get('confidence', 0.0))
                cand_times.append((t_c, conf_c))
                cand_list_records.append({'time_sec': t_c, 'confidence': conf_c})
            if cand_times:
                # Rebuild HTATC with evidence focused around top candidate time (middle of montage), keep 3 frames around it
                # Create three evidence timestamps: +/- 1.0s around best candidate
                t_best = cand_times[0][0]
                evidence_times = [max(0.0, t_best-1.0), t_best, t_best+1.0]
                focus_htatc = None
                if enable_tatc and narrative_focus_locating:
                    focus_htatc = build_htatc(
                        robot_prof.as_prompt_text(),
                        plan_steps,
                        global_scene_text,
                        evidence_times,  # narrowed keyframes for narrative focus
                        motion_contact_line="motion_focus=t_best",
                        enable_groups=enable_groups,
                        structured=True
                    )
                # Store candidate times and confidences for montage labeling
                cand_times_sorted = sorted(cand_times, key=lambda x: x[0])
            else:
                cand_times_sorted = []
                cand_list_records = []

        if enable_final_narrative:
            with Timer(f"final_narrative:{video_id}"):
                # Collect ALL keyframes as a list (chronological) for broader context
                all_rgb_frames: List[Any] = []
                all_rgb_labels: List[str] = []  # labels kept for optional saving/diagnostics
                for t in keyframe_times:
                    fr = extract_frame_at_time(vpath, t)
                    all_rgb_frames.append(fr)
                    all_rgb_labels.append(f"t={t:.2f}")
                # Optionally save a montage for traceability, though the model will receive individual frames
                try:
                    all_kf_montage = montage_1xN(all_rgb_frames, labels=all_rgb_labels)
                    save_montage_image(all_kf_montage, os.path.join(out_dir, f"{video_id}_storyboard_all_keyframes.jpg"))
                except Exception:
                    pass

                # Use evidence_times if created (candidate focusing) for a compact candidate montage; otherwise, pick top 3 candidates
                candidate_frames = []
                candidate_labels: List[str] = []
                selected_candidate_times: List[float] = []  # for BEV montage alignment
                if 'cand_times_sorted' in locals() and cand_times_sorted:
                    top_cands = cand_times_sorted[:3]
                    for t_cand, conf in top_cands:
                        candidate_frames.append(extract_frame_at_time(vpath, t_cand))
                        candidate_labels.append(f"t={t_cand:.2f} (conf={conf:.2f})")
                        selected_candidate_times.append(t_cand)
                elif keyframe_times:
                    # Fallback: use mid three keyframes for temporal center
                    mid = len(keyframe_times)//2
                    sel_idx = sorted(set([max(0, mid-1), mid, min(len(keyframe_times)-1, mid+1)]))
                    for i in sel_idx:
                        t = keyframe_times[i]
                        candidate_frames.append(extract_frame_at_time(vpath, t))
                        candidate_labels.append(f"t={t:.2f}")
                        selected_candidate_times.append(t)

                candidates_montage = None
                if candidate_frames:
                    candidates_montage = montage_1xN(candidate_frames, labels=candidate_labels)
                    save_montage_image(candidates_montage, os.path.join(out_dir, f"{video_id}_storyboard_candidates.jpg"))
                # Separate BEV-only montage that MATCHES candidate montage timestamps
                if enable_bev_maps and bev_images_per_keyframe:
                    bev_times = list(selected_candidate_times)
                    bev_frames_sub = []
                    bev_labels_sub = []
                    for t in bev_times:
                        if keyframe_times:
                            idx_near = min(range(len(keyframe_times)), key=lambda i: abs(keyframe_times[i]-t))
                            if idx_near < len(bev_images_per_keyframe):
                                bev_img = bev_images_per_keyframe[idx_near]
                                if bev_img is not None:
                                    bev_frames_sub.append(bev_img)
                                    bev_labels_sub.append(f"BEV t={keyframe_times[idx_near]:.2f}s")
                    if bev_frames_sub:
                        bev_montage_img = bev_montage_1xN(bev_frames_sub, labels=bev_labels_sub)
                        save_bev_montage_image(bev_montage_img, os.path.join(out_dir, f"{video_id}_storyboard_bev.jpg"))
                
                # Build narrative context: include robot profile and global scene graph text to aid grounding
                narrative_context = ""
                try:
                    narrative_context = "\n".join([
                        "[ROBOT PROFILE]", robot_prof.as_prompt_text(),
                        "", "[GLOBAL SCENE GRAPH]", global_scene_text if 'global_scene_text' in locals() else "",
                    ])
                except Exception:
                    narrative_context = robot_prof.as_prompt_text()
                # Call new summarizer API with both montages and candidate list
                narrative_txt = asyncio.run(
                    generate_final_narrative(
                        adapter,
                        model_name,
                        model_url,
                        all_rgb_frames,
                        candidates_montage,
                        robot_prof.as_prompt_text(),
                        plan_steps,
                        htatc if enable_tatc else "",
                        intended_task=task_name,
                        failure_detection=str(preds_per_q.get('Failure detection', '')),
                        failure_identification=str(preds_per_q.get('Failure identification', '')),
                        failure_candidates=(cand_list_records if 'cand_list_records' in locals() else None),
                        # extra_context=narrative_context,
                        extra_context="",
                    )
                )
                with open(os.path.join(out_dir, f"{video_id}_final_narrative.txt"), 'w') as f:
                    f.write(narrative_txt)

        if task_name not in results_data:
            results_data[task_name] = {}
        for qt, pred in preds_per_q.items():
            if qt not in results_data[task_name]:
                results_data[task_name][qt] = []
            # Prepare entry and attach text metrics for descriptive QAs
            entry: Dict[str, Any] = {
                'id': video_id,
                'video': video_dict['video'],
                'conversation': annos[qt],
                'pred': pred
            }
            # Identify descriptive QA types (exclude MCQs scored elsewhere)
            if qt not in ['Failure detection','Failure identification','Failure locating']:
                try:
                    ref_raw = annos[qt][-1]['value'] if annos[qt] else ''
                    entry['metrics'] = compute_text_metrics(str(pred), str(ref_raw))
                except Exception:
                    entry['metrics'] = None
            results_data[task_name][qt].append(entry)

        # Write local graphs JSONL
        # with open(os.path.join(out_dir, f"{video_id}_local3d.jsonl"), 'w') as f:
        #     for rec in local_graph_json:
        #         f.write(json.dumps(rec) + "\n")

    if dump_htatc and htatc_dump_rows:
        with open(os.path.join(out_dir, 'htatc_dump.jsonl'), 'w') as f:
            for r in htatc_dump_rows:
                f.write(json.dumps(r) + "\n")

    for task in results_data:
        for qt in results_data[task]:
            with open(os.path.join(out_dir, f"{task}_{qt}_results.json"), 'w') as f:
                json.dump(results_data[task][qt], f, indent=2)

    stats_data = []
    FAILURE_MCQS = ['Failure detection','Failure identification','Failure locating']
    for task in results_data:
        for qt, entries in results_data[task].items():
            if qt in FAILURE_MCQS:
                correct = 0
                for e in entries:
                    ref_raw = e['conversation'][-1]['value']
                    pred_raw = e['pred']
                    ref = _normalize_text(ref_raw)
                    pred = _normalize_text(str(pred_raw))
                    # Lenient MCQ match: first token equality OR normalized containment
                    ref_tok = _first_token(ref)
                    pred_tok = _first_token(pred)
                    if ref_tok and pred_tok and ref_tok == pred_tok:
                        matched = True
                    else:
                        matched = bool(ref and pred and (ref in pred or pred in ref))
                    correct += 1 if matched else 0
                stats_data.append({
                    'task': task,
                    'question_type': qt,
                    'score_overall': (correct/len(entries))*100.0 if entries else 0.0,
                    'num_qa': len(entries),
                    'scoring': 'normalized'
                })
            else:
                # Aggregate descriptive QA metrics (mean over available entries)
                agg = {'exact_match': 0.0, 'token_f1': 0.0, 'rougeL_f1': 0.0, 'bleu': 0.0, 'chrf': 0.0, 'sbert_sim': 0.0}
                counts = {k: 0 for k in agg}
                for e in entries:
                    m = e.get('metrics') or {}
                    for k in agg:
                        v = m.get(k, None)
                        if isinstance(v, (int, float)):
                            agg[k] += float(v)
                            counts[k] += 1
                # Compute averages only over present values
                avg = {k: (agg[k]/counts[k] if counts[k] > 0 else None) for k in agg}
                stats_entry = {
                    'task': task,
                    'question_type': qt,
                    'num_qa': len(entries),
                    'metrics': avg,
                    'scoring': 'non-LLM:text-sim (EM,F1,ROUGE-L,BLEU,chrF)'
                }
                stats_data.append(stats_entry)

    with open(os.path.join(out_dir, 'stats_data.json'), 'w') as f:
        json.dump(stats_data, f, indent=2)

    return stats_data



def evaluate_dir(dataset_folder: str, test_dir: str, out_root: str, model_name: str, model_url: str, enable_llm_eval: bool=False, llm_url: str=None, llm_model_name: str=None, robot_profile: Optional[str]=None, yolo_weights: str=None, ovd_backend: str='auto', enable_3d_graph: bool=False, enable_final_narrative: bool=False, enable_tatc: bool=True, enable_bev_maps: bool=True, dump_htatc: bool=False, ablate=None, force_bimanual_tokens: bool=False, seed: Optional[int]=None):
    splits = sorted([fn for fn in os.listdir(test_dir) if fn.endswith('.json')])
    merged: Dict[str, Dict[str, Any]] = {}
    for split in splits:
        out_dir = os.path.join(out_root, split.split('.')[0])
        os.makedirs(out_dir, exist_ok=True)
        stats = evaluate_split(dataset_folder, os.path.join(test_dir, split), out_dir, model_name, model_url, enable_llm_eval, llm_url, llm_model_name, robot_profile, yolo_weights, ovd_backend=ovd_backend, enable_3d_graph=enable_3d_graph, enable_final_narrative=enable_final_narrative, enable_tatc=enable_tatc, enable_bev_maps=enable_bev_maps, dump_htatc=dump_htatc, ablate=ablate, force_bimanual_tokens=force_bimanual_tokens, seed=seed)
        for s in stats:
            key = s['task'] + '/' + s['question_type']
            if key not in merged:
                merged[key] = {k:v for k,v in s.items() if k not in ['task','question_type']}
            else:
                N_old = merged[key]['num_qa']; N_new = s['num_qa']
                for k, v in s.items():
                    if k in ['task', 'question_type', 'num_qa']:
                        continue
                    # Merge nested metrics dicts
                    if isinstance(v, dict):
                        if k not in merged[key] or not isinstance(merged[key][k], dict):
                            merged[key][k] = {}
                        for mk, mv in v.items():
                            if mv is None:
                                continue
                            old_val = merged[key][k].get(mk)
                            if old_val is None:
                                # Initialize only if numeric; else copy as-is
                                merged[key][k][mk] = float(mv) if isinstance(mv, (int, float)) else mv
                            elif isinstance(old_val, (int, float)) and isinstance(mv, (int, float)):
                                merged[key][k][mk] = (float(old_val) * N_old + float(mv) * N_new) / (N_old + N_new)
                            else:
                                # Non-numeric nested value or type mismatch: prefer latest
                                merged[key][k][mk] = mv
                        continue
                    # Initialize if missing
                    if k not in merged[key]:
                        merged[key][k] = v
                        continue
                    # Average numeric scalars only
                    old_scalar = merged[key][k]
                    if isinstance(v, (int, float)) and isinstance(old_scalar, (int, float)):
                        merged[key][k] = (float(old_scalar) * N_old + float(v) * N_new) / (N_old + N_new)
                    else:
                        # For strings or mismatched types (e.g., 'scoring'), keep latest or preserve if identical
                        merged[key][k] = v if v != old_scalar else old_scalar
                merged[key]['num_qa'] = N_old + N_new

    with open(os.path.join(out_root, 'results_merged.json'), 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"[MERGE] wrote merged metrics for {len(merged)} task/question_type pairs to results_merged.json")
    return merged

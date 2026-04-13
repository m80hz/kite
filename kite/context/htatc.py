from typing import List, Tuple, Dict, Optional, Any
import json

def build_htatc(
    robot_profile_text: str,
    plan_steps: List[str],
    scene3d_text_global: str,
    keyframe_times: List[float],
    motion_contact_line: Optional[str] = None,
    track_legend: Optional[List[str]] = None,  # deprecated: now embedded in global scene JSON
    enable_groups: Optional[Dict[str,bool]] = None,
    structured: bool = True
) -> str:
    """Build Hierarchical Task-Aware Tokenized Context (H-TATC).

    If structured=True returns curated multi-section context; otherwise returns compact legacy token stream.

    Parameters
    ---------
    robot_profile_text : str
        Description/capabilities of robot for grounding.
    plan_steps : List[str]
        High-level plan (numbered natural language steps).
    scene3d_text_global : str
        JSON string of global 3D scene graph (now expected). May already contain 'track_legend'.
    keyframe_times : List[float]
        Selected salient frame times in seconds.
    motion_contact_line : Optional[str]
        Preformatted concise motion/contact summary line.
    track_legend : Optional[List[str]] (deprecated)
        Legacy injection path; retained for backward compatibility. If provided and not already
        present in global scene JSON, it's added under global_scene.track_legend.
    structured : bool
        When False emits a lightweight single-line token stream.
    """
    if enable_groups is None:
        enable_groups = {k: True for k in ['ROBOT','PLAN','SCENE3D','KEYFRAMES']}

    if not structured:
        toks: List[str] = []
        if enable_groups.get('ROBOT', True) and robot_profile_text:
            toks.append(f"[ROBOT] {robot_profile_text}")
        if enable_groups.get('PLAN', True) and plan_steps:
            step_strs = [f"[STEP] {i+1}. {s}" for i,s in enumerate(plan_steps[:8])]
            toks.append("[PLAN] " + " ".join(step_strs))
        if enable_groups.get('SCENE3D', True):
            if scene3d_text_global:
                toks.append(scene3d_text_global)
        if enable_groups.get('KEYFRAMES', True):
            if keyframe_times:
                times = ",".join([f"{t:.2f}s" for t in sorted(keyframe_times)[:8]])
                toks.append(f"[KEYFRAMES] {times}")
        return " ".join(toks)

    # Structured mode: build timeline core JSON then merge parsed global scene JSON if available
    timeline_json: Dict[str, Any] = {
        'keyframes_sec': [round(float(t),2) for t in keyframe_times],
    }
    scene_json: Optional[Dict[str, Any]] = None
    if scene3d_text_global:
        try:
            scene_json = json.loads(scene3d_text_global)
        except Exception:
            scene_json = {'raw_scene_text': scene3d_text_global[:400]}
    # Embed track legend unless already present (back-compat path)
    if track_legend:
        try:
            existing = scene_json.get('track_legend') if scene_json else None
        except Exception:
            existing = None
        if not existing:
            if scene_json is None:
                scene_json = {}
            scene_json['track_legend'] = track_legend
    context_json = {
        'timeline': timeline_json,
        'global_scene': scene_json
    }
    # Build dynamic token legend only for tokens used
    legend_parts = ["times in seconds"]
    token_legend = "Token legend: " + "; ".join(legend_parts)
    plan_steps_text = ("\n".join(f"{i+1}. {s}" for i,s in enumerate(plan_steps))) if plan_steps else "(none)"
    motion_line = motion_contact_line.strip() if motion_contact_line else "(motion summary unavailable)"
    header_sections = [
        "=== ROBOT PROFILE ===\n" + robot_profile_text.strip(),
        "=== PLAN STEPS ===\n" + plan_steps_text,
        # Global scene summary text removed (JSON below is canonical machine-readable form)
        "=== MOTION CONTACT TIMELINE ===\n" + motion_line,
        "=== CONTEXT JSON ===\n" + json.dumps(context_json, ensure_ascii=False),
        "=== LEGENDS ===\n" + token_legend
    ]
    return "\n\n".join(header_sections)

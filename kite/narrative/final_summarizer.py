from typing import List, Optional, Tuple, Dict
import numpy as np
from ..qa.adapter import RoboFACAdapter

"""
Updated narrative prompt: receives a list of keyframe images plus an optional candidates montage.
"""

FINAL_PROMPT_TEMPLATE = """
You are a robotic introspection narrator. Write a precise, non-speculative summary of the episode in 6-8 sentences.

Context (only include if provided; do not guess missing items):
- Robot profile: {robot_profile}
- Intended task: {intended_task}
- Hypothesized plan steps: {steps}
- Failure detection summary: {failure_detection}
- Failure identification (label): {failure_identification}
- Failure candidates (time[s], confidence): {failure_candidates_list}
- TATC/context: {tatc}

Evidence:
- Images 1..K: ALL keyframes in chronological order (each shows a different time).
- Image K+1 (the last image): Montage of TOP suspected failure frames with their time and confidence labels.

Start by clearly stating: the intended task and the robot's purpose, what the robot is doing in the video, and since there is a failure in the task, state what happened and approximately when it happened (explicitely refer to keyframes and BEV diagrams and their timestamps).

Requirements:
1) Start by describing the robot based on its profile and state the task it is attempting to perform in the video, and the intended plan steps.
2) Recount the episode progression using the keyframe timestamps on images.
3) Localize the failure using the candidates montage if provided; otherwise infer conservatively from keyframes only. State the most likely failure time (or a short range). If confidence is low or ambiguous, say it is uncertain.
4) Do not invent unseen objects, states, or numbers. If something is unclear, explicitly state that it is unclear.
5) Avoid generic language; cite concrete visual cues from the images (e.g., object interactions, spatial relations) without fabricating details.
6) End with exactly two sentences: one high-level correction and one low-level correction.
7) Append two short correction policies (1-2 bullet-like sentences each) describing concrete system changes to avoid this failure in future runs.
"""

def _format_time_sec(t: float) -> str:
    if t is None or t < 0:
        return "(unknown)"
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"

async def generate_final_narrative(
    adapter: RoboFACAdapter,
    model_name: str,
    model_url: str,
    all_keyframe_images: List[np.ndarray],
    candidates_montage: Optional[np.ndarray],
    robot_profile_text: str,
    steps_list: List[str],
    tatc_text: str,
    intended_task: Optional[str] = None,
    failure_detection: Optional[str] = None,
    failure_identification: Optional[str] = None,
    failure_candidates: Optional[List[Dict[str, float]]] = None,
    confidence_threshold: float = 0.5,
    extra_context: str = "",
) -> str:
    # Best-effort parse of intended task / failure details from TATC if not provided
    def _parse_from_tatc(tatc: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        t_task = None; t_fail = None; t_time = None
        if not tatc:
            return t_task, t_fail, t_time
        txt = tatc.lower()
        # crude heuristics
        for key in ["task:", "intended task:", "goal:"]:
            if key in txt:
                try:
                    seg = txt.split(key, 1)[1].split('\n', 1)[0]
                    t_task = seg.strip()[:120]
                    break
                except Exception:
                    pass
        for key in ["failure:", "failure identification:", "what went wrong:"]:
            if key in txt:
                try:
                    seg = txt.split(key, 1)[1].split('\n', 1)[0]
                    t_fail = seg.strip()[:160]
                    break
                except Exception:
                    pass
        for key in ["failure time:", "time:", "t=", "at "]:
            if key in txt:
                try:
                    seg = txt.split(key, 1)[1].split('\n', 1)[0]
                    # attempt to find seconds
                    import re
                    m = re.search(r"([0-9]+\.?[0-9]*)\s*s(ec)?", seg)
                    if m:
                        t_time = float(m.group(1))
                        break
                except Exception:
                    pass
        return t_task, t_fail, t_time

    # Fill only what we don't already have, but avoid forcing uncertain info
    if intended_task is None or failure_identification is None:
        p_task, p_fail, _ = _parse_from_tatc(tatc_text)
        intended_task = intended_task or p_task
        failure_identification = failure_identification or p_fail

    # Prepare failure candidates list string and a best guess time if confident
    failure_candidates_list = "(none)"
    best_time_str = "(unknown)"
    if failure_candidates:
        # normalize list of dicts: {time_sec: float, confidence: float}
        pairs = []
        for c in failure_candidates:
            try:
                t = float(c.get('time_sec'))
                conf = float(c.get('confidence'))
                pairs.append((t, conf))
            except Exception:
                continue
        if pairs:
            pairs_sorted = sorted(pairs, key=lambda x: (-x[1], x[0]))
            failure_candidates_list = ", ".join([f"{t:.2f}s ({conf:.2f})" for t, conf in pairs_sorted[:5]])
            # choose top if confidently above threshold
            if pairs_sorted[0][1] >= confidence_threshold:
                best_time_str = _format_time_sec(pairs_sorted[0][0])

    steps_txt = "; ".join([f"{i+1}. {s}" for i,s in enumerate(steps_list)]) if steps_list else "(none)"
    prompt = FINAL_PROMPT_TEMPLATE.format(
        robot_profile=robot_profile_text or "(unknown)",
        intended_task=intended_task or "(unspecified)",
        steps=steps_txt,
        failure_detection=(failure_detection or "(none)"),
        failure_identification=(failure_identification or "(unspecified)"),
        failure_candidates_list=failure_candidates_list,
        tatc=tatc_text or "(minimal)"
    )
    # Build image list: include each keyframe image individually, then append candidates montage if provided.
    images: List[np.ndarray] = list(all_keyframe_images) if all_keyframe_images else []
    if candidates_montage is not None:
        images.append(candidates_montage)
    # Submit the narrative prompt as the question; include extra context (e.g., robot profile, scene graph) if provided.
    return await adapter.qa_with_images_and_context(model_name, model_url, images, prompt, extra_context or "")

#!/usr/bin/env python3
import os, json, argparse, re
import cv2
import numpy as np
import matplotlib.pyplot as plt

from kite.video.events import select_event_times
from kite.video.keyframes import extract_frame_at_time, montage_1xN

def find_htatc_for_video(htatc_jsonl: str, video_id: str):
    if not os.path.exists(htatc_jsonl):
        return None
    best = None
    with open(htatc_jsonl, 'r') as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get('video_id') == video_id:
                    # prefer Failure explanation if multiple
                    if r.get('question_type','').lower().startswith('failure explanation'):
                        return r.get('htatc')
                    best = r.get('htatc', best)
            except Exception:
                continue
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_folder', type=str, required=True)
    ap.add_argument('--inputs_csv', type=str, required=True, help='CSV with columns: video_id,rel_video_path,run_dir')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--max_items', type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with open(args.inputs_csv, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'): continue
            parts = [p.strip() for p in ln.split(',')]
            if len(parts) < 3: 
                print('[WARN] bad line:', ln); 
                continue
            rows.append({'video_id':parts[0], 'rel_video':parts[1], 'run_dir':parts[2]})
    rows = rows[:args.max_items]

    for r in rows:
        video_path = os.path.join(args.dataset_folder, r['rel_video'])
        run_dir = r['run_dir']
        vid = r['video_id']
        # narrative
        narrative_path = os.path.join(run_dir, f"{vid}_final_narrative.txt")
        narrative = "(narrative not found)"
        if os.path.exists(narrative_path):
            with open(narrative_path,'r') as f:
                narrative = f.read().strip()
        # htatc
        htatc_path = os.path.join(run_dir, 'htatc_dump.jsonl')
        htatc = find_htatc_for_video(htatc_path, vid) or "(H‑TATC not found)"
        # frames
        evts = select_event_times(video_path, stride=2, topk=5)
        if len(evts)>=3:
            times = [evts[0], evts[len(evts)//2], evts[-1]]
        elif len(evts)>0:
            times = [evts[0]]*3
        else:
            times = [0.0, 0.5, 1.0]
        frames = [extract_frame_at_time(video_path, t) for t in times]
        montage = montage_1xN(frames, labels=['before','failure','after'])

        # make figure
        fig = plt.figure(figsize=(12,8))
        ax = plt.subplot(2,1,1)
        ax.imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
        ax.set_title(f"{vid}  |  {os.path.basename(video_path)}")
        ax2 = plt.subplot(2,2,3)
        ax2.text(0.01, 0.99, "H‑TATC", va='top', ha='left', fontsize=10, fontweight='bold')
        ax2.text(0.01, 0.90, (htatc if len(htatc)<=600 else htatc[:600]+'…'), va='top', ha='left', wrap=True)
        ax2.set_axis_off()
        ax3 = plt.subplot(2,2,4)
        ax3.text(0.01, 0.99, "Final narrative", va='top', ha='left', fontsize=10, fontweight='bold')
        ax3.text(0.01, 0.90, (narrative if len(narrative)<=700 else narrative[:700]+'…'), va='top', ha='left', wrap=True)
        ax3.set_axis_off()
        out_png = os.path.join(args.out_dir, f"qual_{vid}.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print("[OK]", out_png)

if __name__ == '__main__':
    main()

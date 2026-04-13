import os, json, argparse, csv, glob

def summarize_run(run_dir: str):
    stats_path = os.path.join(run_dir, 'stats_data.json')
    htatc_path = os.path.join(run_dir, 'htatc_dump.jsonl')
    if not os.path.exists(stats_path):
        return None
    with open(stats_path,'r') as f:
        stats = json.load(f)
    # map question_type -> score_overall
    metrics = { (s['question_type']): s['score_overall'] for s in stats if 'score_overall' in s }
    # context lengths
    n_rows = 0; sum_chars = 0; sum_words = 0
    if os.path.exists(htatc_path):
        with open(htatc_path,'r') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    sum_chars += int(r.get('htatc_len_chars', len(r.get('htatc',''))))
                    sum_words += int(r.get('htatc_len_words', len(r.get('htatc','').split())))
                    n_rows += 1
                except Exception:
                    continue
    avg_chars = (sum_chars / n_rows) if n_rows>0 else 0
    avg_words = (sum_words / n_rows) if n_rows>0 else 0
    return {
        'run_dir': run_dir,
        'fd_acc': metrics.get('Failure detection', 0.0),
        'fi_acc': metrics.get('Failure identification', 0.0),
        'fl_acc': metrics.get('Failure locating', 0.0),
        'avg_htatc_chars': round(avg_chars,2),
        'avg_htatc_words': round(avg_words,2),
        'num_contexts': n_rows
    }

def parse_group_name(run_dir: str):
    base = os.path.basename(run_dir.rstrip('/'))
    if base.startswith('ablate_'):
        return base.replace('ablate_','').upper()
    if base == 'baseline':
        return 'NONE'
    return base.upper()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='Root folder containing ablation subfolders (ablate_plan, ablate_scene3d, baseline, ...)')
    ap.add_argument('--out_csv', type=str, required=True)
    args = ap.parse_args()

    runs = sorted([d for d in glob.glob(os.path.join(args.root, '*')) if os.path.isdir(d)])
    rows = []
    for run in runs:
        summ = summarize_run(run)
        if summ:
            summ['group_off'] = parse_group_name(run)
            rows.append(summ)

    fieldnames = ['group_off','fd_acc','fi_acc','fl_acc','avg_htatc_chars','avg_htatc_words','num_contexts','run_dir']
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] Wrote {args.out_csv} with {len(rows)} rows.")

if __name__ == '__main__':
    main()
